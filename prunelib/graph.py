"""
Generic structural surgery: given a layer and the output indices to keep,
trace the model with `torch.fx` to discover every *other* layer that prune
decision couples to -- a residual/skip-connection partner, a downstream
conv/Linear whose input must shrink to match, a BatchNorm riding along, a
depthwise conv passing indices straight through -- and prune all of them
together. This is what closes the gap `surgery.py`'s functions leave open:
`prune_conv_bn` and `prune_ffn_block` are correct and tested, but every
caller has to already know the seam (pass `next_conv=` explicitly, or know
which two Linears form an FFN block). `vgg.py` only works at all because it
hardcodes that VGG's `.features` is a flat `Sequential`. Nothing before this
module looked at an arbitrary `nn.Module` and worked out its own wiring.

Scope, stated up front: this handles the CNN/MLP channel-pruning dependency
problem -- Conv2d/BatchNorm/Linear chains coupled by elementwise add
(residual/skip), `torch.cat` (concatenation), the conv-output-flattened-into-
a-Linear boundary, and depthwise convs. It does **not** attempt automatic
discovery of attention blocks -- recognizing a reshape/matmul/softmax/matmul
sequence as "one attention head" is a different, much harder problem, and
attention pruning stays manual via `prune_attention_heads`. Anything this
module doesn't recognize on the way -- a non-depthwise grouped conv, a
reshape that isn't the classifier-head flatten pattern, an unrecognized
module type -- raises `NotImplementedError` rather than silently producing
wrong output. That's a deliberate continuation of this library's existing
rule (see `prune_conv_bn`'s and `prune_ffn_block`'s seam-mismatch
`ValueError`s): a loud failure here is cheap, a silently corrupted model is
not.

One departure from the rest of `prunelib` worth flagging explicitly:
`prune_conv_bn` et al. always return *new* modules and never touch the ones
passed in. `PruningGroup.prune()` mutates the model in place instead. That's
not an oversight -- the entire point of this module is that the caller
shouldn't need to know the graph shape well enough to wire new modules back
together themselves; something has to do that wiring, and once it's been
computed here there's nothing left for the caller to reassemble.

Known limitations, not fixed here:
- Models with data-dependent control flow (dynamic `if`s on tensor values,
  some HuggingFace Transformer forward passes) may fail `torch.fx`'s default
  tracer entirely; pass a model-specific `tracer=` if you have one.
- Only `torch.add`/`operator.add`/`operator.iadd` (and the tensor `add`
  methods) are recognized as elementwise merges out of the box -- not
  `torch.sub`, not a custom merge function. Adding one is a
  `register_op_propagator(torch.sub, propagate_add)` call, not an edit here.
- `torch.cat` is only understood along the channel dimension, and only one
  of its branches is expected to be pruned at a time -- pruning two
  concatenated branches in the same `get_pruning_group` call can silently
  drop one branch's contribution (the second call to continue past the
  shared `cat` node is a no-op once the first has already visited it).
- Grouped convolutions are only handled in the depthwise special case
  (`groups == in_channels == out_channels`); anything else raises.
- The starting layer of a prune must *produce* its channels (an ungrouped
  Conv2d or a Linear -- `ChannelRole.MIXING`). A depthwise conv or a
  BatchNorm only carries channels its producer decided, so starting there
  raises `TypeError`; start from the layer that feeds it.
- The "is this module safe to pass indices straight through" check
  (`_is_channel_preserving`, used for activations/dropout/pooling) is
  shape-based, not type-based: a module whose output happens to have the
  same channel-dimension width as its input is treated as transparent even
  if it isn't really a channel-preserving op. This is deliberately
  permissive rather than an exhaustive whitelist of "safe" module types; a
  genuine mismatch will still surface as a shape error the next time the
  model runs a forward pass, just not necessarily at graph-construction time.

Neither the layer types nor the operations understood are hard-coded
here. Every Conv2d/Linear/BatchNorm decision above goes through a
`module_rules.ModuleRule` (its `ChannelRole` -- does a prune stop at this
module or pass through it -- and how to rebuild it smaller); pass
`module_rules={nn.Conv1d: MyConv1dRule()}` to one `DependencyGraph`, or
`module_rules.register_module_rule` it globally. Every add/cat/flatten
decision goes through an op propagator in `OP_PROPAGATORS`
(`propagate_add`, `propagate_cat`, `propagate_flatten`), which sees one
walk only through the small `Propagation` interface; pass
`op_propagators={...}` to one graph, or `register_op_propagator` globally.
Rebuilding itself is `surgery.py`'s `slice_*` primitives, the same code
`prune_conv_bn`/`prune_ffn_block` use.

Three additions on top of the above, all opt-in and none changing existing
behavior:

- `PruningGroup.mask()` / `.commit_and_compress()`: the two-phase
  mask-then-compress workflow from `masking.py`, extended to a whole
  dependency group instead of one layer at a time -- `.mask()` zeroes every
  targeted module's channels via reparametrization (safe to fine-tune
  against, shapes unchanged), `.commit_and_compress()` bakes the zeros in
  and runs the same rebuild `.prune()` does. `.prune()` itself is unchanged
  and still available for one-shot surgery.
- `special_handlers`: a `{module_type: handler}` map passed to
  `DependencyGraph.__init__`. When `get_pruning_group`'s starting `layer` is
  an instance of a registered type (or a subclass of one), the handler --
  not the module rules above -- decides how to rebuild it (see `PruningGroup`'s
  docstring for the handler's signature). This is the extension point for
  attention-block pruning (`surgery.prune_attention_heads` needs four
  Linears rewired together, which doesn't fit the single-module output/input
  index model above) -- deliberately scoped to the *starting* layer only,
  not to a block encountered mid-propagation: translating an arbitrary
  upstream channel prune into a head-pruning decision is the "much harder
  problem" this docstring already disclaims above.
- `LeafTracer`: a custom class isn't a leaf under `fx.Tracer`'s default
  rule (only modules whose class lives in `torch.nn.modules` are), so
  `special_handlers` needs a tracer that says otherwise for whichever type
  you're registering a handler for -- this is that tracer.

`prune_model()`, a module-level function below, is the other addition: a
`vgg.py::prune_vgg_layer`-style one-call convenience (score, select, prune)
that works on any model `DependencyGraph` can trace, not just VGG's flat
`.features` Sequential.
"""
from __future__ import annotations

import operator
from typing import Callable, Union

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp

from .indices import complement_indices, expand_blocks
from .masking import commit_mask, mask_channels
from .module_rules import MODULE_RULES, ChannelRole, ModuleRule, find_module_rule
from .registry import resolve_by_type
from .selection import Selector, select_prune_indices_by_method

SpecialHandler = Callable[[nn.Module, torch.Tensor], nn.Module]
OpPropagator = Callable[["Propagation", fx.Node, fx.Node, torch.Tensor], None]  # (walk, source, node, idx), see register_op_propagator


def _set_submodule(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    """`setattr` a module back onto its parent by dotted path. `nn.Module`
    has no `set_submodule` guaranteed at this package's `torch>=2.0` floor,
    so this is the hand-written equivalent of `get_submodule`."""
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _record_target(targets: dict[str, torch.Tensor], name: str, idx: torch.Tensor) -> bool:
    """Store `idx` under `name`; return True if this is the first time
    `name` has been targeted. A second, *different* index set for the same
    module is a real conflict (two branches disagreeing on what a shared
    layer's surviving channels are) and raises rather than silently keeping
    whichever arrived first."""
    existing = targets.get(name)
    if existing is None:
        targets[name] = idx
        return True
    if existing.numel() == idx.numel() and torch.equal(torch.sort(existing).values, torch.sort(idx).values):
        return False
    raise ValueError(
        f"{name!r} is already targeted for pruning with a different index set than this "
        f"path computed ({sorted(existing.tolist())} vs {sorted(idx.tolist())}) -- two "
        f"branches of the graph disagree on which channels of a shared layer survive"
    )


class PruningGroup:
    """The result of `DependencyGraph.get_pruning_group`: every module whose
    weights must be resized for one prune decision to be structurally valid,
    keyed by qualified name. `output_targets[name]` are indices to remove
    from that module's *output* dimension (dim 0 of its weight);
    `input_targets[name]` from its *input* dimension (dim 1). A module can
    appear in both (a plain conv-BN-conv chain has the middle conv shrink on
    input from the previous layer and needs no output change here; a
    depthwise conv appears only in `output_targets`, by convention, since
    its input and output channel counts are the same dimension).
    `special_targets[name]` is `(handler, keep_idx)` for a starting layer
    whose type was registered in `DependencyGraph`'s `special_handlers` --
    see `.prune()`'s handling of it below and the module docstring above.

    Add targets through `add_output_target`/`add_input_target`, which
    detect two paths disagreeing about the same module, and
    `add_special_target`; `module_rules` (default: the global
    `module_rules.MODULE_RULES`) decides how each targeted module is
    rebuilt."""

    def __init__(self, model: nn.Module, module_rules: dict[type, ModuleRule] | None = None):
        self.model = model
        self.output_targets: dict[str, torch.Tensor] = {}
        self.input_targets: dict[str, torch.Tensor] = {}
        self.special_targets: dict[str, tuple[SpecialHandler, torch.Tensor]] = {}
        self.module_rules = module_rules

    def add_output_target(self, name: str, idx: torch.Tensor) -> bool:
        """Remove output channels `idx` of module `name`. Returns False if
        this exact target was already recorded; raises `ValueError` if a
        *different* one was."""
        return _record_target(self.output_targets, name, idx)

    def add_input_target(self, name: str, idx: torch.Tensor) -> bool:
        """Remove input channels `idx` of module `name` (see
        `add_output_target`)."""
        return _record_target(self.input_targets, name, idx)

    def add_special_target(self, name: str, handler: SpecialHandler, keep_idx: torch.Tensor) -> None:
        """Rebuild module `name` as `handler(module, keep_idx)` on `.prune()`,
        instead of through its module rule."""
        self.special_targets[name] = (handler, keep_idx)

    def prune(self) -> nn.Module:
        """Rebuild every targeted module with the recorded indices removed,
        and write each one back onto `self.model` in place. Returns the
        (mutated) model for convenience."""
        for name in dict.fromkeys([*self.output_targets, *self.input_targets]):
            module = self.model.get_submodule(name)
            rule = find_module_rule(module, self.module_rules)
            if rule is None:
                raise TypeError(f"{name!r}: don't know how to prune a {type(module).__name__}")
            new_module = rule.rebuild(module, self.output_targets.get(name), self.input_targets.get(name))
            _set_submodule(self.model, name, new_module)

        for name, (handler, keep_idx) in self.special_targets.items():
            module = self.model.get_submodule(name)
            new_module = handler(module, keep_idx)
            _set_submodule(self.model, name, new_module)

        return self.model

    def mask(self) -> None:
        """Phase 1 of the two-phase mask-then-compress workflow (see
        `masking.py`'s module docstring), extended here to a whole
        dependency group instead of one layer at a time: zero every
        *output*-target module's pruned channels via reparametrization
        (`masking.mask_channels`) rather than immediately rebuilding smaller
        modules. Shapes are unchanged, so the model keeps running -- gradients
        naturally don't reach the masked positions, so it's safe to fine-tune
        or evaluate with the mask active, across as many groups/iterations as
        you want, before calling `.commit_and_compress()` once ready to
        physically shrink everything this group touches.

        Only `output_targets` are masked. An `input_targets`-only module (a
        downstream conv/Linear whose *input* must shrink to match an
        upstream prune) needs no weight change here -- it already receives
        zero activations from the masked upstream output, so its own surgery
        can wait for `.commit_and_compress()`. Not supported for a group
        with `special_targets` (attention-block handlers act immediately in
        `.prune()`/`.commit_and_compress()`, not via reparametrization).
        """
        for name, idx in self.output_targets.items():
            if idx.numel() == 0:
                continue
            mask_channels(self.model.get_submodule(name), idx)

    def commit_and_compress(self) -> nn.Module:
        """Phase 2: bake every mask `.mask()` applied into real zeros
        (`masking.commit_mask`) on each `output_targets` module, then run
        the same rebuild `.prune()` does. Equivalent to calling `.prune()`
        directly on an unmasked group -- see
        `tests/test_masking.py::test_compress_masked_conv_bn_matches_direct_surgery`
        for the single-layer version of this same equivalence proof -- except
        the model can be fine-tuned with the mask active in between `.mask()`
        and this call. `commit_mask` is a no-op for a module that was never
        masked, so this is also safe to call on a group `.mask()` was never
        called on (identical to plain `.prune()` in that case)."""
        for name in self.output_targets:
            commit_mask(self.model.get_submodule(name))
        return self.prune()


class DependencyGraph:
    """Traces `model` once with `torch.fx` and answers "if I prune these
    output channels of this layer, what else must change?" as many times as
    needed (channel counts change between prunes, but the graph's topology
    and recorded spatial sizes don't, so one trace covers a whole pruning
    session).

    `example_input` is passed to `torch.fx.passes.shape_prop.ShapeProp` --
    a single forward pass used only to record every node's output shape
    (needed for the flatten-boundary and `cat`-offset calculations), not to
    train or evaluate anything. If your model's BatchNorm layers are in
    training mode this will nudge their running statistics like any other
    forward pass would; call `model.eval()` first if that matters to you.

    `module_rules`, if given, adds to (or overrides) the global
    `module_rules.MODULE_RULES` for this graph only -- see that module for
    how to teach `DependencyGraph` a new layer type. `op_propagators` does
    the same for `OP_PROPAGATORS`, the operations a prune is carried
    *through* (add, cat, flatten, ...) -- see `register_op_propagator`.

    The graph only holds what's fixed once traced; each `get_pruning_group`
    call walks it with a fresh `Propagation`, which holds that one walk's
    state.
    """

    def __init__(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        tracer: fx.Tracer | None = None,
        special_handlers: dict[type, SpecialHandler] | None = None,
        module_rules: dict[type, ModuleRule] | None = None,
        op_propagators: dict[object, OpPropagator] | None = None,
    ):
        self.model = model
        self.special_handlers = special_handlers or {}
        self.module_rules = None if module_rules is None else {**MODULE_RULES, **module_rules}
        self.op_propagators = None if op_propagators is None else {**OP_PROPAGATORS, **op_propagators}
        tracer = tracer or fx.Tracer()
        graph = tracer.trace(model)
        traced = fx.GraphModule(model, graph)
        ShapeProp(traced).propagate(example_input)

        self._node_by_target: dict[str, fx.Node] = {
            node.target: node for node in traced.graph.nodes if node.op == "call_module"
        }
        self._name_by_id: dict[int, str] = {id(m): name for name, m in model.named_modules()}

    def get_pruning_group(self, layer: nn.Module | str, keep_idx: torch.Tensor) -> PruningGroup:
        """`layer` (a submodule of `model`, or its dotted qualified name)
        loses every output channel *not* listed in `keep_idx`. Returns the
        full `PruningGroup` this implies across the rest of the model --
        nothing is mutated until `.prune()` is called on the result.

        If `layer`'s type (or a base class of it) is registered in
        `special_handlers` (passed to `__init__`), `keep_idx` is handed to
        that handler unchanged instead of being interpreted as output-channel
        indices -- see the module docstring's `special_handlers` entry. The
        handler is `(module: nn.Module, keep_idx: Tensor) -> nn.Module`: it
        receives `layer` itself and whatever `keep_idx` means for that module
        type (e.g. head indices to keep, for an attention block wired up with
        `surgery.prune_attention_heads`), and returns the module to swap in
        -- the same "return a new module, `PruningGroup` does the swap"
        contract `ModuleRule.rebuild` follows, so the handler is free to
        mutate `module` in place and return it, or build and return a fresh
        one."""
        target = self._resolve_name(layer)
        module = self._module(target)
        group = PruningGroup(self.model, self.module_rules)

        handler = resolve_by_type(self.special_handlers, type(module))
        if handler is not None:
            group.add_special_target(target, handler, keep_idx)
            return group

        node = self._node_by_target.get(target)
        if node is None:
            raise ValueError(f"{target!r} has no call_module node in the traced graph")
        if self._role(target, module) is not ChannelRole.MIXING:
            raise TypeError(
                f"{target!r} is a {type(module).__name__}, expected a layer that produces its own output "
                f"channels (an ungrouped Conv2d or a Linear, or a type registered with a MIXING module rule) "
                f"-- start from the layer that feeds it instead"
            )

        prune_idx = complement_indices(module.weight.shape[0], keep_idx)
        group.add_output_target(target, prune_idx)
        Propagation(self, group).forward(node, prune_idx)
        return group

    # -- internals, shared with `Propagation` --------------------------

    def _resolve_name(self, layer: nn.Module | str) -> str:
        if isinstance(layer, str):
            return layer
        name = self._name_by_id.get(id(layer))
        if name is None:
            raise ValueError(f"{layer!r} is not a submodule of the traced model")
        return name

    def _module(self, name: str) -> nn.Module:
        return self.model.get_submodule(name)

    def _role(self, name: str, module: nn.Module) -> ChannelRole | None:
        """`module`'s `ChannelRole` under its rule, or None if no rule covers
        its type. A rule refusing this particular configuration propagates as
        `NotImplementedError`, prefixed with the module's name."""
        rule = find_module_rule(module, self.module_rules)
        if rule is None:
            return None
        try:
            return rule.role(module)
        except NotImplementedError as exc:
            raise NotImplementedError(f"{name!r}: {exc}") from None

    def _op_propagator(self, node: fx.Node, module: nn.Module | None = None) -> OpPropagator | None:
        """The propagator registered for `node`'s function (`call_function`)
        or method name (`call_method`), or -- for a `call_module` node -- for
        `module`'s class or its nearest registered base class."""
        propagators = OP_PROPAGATORS if self.op_propagators is None else self.op_propagators
        if node.op == "call_module":
            return resolve_by_type(propagators, type(module))
        if node.op in ("call_function", "call_method"):
            return propagators.get(node.target)
        return None

    def _shape(self, node: fx.Node) -> torch.Size:
        meta = node.meta.get("tensor_meta")
        if meta is None:
            raise RuntimeError(
                f"no shape recorded for node {node.name!r} -- ShapeProp should have annotated every "
                f"node when DependencyGraph was constructed"
            )
        return meta.shape

    def _is_channel_preserving(self, node: fx.Node) -> bool:
        """True if `node`'s output has the same channel-dimension size as
        every tensor feeding it -- activations, dropout, and pooling all
        qualify, without this module needing to know their class names. Only
        used to decide whether a prune's index set can pass straight through
        a `call_module` node untouched; it says nothing about whether the
        module *owns* weights that need pruning too (modules with a
        `ModuleRule` are handled separately, before this check runs)."""
        out_shape = self._shape(node)
        inputs = node.all_input_nodes
        if len(out_shape) < 2 or not inputs:
            return False
        for inp in inputs:
            in_shape = self._shape(inp)
            if len(in_shape) != len(out_shape) or in_shape[1] != out_shape[1]:
                return False
        return True


class Propagation:
    """One prune decision's walk through a `DependencyGraph`: the
    `PruningGroup` it fills in, and the nodes it has already passed through.
    `get_pruning_group` starts a fresh one per call.

    This is also everything an op propagator (`register_op_propagator`) is
    given to work with: `shape(node)` to read a recorded shape, `group` to
    record targets into, and `forward` / `find_producer` to continue the
    walk."""

    def __init__(self, graph: DependencyGraph, group: PruningGroup):
        self.graph = graph
        self.group = group
        self._visited: set[str] = set()

    def shape(self, node: fx.Node) -> torch.Size:
        """`node`'s output shape, as recorded when the graph was traced."""
        return self.graph._shape(node)

    def forward(self, node: fx.Node, idx: torch.Tensor) -> None:
        """Carry a prune of channels `idx` of `node`'s output into every node
        that consumes it. A node already walked through is skipped."""
        if node.name in self._visited:
            return
        self._visited.add(node.name)
        for user in list(node.users):
            self._visit(node, user, idx)

    def find_producer(self, node: fx.Node, idx: torch.Tensor) -> fx.Node:
        """Walk backward from `node` (the "other" operand of an elementwise
        add) through channel-preserving ops and PER_CHANNEL modules
        (BatchNorm, depthwise conv -- recording those into `group` as it
        goes, same as the forward walk would) until it finds the MIXING
        module (Conv2d/Linear) that actually produced this branch's channels
        -- e.g. for `out += identity`, this is what finds the block's own
        input-producing conv when `identity` is a bare reference to it, or
        the shortcut/downsample conv when there is one."""
        graph = self.graph
        seen: set[str] = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n.name in seen:
                continue
            seen.add(n.name)

            if n.op == "call_module":
                module = graph._module(n.target)
                role = graph._role(n.target, module)
                if role is ChannelRole.MIXING:
                    return n
                if role is ChannelRole.PER_CHANNEL:
                    self.group.add_output_target(n.target, idx)
                    stack.extend(n.all_input_nodes)
                    continue
                if graph._is_channel_preserving(n):
                    stack.extend(n.all_input_nodes)
                    continue
                raise NotImplementedError(
                    f"can't trace the skip connection back through {n.target!r} ({type(module).__name__})"
                )
            elif n.op in ("call_function", "call_method"):
                stack.extend(n.all_input_nodes)
            elif n.op == "placeholder":
                raise NotImplementedError(
                    f"skip connection traces back to model input {n.name!r} -- no upstream "
                    f"Conv2d/Linear found to couple with the pruned layer"
                )
            else:
                raise NotImplementedError(f"can't trace the skip connection back through node {n.name} (op={n.op})")
        raise NotImplementedError("no Conv2d/Linear producer found upstream of the skip connection")

    def _visit(self, source: fx.Node, node: fx.Node, idx: torch.Tensor) -> None:
        if node.op == "output":
            return
        if node.op == "call_module":
            self._visit_module(source, node, idx)
            return
        propagator = self.graph._op_propagator(node)
        if propagator is None:
            raise NotImplementedError(
                f"{node.op} {node.target!r} ({node.name}) is downstream of a pruned layer and "
                f"DependencyGraph doesn't know how to propagate through it"
            )
        propagator(self, source, node, idx)

    def _visit_module(self, source: fx.Node, node: fx.Node, idx: torch.Tensor) -> None:
        module = self.graph._module(node.target)
        role = self.graph._role(node.target, module)
        if role is ChannelRole.MIXING:
            self.group.add_input_target(node.target, idx)
            return
        if role is ChannelRole.PER_CHANNEL:
            self.group.add_output_target(node.target, idx)
            self.forward(node, idx)
            return
        propagator = self.graph._op_propagator(node, module)
        if propagator is not None:
            propagator(self, source, node, idx)
        elif self.graph._is_channel_preserving(node):
            self.forward(node, idx)
        else:
            raise NotImplementedError(
                f"{node.target!r} ({type(module).__name__}) is downstream of a pruned layer and "
                f"DependencyGraph doesn't know how to propagate through it"
            )


# -- op propagators: how a prune passes through an op that owns no weights --


def propagate_add(walk: Propagation, source: fx.Node, node: fx.Node, idx: torch.Tensor) -> None:
    """An elementwise merge of two branches (a residual connection's `out +
    identity`): channel `c` of the result is channel `c` of both operands,
    so the *other* operand's producer must lose the same channels, and the
    prune continues from both it and the merge. Any elementwise op on two
    same-shape branches couples them this way, e.g.
    `register_op_propagator(torch.sub, propagate_add)`."""
    others = [n for n in node.all_input_nodes if n is not source]
    if len(others) != 1:
        raise NotImplementedError(
            f"{node.name}: elementwise merge with {len(others)} other tensor operand(s) "
            f"(expected exactly 1) is not supported"
        )
    producer = walk.find_producer(others[0], idx)
    walk.group.add_output_target(producer.target, idx)
    walk.forward(producer, idx)
    walk.forward(node, idx)


def propagate_cat(walk: Propagation, source: fx.Node, node: fx.Node, idx: torch.Tensor) -> None:
    """Channel concatenation: `source`'s channels land in the result shifted
    by the widths of the inputs before it."""
    cat_inputs = list(node.args[0])
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    try:
        position = cat_inputs.index(source)
    except ValueError:
        raise NotImplementedError(f"{node.name}: pruned tensor not found among cat's inputs") from None

    ndim = len(walk.shape(source))
    norm_dim = dim if dim >= 0 else dim + ndim
    if norm_dim != 1:
        raise NotImplementedError(f"{node.name}: cat along dim={dim} isn't the channel dimension, not supported")

    offset = sum(int(walk.shape(cat_inputs[i])[norm_dim]) for i in range(position))
    walk.forward(node, idx + offset)


def propagate_flatten(walk: Propagation, source: fx.Node, node: fx.Node, idx: torch.Tensor) -> None:
    """A conv output flattened ahead of a Linear: channel `c` becomes the
    `H * W` consecutive features it owns."""
    shape = walk.shape(source)
    if len(shape) != 4:
        raise NotImplementedError(
            f"{node.name}: flatten/view/reshape of a {len(shape)}D tensor isn't recognized as the "
            f"conv-output-into-classifier-head pattern DependencyGraph handles (only a 4D NCHW "
            f"conv output being flattened ahead of a Linear is supported)"
        )
    spatial = int(shape[2]) * int(shape[3])
    walk.forward(node, expand_blocks(idx, spatial))


# Keyed the way fx records a node's target: the function itself for
# `call_function`, the method name for `call_method`, and -- since a
# `call_module` target is just the submodule's name -- the module class
# (subclasses included) for modules.
OP_PROPAGATORS: dict[object, OpPropagator] = {
    **dict.fromkeys([torch.add, operator.add, operator.iadd, "__add__", "__iadd__", "add", "add_"], propagate_add),
    **dict.fromkeys([torch.cat, torch.concat], propagate_cat),
    **dict.fromkeys([torch.flatten, "view", "reshape", "flatten", nn.Flatten], propagate_flatten),
}


def register_op_propagator(target: object, propagator: OpPropagator, *, overwrite: bool = False) -> OpPropagator:
    """Teach every `DependencyGraph` to carry a prune through `target` -- a
    function (`torch.sub`), a tensor method name (`"sub"`), or a module class
    -- with `propagator(walk, source, node, idx)`, which records into
    `walk.group` and continues with `walk.forward`. Re-registering a target
    raises unless `overwrite=True`; pass `op_propagators={...}` to one
    `DependencyGraph` instead to scope it to that graph."""
    if target in OP_PROPAGATORS and not overwrite:
        raise ValueError(f"a propagator for {target!r} is already registered; pass overwrite=True to replace it")
    OP_PROPAGATORS[target] = propagator
    return propagator


class LeafTracer(fx.Tracer):
    """An `fx.Tracer` that additionally treats every instance of the given
    types as a leaf module -- traced as one `call_module` node instead of
    being decomposed into its own internal ops. `fx.Tracer`'s default
    `is_leaf_module` only says yes for classes that live in `torch.nn.modules`,
    so a custom block class (an attention block, say) gets traced straight
    through by default; that's normally fine, but it means `DependencyGraph`
    never sees it as one addressable node, which is what `special_handlers`
    needs. Use this tracer (naming the type(s) you registered a handler for)
    whenever you pass `special_handlers` to `DependencyGraph`."""

    def __init__(self, leaf_types: tuple[type, ...] | list[type]):
        super().__init__()
        self.leaf_types = tuple(leaf_types)

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, self.leaf_types):
            return True
        return super().is_leaf_module(m, module_qualified_name)


def prune_model(
    model: nn.Module,
    example_input: torch.Tensor,
    layer: nn.Module | str,
    prune_fraction: float,
    method: Union[str, Selector] = "max_k",
    dependency_graph: DependencyGraph | None = None,
    **method_kwargs,
) -> PruningGroup:
    """One-shot generic pruning for an arbitrary model: score `layer`'s
    output channels/neurons (`saliency.compute_score`), keep everything
    except the lowest-scoring `prune_fraction`, and physically resize `layer`
    plus every other module `DependencyGraph` finds it's coupled to -- in one
    call, with no seam (`next_conv=`, which Linears form an FFN block, ...)
    passed in by hand.

    This generalizes what `vgg.py::prune_vgg_layer` does for VGG specifically
    (score -> select -> `surgery.prune_conv_bn`, hardcoding that VGG's
    `.features` is a flat `Sequential`) to any model `DependencyGraph` can
    trace.

    `dependency_graph`, if given, is reused instead of re-tracing `model` --
    pass the same one back in across several `prune_model` calls on the same
    model (the graph's topology doesn't change between prunes, only channel
    counts do, same as a `DependencyGraph` used directly). Otherwise a new
    one is traced from `example_input`.

    `method` is anything `selection.select_prune_indices_by_method` accepts:
    a saliency scorer (`"max_k"`, `"l1"`, `"l2"`, `"random"`), `"kmeans"`
    (cluster channels by weight, prune the lowest-L1 ones within each
    cluster -- see `clustering.kmeans_prune_indices`), any rule registered
    later, or a selector callable; `method_kwargs` are forwarded to it.

    Returns the `PruningGroup` this prune decision implied, already
    `.prune()`d -- inspect `group.output_targets`/`group.input_targets` to
    see what else got touched.
    """
    if not 0.0 <= prune_fraction < 1.0:
        raise ValueError(f"prune_fraction must be in [0, 1), got {prune_fraction}")

    module = model.get_submodule(layer) if isinstance(layer, str) else layer
    if not hasattr(module, "weight"):
        raise TypeError(f"{layer!r} ({type(module).__name__}) has no .weight to score")

    dep = dependency_graph or DependencyGraph(model, example_input)
    n_out = module.weight.shape[0]
    n_to_prune = int(round(prune_fraction * n_out))
    prune_idx = select_prune_indices_by_method(module.weight, n_to_prune, method=method, **method_kwargs)

    group = dep.get_pruning_group(layer, complement_indices(n_out, prune_idx))
    group.prune()
    return group
