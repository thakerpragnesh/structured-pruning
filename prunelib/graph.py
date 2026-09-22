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
- Only `torch.add`/`operator.add`/`operator.iadd` are recognized as
  elementwise merges (not `torch.sub`, not a custom merge function).
- `torch.cat` is only understood along the channel dimension, and only one
  of its branches is expected to be pruned at a time -- pruning two
  concatenated branches in the same `get_pruning_group` call can silently
  drop one branch's contribution (the second call to continue past the
  shared `cat` node is a no-op once the first has already visited it).
- Grouped convolutions are only handled in the depthwise special case
  (`groups == in_channels == out_channels`); anything else raises.
- The "is this module safe to pass indices straight through" check
  (`_is_channel_preserving`, used for activations/dropout/pooling) is
  shape-based, not type-based: a module whose output happens to have the
  same channel-dimension width as its input is treated as transparent even
  if it isn't really a channel-preserving op. This is deliberately
  permissive rather than an exhaustive whitelist of "safe" module types; a
  genuine mismatch will still surface as a shape error the next time the
  model runs a forward pass, just not necessarily at graph-construction time.

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
  an instance of a registered type, the handler -- not the Conv2d/Linear/
  BatchNorm logic above -- decides how to rebuild it (see `PruningGroup`'s
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
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp

from .masking import commit_mask, mask_channels
from .saliency import compute_score, keep_indices

_ADD_FUNCTIONS = {torch.add, operator.add, operator.iadd}
_CAT_FUNCTIONS = {torch.cat, torch.concat}
_ADD_METHODS = {"__add__", "__iadd__", "add", "add_"}
_RESHAPE_METHODS = {"view", "reshape", "flatten"}


def _is_depthwise(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Conv2d)
        and module.groups > 1
        and module.groups == module.in_channels == module.out_channels
    )


def _keep_complement(n: int, prune_idx: torch.Tensor | None) -> torch.Tensor:
    """Indices `0..n-1` not present in `prune_idx`, in ascending order --
    the same vectorized boolean-mask idiom used throughout `prunelib`
    (`saliency.keep_indices`, `vgg.prune_vgg_layer`) instead of a Python
    `set()` + list comprehension."""
    if prune_idx is None:
        return torch.arange(n)
    keep_mask = torch.ones(n, dtype=torch.bool)
    keep_mask[prune_idx.to(torch.long)] = False
    return keep_mask.nonzero(as_tuple=True)[0]


def _set_submodule(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    """`setattr` a module back onto its parent by dotted path. `nn.Module`
    has no `set_submodule` guaranteed at this package's `torch>=2.0` floor,
    so this is the hand-written equivalent of `get_submodule`."""
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _prune_conv(conv: nn.Conv2d, out_prune_idx: torch.Tensor | None, in_prune_idx: torch.Tensor | None) -> nn.Conv2d:
    if _is_depthwise(conv):
        # Depthwise: input and output channel counts are the same dimension
        # by definition (each filter owns exactly one channel), so whichever
        # side supplied a prune target is authoritative for both, and
        # `groups` must shrink to match the new channel count.
        prune_idx = out_prune_idx if out_prune_idx is not None else in_prune_idx
        if prune_idx is None:
            raise ValueError(f"depthwise conv has neither an output nor an input prune target recorded")
        keep = _keep_complement(conv.out_channels, prune_idx)
        new_conv = nn.Conv2d(
            keep.numel(), keep.numel(), kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, dilation=conv.dilation, groups=keep.numel(), bias=conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight.index_select(0, keep))
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias.index_select(0, keep))
        return new_conv

    keep_out = _keep_complement(conv.out_channels, out_prune_idx)
    keep_in = _keep_complement(conv.in_channels, in_prune_idx)
    new_conv = nn.Conv2d(
        keep_in.numel(), keep_out.numel(), kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation, groups=1, bias=conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.index_select(0, keep_out).index_select(1, keep_in))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias.index_select(0, keep_out))
    return new_conv


def _prune_linear(linear: nn.Linear, out_prune_idx: torch.Tensor | None, in_prune_idx: torch.Tensor | None) -> nn.Linear:
    keep_out = _keep_complement(linear.out_features, out_prune_idx)
    keep_in = _keep_complement(linear.in_features, in_prune_idx)
    new_linear = nn.Linear(keep_in.numel(), keep_out.numel(), bias=linear.bias is not None)
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight.index_select(0, keep_out).index_select(1, keep_in))
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias.index_select(0, keep_out))
    return new_linear


def _prune_bn(bn: nn.BatchNorm2d | nn.BatchNorm1d, out_prune_idx: torch.Tensor | None) -> nn.Module:
    keep = _keep_complement(bn.num_features, out_prune_idx)
    new_bn = type(bn)(
        keep.numel(), eps=bn.eps, momentum=bn.momentum, affine=bn.affine,
        track_running_stats=bn.track_running_stats,
    )
    with torch.no_grad():
        if bn.affine:
            new_bn.weight.copy_(bn.weight.index_select(0, keep))
            new_bn.bias.copy_(bn.bias.index_select(0, keep))
        if bn.track_running_stats:
            new_bn.running_mean.copy_(bn.running_mean.index_select(0, keep))
            new_bn.running_var.copy_(bn.running_var.index_select(0, keep))
            new_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new_bn


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
    see `.prune()`'s handling of it below and the module docstring above."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.output_targets: dict[str, torch.Tensor] = {}
        self.input_targets: dict[str, torch.Tensor] = {}
        self.special_targets: dict[str, tuple[Callable[[nn.Module, torch.Tensor], nn.Module], torch.Tensor]] = {}

    def prune(self) -> nn.Module:
        """Rebuild every targeted module with the recorded indices removed,
        and write each one back onto `self.model` in place. Returns the
        (mutated) model for convenience."""
        for name in set(self.output_targets) | set(self.input_targets):
            module = self.model.get_submodule(name)
            out_idx = self.output_targets.get(name)
            in_idx = self.input_targets.get(name)
            if isinstance(module, nn.Conv2d):
                new_module = _prune_conv(module, out_idx, in_idx)
            elif isinstance(module, nn.Linear):
                new_module = _prune_linear(module, out_idx, in_idx)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                new_module = _prune_bn(module, out_idx)
            else:
                raise TypeError(f"{name!r}: don't know how to prune a {type(module).__name__}")
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
    """

    def __init__(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        tracer: fx.Tracer | None = None,
        special_handlers: dict[type, Callable[[nn.Module, torch.Tensor], nn.Module]] | None = None,
    ):
        self.model = model
        self.special_handlers = special_handlers or {}
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

        If `layer`'s type is registered in `special_handlers` (passed to
        `__init__`), `keep_idx` is handed to that handler unchanged instead
        of being interpreted as Conv2d/Linear output-channel indices -- see
        the module docstring's `special_handlers` entry. The handler is
        `(module: nn.Module, keep_idx: Tensor) -> nn.Module`: it receives
        `layer` itself and whatever `keep_idx` means for that module type
        (e.g. head indices to keep, for an attention block wired up with
        `surgery.prune_attention_heads`), and returns the module to swap in
        -- the same "return a new module, `PruningGroup` does the swap"
        contract `_prune_conv`/`_prune_linear`/`_prune_bn` follow, so the
        handler is free to mutate `module` in place and return it, or build
        and return a fresh one."""
        target = self._resolve_name(layer)
        module = self._module(target)

        handler = self.special_handlers.get(type(module))
        if handler is not None:
            group = PruningGroup(self.model)
            group.special_targets[target] = (handler, keep_idx)
            return group

        node = self._node_by_target.get(target)
        if node is None:
            raise ValueError(f"{target!r} has no call_module node in the traced graph")

        if isinstance(module, nn.Conv2d):
            if module.groups > 1 and not _is_depthwise(module):
                raise NotImplementedError(
                    f"{target!r}: grouped conv (groups={module.groups}) that isn't depthwise "
                    f"can't be the starting layer for DependencyGraph"
                )
            out_features = module.out_channels
        elif isinstance(module, nn.Linear):
            out_features = module.out_features
        else:
            raise TypeError(f"{target!r} is a {type(module).__name__}, expected Conv2d or Linear")

        keep_idx = keep_idx.to(torch.long)
        keep_mask = torch.zeros(out_features, dtype=torch.bool)
        keep_mask[keep_idx] = True
        prune_idx = (~keep_mask).nonzero(as_tuple=True)[0]

        group = PruningGroup(self.model)
        group.output_targets[target] = prune_idx
        self._propagate_forward(node, prune_idx, group, visited=set())
        return group

    # -- internals -----------------------------------------------------

    def _resolve_name(self, layer: nn.Module | str) -> str:
        if isinstance(layer, str):
            return layer
        name = self._name_by_id.get(id(layer))
        if name is None:
            raise ValueError(f"{layer!r} is not a submodule of the traced model")
        return name

    def _module(self, name: str) -> nn.Module:
        return self.model.get_submodule(name)

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
        module *owns* weights that need pruning too (BatchNorm/depthwise
        conv are handled separately, before this check runs)."""
        out_shape = self._shape(node)
        inputs = node.all_input_nodes
        if len(out_shape) < 2 or not inputs:
            return False
        for inp in inputs:
            in_shape = self._shape(inp)
            if len(in_shape) != len(out_shape) or in_shape[1] != out_shape[1]:
                return False
        return True

    def _propagate_forward(self, node: fx.Node, idx: torch.Tensor, group: PruningGroup, visited: set[str]) -> None:
        if node.name in visited:
            return
        visited.add(node.name)
        for user in list(node.users):
            self._visit(node, user, idx, group, visited)

    def _visit(self, source: fx.Node, node: fx.Node, idx: torch.Tensor, group: PruningGroup, visited: set[str]) -> None:
        if node.op == "call_module":
            mod = self._module(node.target)
            if isinstance(mod, nn.Conv2d):
                if mod.groups == 1:
                    self._record(group.input_targets, node.target, idx)
                    return
                if _is_depthwise(mod):
                    self._record(group.output_targets, node.target, idx)
                    self._propagate_forward(node, idx, group, visited)
                    return
                raise NotImplementedError(
                    f"{node.target!r}: grouped conv (groups={mod.groups}) that isn't depthwise is not supported"
                )
            if isinstance(mod, nn.Linear):
                self._record(group.input_targets, node.target, idx)
                return
            if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self._record(group.output_targets, node.target, idx)
                self._propagate_forward(node, idx, group, visited)
                return
            if isinstance(mod, nn.Flatten):
                self._handle_flatten(source, node, idx, group, visited)
                return
            if self._is_channel_preserving(node):
                self._propagate_forward(node, idx, group, visited)
                return
            raise NotImplementedError(
                f"{node.target!r} ({type(mod).__name__}) is downstream of a pruned layer and "
                f"DependencyGraph doesn't know how to propagate through it"
            )

        if node.op == "call_function":
            fn = node.target
            if fn in _ADD_FUNCTIONS:
                self._handle_add(source, node, idx, group, visited)
                return
            if fn in _CAT_FUNCTIONS:
                self._handle_cat(source, node, idx, group, visited)
                return
            if fn is torch.flatten:
                self._handle_flatten(source, node, idx, group, visited)
                return
            raise NotImplementedError(
                f"call_function {fn} ({node.name}) is downstream of a pruned layer and "
                f"DependencyGraph doesn't know how to propagate through it"
            )

        if node.op == "call_method":
            if node.target in _RESHAPE_METHODS:
                self._handle_flatten(source, node, idx, group, visited)
                return
            if node.target in _ADD_METHODS:
                self._handle_add(source, node, idx, group, visited)
                return
            raise NotImplementedError(
                f"call_method {node.target!r} ({node.name}) is downstream of a pruned layer and "
                f"DependencyGraph doesn't know how to propagate through it"
            )

        if node.op == "output":
            return

        raise NotImplementedError(
            f"node {node.name} (op={node.op}) is downstream of a pruned layer and "
            f"DependencyGraph doesn't know how to propagate through it"
        )

    def _record(self, targets: dict[str, torch.Tensor], name: str, idx: torch.Tensor) -> bool:
        """Store `idx` under `name`; return True if this is the first time
        `name` has been targeted. A second, *different* index set for the
        same module is a real conflict (two branches disagreeing on what a
        shared layer's surviving channels are) and raises rather than
        silently keeping whichever arrived first."""
        existing = targets.get(name)
        if existing is None:
            targets[name] = idx
            return True
        if existing.numel() == idx.numel() and torch.equal(
            torch.sort(existing).values, torch.sort(idx).values
        ):
            return False
        raise ValueError(
            f"{name!r} is already targeted for pruning with a different index set than this "
            f"path computed ({sorted(existing.tolist())} vs {sorted(idx.tolist())}) -- two "
            f"branches of the graph disagree on which channels of a shared layer survive"
        )

    def _find_producer(self, node: fx.Node, idx: torch.Tensor, group: PruningGroup) -> fx.Node:
        """Walk backward from `node` (the "other" operand of an elementwise
        add) through channel-preserving ops, BatchNorm, and depthwise convs
        (recording those into `group` as it goes, same as the forward walk
        would) until it finds the Conv2d/Linear that actually produced this
        branch's channels -- e.g. for `out += identity`, this is what finds
        the block's own input-producing conv when `identity` is a bare
        reference to it, or the shortcut/downsample conv when there is one."""
        seen: set[str] = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n.name in seen:
                continue
            seen.add(n.name)

            if n.op == "call_module":
                mod = self._module(n.target)
                if isinstance(mod, nn.Conv2d) and mod.groups == 1:
                    return n
                if isinstance(mod, nn.Linear):
                    return n
                if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)) or _is_depthwise(mod):
                    self._record(group.output_targets, n.target, idx)
                    stack.extend(n.all_input_nodes)
                    continue
                if self._is_channel_preserving(n):
                    stack.extend(n.all_input_nodes)
                    continue
                raise NotImplementedError(
                    f"can't trace the skip connection back through {n.target!r} ({type(mod).__name__})"
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

    def _handle_add(self, source: fx.Node, node: fx.Node, idx: torch.Tensor, group: PruningGroup, visited: set[str]) -> None:
        others = [n for n in node.all_input_nodes if n is not source]
        if len(others) != 1:
            raise NotImplementedError(
                f"{node.name}: add with {len(others)} other tensor operand(s) (expected exactly 1) is not supported"
            )
        producer = self._find_producer(others[0], idx, group)
        self._record(group.output_targets, producer.target, idx)
        self._propagate_forward(producer, idx, group, visited)
        self._propagate_forward(node, idx, group, visited)

    def _handle_cat(self, source: fx.Node, node: fx.Node, idx: torch.Tensor, group: PruningGroup, visited: set[str]) -> None:
        cat_inputs = list(node.args[0])
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        try:
            position = cat_inputs.index(source)
        except ValueError:
            raise NotImplementedError(f"{node.name}: pruned tensor not found among cat's inputs") from None

        ndim = len(self._shape(source))
        norm_dim = dim if dim >= 0 else dim + ndim
        if norm_dim != 1:
            raise NotImplementedError(f"{node.name}: cat along dim={dim} isn't the channel dimension, not supported")

        offset = sum(int(self._shape(cat_inputs[i])[norm_dim]) for i in range(position))
        self._propagate_forward(node, idx + offset, group, visited)

    def _handle_flatten(self, source: fx.Node, node: fx.Node, idx: torch.Tensor, group: PruningGroup, visited: set[str]) -> None:
        shape = self._shape(source)
        if len(shape) != 4:
            raise NotImplementedError(
                f"{node.name}: flatten/view/reshape of a {len(shape)}D tensor isn't recognized as the "
                f"conv-output-into-classifier-head pattern DependencyGraph handles (only a 4D NCHW "
                f"conv output being flattened ahead of a Linear is supported)"
            )
        spatial = int(shape[2]) * int(shape[3])
        if idx.numel() == 0:
            expanded = idx.new_empty(0)
        else:
            expanded = torch.cat([torch.arange(c * spatial, (c + 1) * spatial) for c in idx.tolist()])
        self._propagate_forward(node, expanded, group, visited)


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
    method: str = "max_k",
    dependency_graph: DependencyGraph | None = None,
    **score_kwargs,
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
    scores = compute_score(module.weight, method=method, **score_kwargs)
    n_to_prune = int(round(prune_fraction * scores.numel()))
    to_keep = keep_indices(scores, n_to_prune)

    group = dep.get_pruning_group(layer, to_keep)
    group.prune()
    return group
