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

This module only *finds* what a prune decision touches. Three neighbours
own the rest, each split out of this file so it has one reason to change:
`group.PruningGroup` applies the result (rebuild, or mask then compress --
and mutates the model in place; see that module for why), `op_rules.py`
says how a prune passes through add/cat/flatten, and `oneshot.prune_model`
decides what to prune in the first place.

Known limitations, not fixed here:
- Models with data-dependent control flow (dynamic `if`s on tensor values,
  some HuggingFace Transformer forward passes) may fail `torch.fx`'s default
  tracer entirely; pass a model-specific `tracer=` if you have one.
- Elementwise `+ - * /` (as functions, operators or tensor methods) are
  recognized as merges out of the box, and common activations, dropout,
  pooling and resizing as channel-wise functions -- not `torch.maximum`,
  not a custom merge function or activation. Adding one is a
  `register_op_propagator(torch.maximum, propagate_elementwise)` (or
  `propagate_channelwise`) call, not an edit here. A function or method
  nobody registered raises, even if it keeps the channel count: unlike a
  module, a function's shape says too little (`torch.flip(x, [1])` keeps
  every channel and reorders them).
- Channels are dim 1, as in an NCHW conv output. A Linear applied to a
  `[N, T, features]` sequence keeps its features in the last dimension,
  which an elementwise merge or a `cat` does not see as channels.
- `torch.cat` is only understood along the channel dimension, and only one
  of its branches is expected to be pruned at a time -- pruning two
  concatenated branches in the same `get_pruning_group` call can silently
  drop one branch's contribution (the second call to continue past the
  shared `cat` node is a no-op once the first has already visited it).
- Grouped convolutions are only handled in the depthwise special case
  (`groups == in_channels == out_channels`); anything else raises.
- A skip connection's other branch is traced back only through ops that
  keep channels one-to-one (adds, activations, scaling). A branch built
  with `cat`, `chunk` or a flatten raises `NotImplementedError` naming the
  op, rather than guessing how its indices map.
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
`module_rules.register_module_rule` it globally. Every decision about an
operation between layers -- an elementwise merge, a functional activation,
a cat, a flatten -- goes through an op propagator in `op_rules.OP_PROPAGATORS`
(`propagate_elementwise`, `propagate_channelwise`, `propagate_cat`,
`propagate_flatten`), which sees one
walk only through the small `Propagation` interface (`shape`,
`add_output_target` / `add_input_target`, `forward`, `find_producer`); pass
`op_propagators={...}` to one graph, or `register_op_propagator` globally.
Per-graph tables are layered over the global registries, not copied from
them, so a rule registered globally later still reaches an existing graph.
Rebuilding itself is `surgery.py`'s `slice_*` primitives, the same code
`prune_conv_bn`/`prune_ffn_block` use.

Two opt-in additions for blocks the module rules can't describe:

- `special_handlers`: a `{module_type: handler}` map passed to
  `DependencyGraph.__init__`. When `get_pruning_group`'s starting `layer` is
  an instance of a registered type (or a subclass of one), the handler --
  not the module rules above -- decides how to rebuild it (see
  `get_pruning_group` for the handler's signature). This is the extension point for
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
"""
from __future__ import annotations

import itertools
from collections import ChainMap
from typing import Mapping

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata

from .group import PruningGroup, SpecialHandler
from .indices import complement_indices
from .module_rules import MODULE_RULES, ChannelRole, ModuleRule, find_module_rule
from .op_rules import OP_PROPAGATORS, OpPropagator
from .registry import resolve_by_type


class DependencyGraph:
    """Traces `model` with `torch.fx` and answers "if I prune these output
    channels of this layer, what else must change?" as many times as needed.
    One graph covers a whole pruning session: a prune changes channel
    counts, not topology, and when one has resized a parameter or buffer
    since the graph last looked (through this graph or any other route),
    `get_pruning_group` re-traces before walking. It used to keep the shapes
    from its first trace, so after one branch of a `cat` was pruned, the
    next prune of another branch offset its indices by the first branch's
    old width -- the wrong input channels of the layer after the `cat`.

    `example_input` is passed to `torch.fx.passes.shape_prop.ShapeProp` --
    a forward pass used only to record every node's output shape (needed
    for the flatten-boundary and `cat`-offset calculations), not to train or
    evaluate anything. It runs without gradients and in eval mode, each
    submodule's own mode restored afterwards, so it leaves BatchNorm running
    statistics untouched. (It ran in whatever mode the model was in, which
    nudged them once per trace.) The trace itself still sees the model's own
    mode, so a branch that only runs in training -- an auxiliary head -- is
    part of the graph when the model is in training mode.

    `module_rules`, if given, adds to (or overrides) the global
    `module_rules.MODULE_RULES` for this graph only -- see that module for
    how to teach `DependencyGraph` a new layer type. `op_propagators` does
    the same for `op_rules.OP_PROPAGATORS`, the operations a prune is carried
    *through* (add, cat, flatten, ...) -- see `register_op_propagator`. Both
    are layered over the live global registries (a `ChainMap`), so the graph
    sees one rule table either way.

    The graph only holds what's fixed for the model as it currently is; each
    `get_pruning_group` call walks it with a fresh `Propagation`, which holds
    that one walk's state. A layer passed as a module object is looked up
    in the model as it is now, so a module an earlier prune swapped in is
    found too (a lookup table built at trace time didn't know it).
    """

    def __init__(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        tracer: fx.Tracer | None = None,
        special_handlers: Mapping[type, SpecialHandler] | None = None,
        module_rules: Mapping[type, ModuleRule] | None = None,
        op_propagators: Mapping[object, OpPropagator] | None = None,
    ):
        self.model = model
        self.special_handlers = special_handlers or {}
        self.module_rules = ChainMap(dict(module_rules or {}), MODULE_RULES.entries())
        self.op_propagators = ChainMap(dict(op_propagators or {}), OP_PROPAGATORS.entries())
        self._tracer = tracer or fx.Tracer()
        self._example_input = example_input
        self._trace()

    def _tensor_shapes(self) -> list[tuple[str, torch.Size]]:
        """Every parameter's and buffer's name and shape: what a prune
        changes, however it was done."""
        return [(name, t.shape) for name, t in itertools.chain(self.model.named_parameters(), self.model.named_buffers())]

    def _trace(self) -> None:
        """Trace the model as it is now and record every node's output
        shape (see the class docstring for why ShapeProp runs in eval mode)."""
        traced = fx.GraphModule(self.model, self._tracer.trace(self.model))
        modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        try:
            with torch.no_grad():
                ShapeProp(traced).propagate(self._example_input)
        finally:
            for m, training in modes:
                m.training = training

        self._node_by_target: dict[str, fx.Node] = {
            node.target: node for node in traced.graph.nodes if node.op == "call_module"
        }
        self._traced_shapes = self._tensor_shapes()

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
        if self._tensor_shapes() != self._traced_shapes:
            self._trace()  # something was resized since: the recorded channel widths are stale
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

        n_out = find_module_rule(module, self.module_rules).output_weight(module).shape[0]  # a MIXING role means a rule
        prune_idx = complement_indices(n_out, keep_idx)
        group.add_output_target(target, prune_idx)
        Propagation(self, group).forward(node, prune_idx)
        return group

    # -- internals, shared with `Propagation` --------------------------

    def _resolve_name(self, layer: nn.Module | str) -> str:
        if isinstance(layer, str):
            return layer
        name = next((name for name, m in self.model.named_modules() if m is layer), None)
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
        if node.op == "call_module":
            return resolve_by_type(self.op_propagators, type(module))
        if node.op in ("call_function", "call_method"):
            return self.op_propagators.get(node.target)
        return None

    def _shape(self, node: fx.Node) -> torch.Size:
        meta = node.meta.get("tensor_meta")
        if meta is None:
            raise RuntimeError(
                f"no shape recorded for node {node.name!r} -- ShapeProp should have annotated every "
                f"node when DependencyGraph was constructed"
            )
        if not isinstance(meta, TensorMetadata):
            raise NotImplementedError(
                f"node {node.name!r} returns a {type(meta).__name__} of tensors, not one tensor, so "
                f"DependencyGraph can't follow channels through it"
            )
        return meta.shape

    def _is_channel_preserving(self, node: fx.Node) -> bool:
        """True if `node`'s output has the same channel-dimension size as
        every tensor feeding it -- activations, dropout, and pooling all
        qualify, without this module needing to know their class names. Only
        used to decide whether a prune's index set can pass straight through
        a `call_module` node untouched; it says nothing about whether the
        module *owns* weights that need pruning too (modules with a
        `ModuleRule` are handled separately, before this check runs). A node
        that takes or returns a tuple (`chunk`, `split`) never qualifies."""
        inputs = node.all_input_nodes
        if not inputs or not all(isinstance(n.meta.get("tensor_meta"), TensorMetadata) for n in (node, *inputs)):
            return False
        out_shape = self._shape(node)
        if len(out_shape) < 2:
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

    This is also the whole interface an op propagator
    (`register_op_propagator`) is given: `shape(node)` to read a recorded
    shape, `add_output_target` / `add_input_target` to record what must
    shrink, and `forward` / `find_producer` to continue the walk. The graph
    and the group stay private -- a propagator used to be handed both, so
    it could reach the graph's internals or call `group.prune()` halfway
    through a walk."""

    def __init__(self, graph: DependencyGraph, group: PruningGroup):
        self._graph = graph
        self._group = group
        self._visited: set[str] = set()

    def shape(self, node: fx.Node) -> torch.Size:
        """`node`'s output shape, as recorded when the graph was traced."""
        return self._graph._shape(node)

    def add_output_target(self, name: str, idx: torch.Tensor) -> bool:
        """Module `name` loses output channels `idx` (see
        `PruningGroup.add_output_target`)."""
        return self._group.add_output_target(name, idx)

    def add_input_target(self, name: str, idx: torch.Tensor) -> bool:
        """Module `name` loses input channels `idx` (see
        `PruningGroup.add_input_target`)."""
        return self._group.add_input_target(name, idx)

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
        the shortcut/downsample conv when there is one.

        It only walks back through ops that keep channels one-to-one (an
        add, an activation, a scale); anything that moves them -- a `cat`,
        a flatten -- raises `NotImplementedError` naming the op. It used to
        walk through every op as if indices carried over unchanged, so a
        skip branch built with `cat` recorded the wrong channels and failed
        later with a misleading "two branches disagree" error."""
        graph = self._graph
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
                    self.add_output_target(n.target, idx)
                    stack.extend(n.all_input_nodes)
                    continue
                if graph._is_channel_preserving(n):
                    stack.extend(n.all_input_nodes)
                    continue
                raise NotImplementedError(
                    f"can't trace the skip connection back through {n.target!r} ({type(module).__name__})"
                )
            elif n.op in ("call_function", "call_method"):
                if not graph._is_channel_preserving(n):
                    raise NotImplementedError(
                        f"can't trace the skip connection back through {n.name!r} ({n.op}): its output channels "
                        f"aren't its inputs' channels one-to-one, so the pruned indices don't carry over"
                    )
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
        propagator = self._graph._op_propagator(node)
        if propagator is None:
            raise NotImplementedError(
                f"{node.op} {node.target!r} ({node.name}) is downstream of a pruned layer and "
                f"DependencyGraph doesn't know how to propagate through it -- if it keeps channel c as "
                f"channel c, register_op_propagator(that target, propagate_channelwise)"
            )
        propagator(self, source, node, idx)

    def _visit_module(self, source: fx.Node, node: fx.Node, idx: torch.Tensor) -> None:
        module = self._graph._module(node.target)
        role = self._graph._role(node.target, module)
        if role is ChannelRole.MIXING:
            self.add_input_target(node.target, idx)
            return
        if role is ChannelRole.PER_CHANNEL:
            self.add_output_target(node.target, idx)
            self.forward(node, idx)
            return
        propagator = self._graph._op_propagator(node, module)
        if propagator is not None:
            propagator(self, source, node, idx)
        elif self._graph._is_channel_preserving(node):
            self.forward(node, idx)
        else:
            raise NotImplementedError(
                f"{node.target!r} ({type(module).__name__}) is downstream of a pruned layer and "
                f"DependencyGraph doesn't know how to propagate through it"
            )


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
