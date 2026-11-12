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
problem -- Conv2d/BatchNorm/Linear chains coupled by elementwise merges
(residual/skip, squeeze-and-excitation gates), `torch.cat` (concatenation),
the conv-output-flattened-into-a-Linear boundary, and depthwise convs. It
does **not** attempt automatic discovery of attention blocks -- recognizing
a reshape/matmul/softmax/matmul sequence as "one attention head" is a
different, much harder problem, and attention pruning stays manual via
`prune_attention_heads`. Anything this module doesn't recognize on the way
-- a non-depthwise grouped conv, a reshape that isn't the classifier-head
flatten pattern, a module with weights and no rule -- raises
`NotImplementedError` rather than silently producing wrong output. That's a
deliberate continuation of this library's existing rule (see
`prune_conv_bn`'s and `prune_ffn_block`'s seam-mismatch `ValueError`s): a
loud failure here is cheap, a silently corrupted model is not.

This module traces the model and answers questions about the trace. Four
neighbours own the rest, each split out of this file so it has one reason
to change: `propagation.Propagation` follows one prune decision through the
trace, `group.PruningGroup` applies the result (rebuild, or mask then
compress -- and mutates the model in place; see that module for why),
`op_rules.py` says how a prune passes through add/cat/flatten, and
`oneshot.prune_model` decides what to prune in the first place.

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
- Channels are on the dimension the layer that produced them says
  (`ModuleRule.channel_dim`: dim 1 for a conv, the last for a Linear).
  Nothing moves them: a `permute`/`transpose` into channels-last raises,
  and so does a layer reading another dimension than the one they're on.
- `torch.cat` is only understood along the channel dimension.
- Grouped convolutions are only handled in the depthwise special case
  (`groups == in_channels == out_channels`); anything else raises.
- A skip connection's other branch is traced back only through ops that
  keep channels one-to-one (merges, activations, scaling). A branch built
  with `cat`, `chunk` or a flatten raises `NotImplementedError` naming the
  op, rather than guessing how its indices map.
- The starting layer of a prune must *produce* its channels (an ungrouped
  Conv2d or a Linear -- `ChannelRole.MIXING`). A depthwise conv or a
  BatchNorm only carries channels its producer decided, so starting there
  raises `TypeError`; start from the layer that feeds it.
- A module called more than once in `forward()` (shared weights) can't be
  pruned, and raises.
- A module with no rule is passed through by its shape -- if it keeps the
  channel count -- only when it owns no parameters or buffers (an
  activation, pooling, dropout). One that does raises, naming the fix
  (see `propagation.Propagation`).

Neither the layer types nor the operations understood are hard-coded
here. Every Conv2d/Linear/BatchNorm decision above goes through a
`module_rules.ModuleRule` (its `ChannelRole` -- does a prune stop at this
module or pass through it -- where its channels are, and how to rebuild it
smaller); pass `module_rules={nn.Conv1d: MyConv1dRule()}` to one
`DependencyGraph`, or `module_rules.register_module_rule` it globally. Every
decision about an operation between layers -- an elementwise merge, a
functional activation, a cat, a flatten -- goes through an op propagator in
`op_rules.OP_PROPAGATORS` (`propagate_elementwise`, `propagate_channelwise`,
`propagate_cat`, `propagate_flatten`), which sees one walk only through the
small `Propagation` interface; pass `op_propagators={...}` to one graph, or
`register_op_propagator` globally. Per-graph tables are layered over the
global registries, not copied from them, so a rule registered globally
later still reaches an existing graph. Rebuilding itself is `surgery.py`'s
`slice_*` primitives, the same code `prune_conv_bn`/`prune_ffn_block` use.

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

from collections import ChainMap, Counter
from typing import Mapping

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata

from .group import PruningGroup, SpecialHandler, tensor_layout
from .indices import complement_indices
from .module_rules import MODULE_RULES, ChannelRole, ModuleRule, find_module_rule
from .op_rules import OP_PROPAGATORS, OpPropagator
from .propagation import Propagation
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
    evaluate anything. A tuple is unpacked into positional arguments, for a
    model whose `forward` takes more than one tensor (a two-stream network,
    an input plus a mask); wrap a single tuple-valued argument as
    `(inputs,)`. Only one tensor used to be accepted, so such a model
    couldn't be traced at all. The pass runs without gradients and in eval
    mode, each submodule's own mode restored afterwards, so it leaves
    BatchNorm running statistics untouched. (It ran in whatever mode the
    model was in, which nudged them once per trace.) The trace itself still
    sees the model's own mode, so a branch that only runs in training -- an
    auxiliary head -- is part of the graph when the model is in training
    mode.

    `module_rules`, if given, adds to (or overrides) the global
    `module_rules.MODULE_RULES` for this graph only -- see that module for
    how to teach `DependencyGraph` a new layer type. `op_propagators` does
    the same for `op_rules.OP_PROPAGATORS`, the operations a prune is carried
    *through* (add, cat, flatten, ...) -- see `register_op_propagator`. Both
    are layered over the live global registries (a `ChainMap`), so the graph
    sees one rule table either way.

    The graph only holds what's fixed for the model as it currently is; each
    `get_pruning_group` call walks it with a fresh `propagation.Propagation`,
    which holds that one walk's state. A layer passed as a module object is
    looked up in the model as it is now, so a module an earlier prune
    swapped in is found too (a lookup table built at trace time didn't know
    it).
    """

    def __init__(
        self,
        model: nn.Module,
        example_input: torch.Tensor | tuple[torch.Tensor, ...],
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
        self._example_inputs = example_input if isinstance(example_input, tuple) else (example_input,)
        self._trace()

    def _trace(self) -> None:
        """Trace the model as it is now and record every node's output
        shape (see the class docstring for why ShapeProp runs in eval mode),
        each node's position, and how many times each module is called."""
        traced = fx.GraphModule(self.model, self._tracer.trace(self.model))
        modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        try:
            with torch.no_grad():
                ShapeProp(traced).propagate(*self._example_inputs)
        finally:
            for m, training in modes:
                m.training = training

        nodes = list(traced.graph.nodes)
        self._position = {node.name: i for i, node in enumerate(nodes)}
        calls = [node for node in nodes if node.op == "call_module"]
        self._calls = Counter(node.target for node in calls)
        self._node_by_target: dict[str, fx.Node] = {}
        for node in calls:
            self._node_by_target.setdefault(node.target, node)
        self._traced_layout = tensor_layout(self.model)

    def get_pruning_group(self, layer: nn.Module | str, keep_idx: torch.Tensor) -> PruningGroup:
        """`layer` (a submodule of `model`, or its dotted qualified name)
        loses every output channel *not* listed in `keep_idx`. Returns the
        full `PruningGroup` this implies across the rest of the model --
        nothing is mutated until `.prune()` is called on the result. At
        least one channel must survive: a `keep_idx` that keeps none raises
        `ValueError` (it used to leave a zero-width layer -- a Linear that
        still ran, feeding the next layer nothing).

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
        if tensor_layout(self.model) != self._traced_layout:
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
        if prune_idx.numel() == n_out:
            raise ValueError(f"keep_idx keeps none of the {n_out} output channels of {target!r}")
        walk = Propagation(self, group)
        walk.add_output_target(target, prune_idx)
        walk._run(node, prune_idx, self._module_dim(target, module, len(self._shape(node))))
        return group

    # -- internals, shared with `propagation.Propagation` ----------------

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

    def _module_dim(self, name: str, module: nn.Module, rank: int) -> int:
        """Where `module` (which has a rule) keeps its channels in a
        `rank`-D tensor, counted from the end (`ModuleRule.channel_dim`)."""
        dim = find_module_rule(module, self.module_rules).channel_dim(module)
        if not -rank <= dim < rank:
            raise NotImplementedError(f"{name!r}: its rule puts channels on dim {dim}, but it handles {rank}-D tensors")
        return dim - rank if dim >= 0 else dim

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
