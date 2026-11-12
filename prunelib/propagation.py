"""
One prune decision's walk through a traced model: `Propagation`, which
`graph.DependencyGraph.get_pruning_group` starts once per call. The graph
owns what's fixed for the model as it currently is -- the `torch.fx` trace,
every node's shape, which module rule or op propagator covers what. This
module owns how one prune is followed through it: where its channels are on
each tensor, which layers must shrink, and when a node the prune reaches
from several directions is done.

This used to be the bottom half of `graph.py`. Following a prune and
tracing a model are different reasons to change the code -- the walk has
changed far more often than the trace -- so they are different modules,
the way `group.py` and `op_rules.py` were split out before. The two classes
are still one mechanism: `Propagation` reads the trace through
`DependencyGraph`'s underscore methods, which nothing else calls.
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.passes.shape_prop import TensorMetadata

from .indices import as_index
from .module_rules import ChannelRole
from .op_rules import propagate_channelwise, propagate_elementwise, shares_channels

if TYPE_CHECKING:  # only for annotations; graph.py imports this module
    from .graph import DependencyGraph
    from .group import PruningGroup


def owns_tensors(module: nn.Module) -> bool:
    """Whether `module` has any parameter or buffer (its submodules'
    included) -- something a prune passing through it might have to
    resize."""
    return any(True for _ in itertools.chain(module.parameters(), module.buffers()))


def _no_rule(name: str, module: nn.Module) -> str:
    kind = type(module).__name__
    return (
        f"{name!r} ({kind}) owns weights or buffers but has no module rule, so DependencyGraph can't tell how a "
        f"prune changes it -- register_module_rule({kind}, ...) to prune it, or register_op_propagator({kind}, "
        f"propagate_channelwise) if none of its tensors depend on the channels"
    )


class Propagation:
    """One prune decision's walk through a `DependencyGraph`: the
    `PruningGroup` it fills in, the nodes it has already passed through,
    and where the pruned channels are on each tensor it has reached.
    `get_pruning_group` starts a fresh one per call.

    This is also the whole interface an op propagator
    (`register_op_propagator`) is given: `shape(node)` and
    `channel_dim(node)` to read a recorded shape and where its channels
    are, `add_output_target` / `add_input_target` to record what must
    shrink, and `forward` / `forward_part` / `find_producer` to continue
    the walk. The graph and the group stay private -- a propagator used to
    be handed both, so it could reach the graph's internals or call
    `group.prune()` halfway through a walk.

    What the walk refuses, rather than guess at (each raised
    `NotImplementedError` naming the module or op):

    - A module called more than once in `forward()`. A prune that reaches
      one call changes the module for every call, and only that call's
      consumers used to be walked: the others kept expecting the old width,
      and the model failed on its next forward pass.
    - A module that owns weights or buffers but has no module rule. It used
      to be passed through whenever its output had its input's channel
      count, so a `GroupNorm`, a `PReLU(C)`, a `ConvTranspose2d(C, C)` or a
      `SyncBatchNorm` kept its old width while everything around it shrank.
      A module that owns no tensors (an activation, pooling, dropout) still
      passes by its shape.
    - A layer that reads its channels from a different dimension than the
      one the pruned channels are on (`ModuleRule.channel_dim`).
    """

    def __init__(self, graph: DependencyGraph, group: PruningGroup):
        self._graph = graph
        self._group = group
        self._visited: set[str] = set()
        self._dims: dict[str, int] = {}  # node name -> its channel dimension, counted from the end
        self._parts: dict[str, tuple[fx.Node, int, list[torch.Tensor]]] = {}  # see forward_part
        self._merged: set[str] = set()
        self._at: fx.Node | None = None  # the node whose consumers are being visited

    # -- what an op propagator sees -------------------------------------

    def shape(self, node: fx.Node) -> torch.Size:
        """`node`'s output shape, as recorded when the graph was traced."""
        return self._graph._shape(node)

    def channel_dim(self, node: fx.Node) -> int:
        """Which dimension of `node`'s output holds the channels this walk
        is following, counted from the end (-1 is the last) so it lines up
        across broadcasting -- index `walk.shape(node)` with it directly.
        Known for every node the walk has carried the prune out of, `source`
        included."""
        try:
            return self._dims[node.name]
        except KeyError:
            raise ValueError(f"the walk hasn't carried this prune out of {node.name!r}") from None

    def add_output_target(self, name: str, idx: torch.Tensor) -> bool:
        """Module `name` loses output channels `idx` (see
        `PruningGroup.add_output_target`)."""
        self._check_single_call(name)
        return self._group.add_output_target(name, idx)

    def add_input_target(self, name: str, idx: torch.Tensor) -> bool:
        """Module `name` loses input channels `idx` (see
        `PruningGroup.add_input_target`)."""
        self._check_single_call(name)
        return self._group.add_input_target(name, idx)

    def forward(self, node: fx.Node, idx: torch.Tensor, dim: int | None = None) -> None:
        """Carry a prune of channels `idx` of `node`'s output into every node
        that consumes it. A node already walked through is skipped.

        `dim` is where `node`'s output holds those channels. By default it's
        where the layer `node` calls keeps them, if `node` calls a module
        with a rule, else the same place as in the tensor the walk arrived
        from (aligned from the end, as broadcasting aligns) -- right for
        anything that keeps the layout. A propagator that moves the channels
        to another dimension passes it (`propagate_flatten` does)."""
        if node.name in self._visited:
            return
        self._visited.add(node.name)
        self._dims[node.name] = self._dim_of(node) if dim is None else self._from_end(node, dim)
        at, self._at = self._at, node
        try:
            for user in list(node.users):
                self._visit(node, user, idx)
        finally:
            self._at = at

    def forward_part(self, node: fx.Node, idx: torch.Tensor) -> None:
        """`forward`, for a node whose output loses channels from several of
        its inputs at once -- a concatenation, where each pruned input
        accounts for its own slice. `idx` is this input's part. The walk
        carries every part on from `node` together, once nothing else in
        the walk can add another: after everything else, nodes earliest in
        the graph first.

        `forward` alone can't do this: it skips a node it has already left,
        so a `cat` two pruned branches reached used to carry on only the
        first one's channels."""
        if node.name in self._merged:
            raise NotImplementedError(
                f"{node.name}: the walk reached it with more pruned channels after it had carried its parts on "
                f"-- a skip connection loops back into it"
            )
        dim = self.channel_dim(self._at)
        _, part_dim, parts = self._parts.setdefault(node.name, (node, dim, []))
        if part_dim != dim:
            raise NotImplementedError(f"{node.name}: its parts arrive with their channels on different dimensions")
        parts.append(as_index(idx))

    def find_producer(self, node: fx.Node, idx: torch.Tensor) -> fx.Node:
        """Walk backward from `node` (the "other" operand of an elementwise
        merge) through channel-preserving ops and PER_CHANNEL modules
        (BatchNorm, depthwise conv -- recording those into `group` as it
        goes, same as the forward walk would) until it finds the MIXING
        module (Conv2d/Linear) that actually produced this branch's channels
        -- e.g. for `out += identity`, this is what finds the block's own
        input-producing conv when `identity` is a bare reference to it, or
        the shortcut/downsample conv when there is one. `node`'s channels
        are where the tensor the walk arrived from has them.

        It only walks back through ops that keep channel `c` as channel `c`:
        a module that owns no tensors and keeps the channel count, a
        function registered with `propagate_channelwise`, or one registered
        with `propagate_elementwise` -- through whichever of its operands
        carry the channels, so a skip branch gated by a `[N, 1, H, W]` map
        leads back to the gated tensor's producer, not the gate's. (It
        followed a merge only if every operand had the same channel count,
        so a gated skip branch was refused.) Anything else raises
        `NotImplementedError` naming the op. It used to walk through every
        op as if indices carried over unchanged, so a skip branch built with
        `cat` recorded the wrong channels and failed later with a misleading
        "two branches disagree" error."""
        graph = self._graph
        dim = self.channel_dim(self._at)
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
                if role is not None:
                    self._expect_channels(n.target, module, self.shape(n), dim)
                if role is ChannelRole.MIXING:
                    return n
                if role is ChannelRole.PER_CHANNEL:
                    self.add_output_target(n.target, idx)
                    stack.extend(n.all_input_nodes)
                    continue
                if owns_tensors(module):
                    raise NotImplementedError(f"can't trace the skip connection back through {_no_rule(n.target, module)}")
                if self._keeps_channels(n, dim):
                    stack.extend(n.all_input_nodes)
                    continue
                raise NotImplementedError(
                    f"can't trace the skip connection back through {n.target!r} ({type(module).__name__})"
                )
            elif n.op in ("call_function", "call_method"):
                propagator = graph._op_propagator(n)
                if propagator is propagate_elementwise:
                    carriers = [i for i in n.all_input_nodes if shares_channels(self.shape(i), dim)]
                    if carriers:
                        stack.extend(carriers)
                        continue
                elif propagator is propagate_channelwise and self._keeps_channels(n, dim):
                    stack.extend(n.all_input_nodes)
                    continue
                raise NotImplementedError(
                    f"can't trace the skip connection back through {n.name!r} ({n.op}): its output channels "
                    f"aren't its inputs' channels one-to-one, so the pruned indices don't carry over"
                )
            elif n.op == "placeholder":
                raise NotImplementedError(
                    f"skip connection traces back to model input {n.name!r} -- no upstream "
                    f"Conv2d/Linear found to couple with the pruned layer"
                )
            elif n.op == "get_attr":
                raise NotImplementedError(
                    f"{n.target!r}, a parameter or buffer used directly in forward(), carries the pruned channels "
                    f"-- DependencyGraph only prunes tensors a module with a rule owns"
                )
            else:
                raise NotImplementedError(f"can't trace the skip connection back through node {n.name} (op={n.op})")
        raise NotImplementedError("no Conv2d/Linear producer found upstream of the skip connection")

    # -- the walk itself --------------------------------------------------

    def _run(self, start: fx.Node, idx: torch.Tensor, dim: int) -> None:
        """Carry a prune of `start`'s output channels `idx` (on `dim`)
        through the whole graph, then every node `forward_part` held back,
        earliest in the graph first -- by then only nodes after it can still
        add a part to a later one."""
        self.forward(start, idx, dim)
        while self._parts:
            name = min(self._parts, key=self._graph._position.__getitem__)
            node, part_dim, parts = self._parts.pop(name)
            self._merged.add(name)
            self.forward(node, torch.cat(parts).unique(), part_dim)

    def _check_single_call(self, name: str) -> None:
        calls = self._graph._calls.get(name, 0)
        if calls > 1:
            raise NotImplementedError(
                f"{name!r} is called {calls} times in forward(): its weights are shared by every call, so a "
                f"prune that reaches one call changes them all, and DependencyGraph follows it from one"
            )

    def _from_end(self, node: fx.Node, dim: int) -> int:
        """`dim` counted from the end of `node`'s output, if its shape was
        recorded (a node that returns a tuple keeps `dim` as given)."""
        if not isinstance(node.meta.get("tensor_meta"), TensorMetadata):
            return dim
        rank = len(self.shape(node))
        if not -rank <= dim < rank:
            raise NotImplementedError(f"{node.name}: no dimension {dim} in its {rank}-D output to carry channels on")
        return dim - rank if dim >= 0 else dim

    def _dim_of(self, node: fx.Node) -> int:
        """Where `node`'s output holds the channels when `forward` isn't
        told (see `forward`)."""
        if node.op == "call_module":
            module = self._graph._module(node.target)
            if self._graph._role(node.target, module) is not None:
                return self._graph._module_dim(node.target, module, len(self.shape(node)))
        if self._at is None:
            raise ValueError(f"forward({node.name!r}) starts the walk, so it needs dim=")
        return self._from_end(node, self.channel_dim(self._at))

    def _expect_channels(self, name: str, module: nn.Module, shape: torch.Size, dim: int) -> None:
        """Raise unless `module` keeps its channels on `dim` of a tensor of
        `shape` (both counted from the end), where the walk has them."""
        want = self._graph._module_dim(name, module, len(shape))
        if want != dim:
            raise NotImplementedError(
                f"{name!r} ({type(module).__name__}) keeps its channels on dim {want + len(shape)} of "
                f"{tuple(shape)}, but the pruned channels are on dim {dim + len(shape)}"
            )

    def _keeps_channels(self, node: fx.Node, dim: int) -> bool:
        """True if `node`'s output has the same width on channel dimension
        `dim` as every tensor feeding it, and the same rank -- activations,
        dropout and pooling all qualify, without this module needing to know
        their class names. Only decides whether a prune's index set can
        pass straight through a node untouched; whether a module *owns*
        weights that need pruning too is checked before this (`owns_tensors`,
        and modules with a `ModuleRule` are handled before that). A node
        that takes or returns a tuple (`chunk`, `split`) never qualifies."""
        inputs = node.all_input_nodes
        if not inputs or not all(isinstance(n.meta.get("tensor_meta"), TensorMetadata) for n in (node, *inputs)):
            return False
        out = self.shape(node)
        if len(out) < -dim:
            return False
        return all(len(self.shape(i)) == len(out) and self.shape(i)[dim] == out[dim] for i in inputs)

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
        graph = self._graph
        module = graph._module(node.target)
        role = graph._role(node.target, module)
        if role is not None:
            self._expect_channels(node.target, module, self.shape(source), self.channel_dim(source))
        if role is ChannelRole.MIXING:
            self.add_input_target(node.target, idx)
            return
        if role is ChannelRole.PER_CHANNEL:
            self.add_output_target(node.target, idx)
            self.forward(node, idx)
            return
        propagator = graph._op_propagator(node, module)
        if propagator is not None:
            propagator(self, source, node, idx)
        elif owns_tensors(module):
            raise NotImplementedError(_no_rule(node.target, module))
        elif self._keeps_channels(node, self.channel_dim(source)):
            self.forward(node, idx)
        else:
            raise NotImplementedError(
                f"{node.target!r} ({type(module).__name__}) is downstream of a pruned layer and "
                f"DependencyGraph doesn't know how to propagate through it"
            )
