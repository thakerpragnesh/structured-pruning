"""
How a prune passes through an operation that owns no weights -- the op-side
twin of `module_rules.py`. A module rule says how a *layer type* takes part
in a prune; an op propagator says how a prune is carried *through* an
operation between layers (an elementwise add, a `torch.cat`, a flatten).

A propagator is `(walk, source, node, idx) -> None`: a prune of channels
`idx` of `source`'s output has reached `node`. It reads recorded shapes with
`walk.shape` and where their channels are with `walk.channel_dim`, records
what must shrink with `walk.add_output_target` / `walk.add_input_target`,
and continues the walk with `walk.forward` / `walk.forward_part` /
`walk.find_producer` -- `propagation.Propagation`'s small public interface,
which is all it sees of the walk (not the graph, and not the group being
filled).

Built-in propagators cover elementwise `+ - * /` (`propagate_elementwise`),
activations, dropout, pooling and resizing written as functions or tensor
methods (`propagate_channelwise`), `torch.cat` along the channel dimension
(`propagate_cat`), a conv output flattened ahead of a Linear
(`propagate_flatten`), and reading a tensor's shape (`x.size(0)`), which
carries no channels on. `register_op_propagator(torch.maximum,
propagate_elementwise)` teaches every `DependencyGraph` a new op; pass
`op_propagators={...}` to one graph to scope it to that graph.

Channels are wherever the layer that produced them keeps them
(`module_rules.ModuleRule.channel_dim`: dim 1 of a conv's `[N, C, H, W]`,
the last dim of a Linear's `[N, T, features]`), and `walk.channel_dim(node)`
says where that is, counted from the end so it lines up across
broadcasting. (They used to read dim 1, whoever produced the channels --
see `ModuleRule.channel_dim` for what that broke.)

These used to live in `graph.py`, next to the walk that calls them. Adding
an op and changing how the graph is walked are different changes, so they
are now different modules, the same way `module_rules.py` was split out.
"""
from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Callable

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from .indices import expand_blocks
from .registry import Registry

if TYPE_CHECKING:  # only for annotations; propagation.py imports this module
    from .propagation import Propagation

OpPropagator = Callable[["Propagation", fx.Node, fx.Node, torch.Tensor], None]  # (walk, source, node, idx)


def shares_channels(shape: torch.Size, dim: int) -> bool:
    """Whether an operand of `shape`, broadcast against a tensor whose
    channels are on `dim` (counted from the end), carries those channels:
    it reaches that dimension and isn't broadcast along it. A scalar, a
    `[W]` vector against `[N, C, H, W]`, or a `[N, 1, H, W]` spatial gate
    doesn't."""
    return len(shape) >= -dim and shape[dim] != 1


def propagate_elementwise(walk: Propagation, source: fx.Node, node: fx.Node, idx: torch.Tensor) -> None:
    """An elementwise op on `source` and at most one other tensor: a
    residual connection's `out + identity`, a squeeze-and-excitation
    block's `x * scale`, `x - y`, `x / 2`. Channel `c` of the result is
    channel `c` of `source`. When the other operand carries the same
    channels, its producer must lose them too, and the prune continues from
    both it and the merge. A scalar, or an operand broadcast across channels
    (a `[N, 1, H, W]` spatial gate), shares no channels with `source`, so
    the prune just passes through. Register it for another elementwise op:
    `register_op_propagator(torch.maximum, propagate_elementwise)`.

    It used to be `propagate_add`, registered for `+` only, and it took
    every merge to be two same-shape branches: `x + 1` was refused, and the
    1-channel conv behind a spatial gate was told to lose channels it
    doesn't have. `-`, `*` and `/` were refused outright, so no
    squeeze-and-excitation block could be pruned through."""
    dim = walk.channel_dim(source)
    source_shape, result_shape = walk.shape(source), walk.shape(node)
    if len(result_shape) < -dim or result_shape[dim] != source_shape[dim]:
        raise NotImplementedError(
            f"{node.name}: the pruned operand {tuple(source_shape)} is broadcast up to "
            f"{tuple(result_shape)}, so its channels aren't the result's"
        )
    others = [n for n in node.all_input_nodes if n is not source]
    if len(others) > 1:
        raise NotImplementedError(
            f"{node.name}: elementwise op with {len(others)} other tensor operands (at most 1 is supported)"
        )
    if others and shares_channels(walk.shape(others[0]), dim):
        producer = walk.find_producer(others[0], idx)
        walk.add_output_target(producer.target, idx)
        walk.forward(producer, idx)
    walk.forward(node, idx)


def propagate_channelwise(walk: Propagation, source: fx.Node, node: fx.Node, idx: torch.Tensor) -> None:
    """An op that keeps channel `c` as channel `c` -- an activation,
    dropout, pooling, a resize -- written as a function or a tensor method
    (`F.relu(x)`, `x.relu()`, `F.max_pool2d(x, 2)`): the prune passes
    straight through. A module (`nn.ReLU()`) needs no entry, since a
    `call_module` node that keeps its channel count already passes indices
    on; the function form of the same op used to be refused, so a model
    pruned or raised depending on how its activations were written.
    Register it for a function of your own:
    `register_op_propagator(my_activation, propagate_channelwise)`."""
    dim = walk.channel_dim(source)
    before, after = walk.shape(source), walk.shape(node)
    if len(after) != len(before) or after[dim] != before[dim]:
        raise NotImplementedError(
            f"{node.name}: {tuple(before)} -> {tuple(after)} changes the channel dimension, "
            f"so it can't be passed through as a channel-wise op"
        )
    walk.forward(node, idx)


def propagate_cat(walk: Propagation, source: fx.Node, node: fx.Node, idx: torch.Tensor) -> None:
    """Channel concatenation: `source`'s channels land in the result shifted
    by the widths of the inputs before it -- once for every place `source`
    appears (`torch.cat([y, y])`). Each is one part of what the result
    loses (`walk.forward_part`): when several of the concatenated tensors
    lose channels in one prune, the walk carries all of them on together.

    It used to `forward` from the result for the first branch to arrive
    only -- the walk skips a node it has already left -- so with two pruned
    branches (`cat([y, relu(y)])`, a GhostNet-style `cat([x, cheap(x)])`)
    the layer after the `cat` kept the inputs the second branch no longer
    produced, and `cat([y, y])` lost the second copy's."""
    cat_inputs = list(node.args[0])
    cat_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    positions = [i for i, n in enumerate(cat_inputs) if n is source]
    if not positions:
        raise NotImplementedError(f"{node.name}: pruned tensor not found among cat's inputs")

    ndim = len(walk.shape(source))
    from_end = cat_dim - ndim if cat_dim >= 0 else cat_dim
    if from_end != walk.channel_dim(source):
        raise NotImplementedError(
            f"{node.name}: cat along dim={cat_dim} isn't the channel dimension "
            f"(dim {walk.channel_dim(source) + ndim} of {tuple(walk.shape(source))}), not supported"
        )
    widths = [int(walk.shape(n)[from_end]) for n in cat_inputs]
    for position in positions:
        walk.forward_part(node, idx + sum(widths[:position]))


_SHAPE_ATTRIBUTES = frozenset({"shape", "dtype", "device", "ndim"})


def _shape_query(walk: Propagation, source: fx.Node, node: fx.Node, idx: torch.Tensor) -> None:
    """`x.size(0)`, `x.dim()`, `x.shape`: reads the tensor's metadata, not
    its values, so no channel flows on from here and there is nothing to
    carry. The reshape that uses the answer is reached from the tensor
    itself. `x.view(x.size(0), -1)`, the usual way to flatten into a
    classifier head, used to be refused at the `size` call."""
    if node.target is getattr and node.args[1] not in _SHAPE_ATTRIBUTES:
        raise NotImplementedError(f"{node.name}: attribute {node.args[1]!r} of a pruned tensor isn't a shape query")


def propagate_flatten(walk: Propagation, source: fx.Node, node: fx.Node, idx: torch.Tensor) -> None:
    """A conv output flattened ahead of a Linear, `[N, C, H, W]` -> `[N, C *
    H * W]`: channel `c` becomes the `H * W` consecutive features it owns.

    Any other reshape raises: one that keeps the channel dimension
    (`x.flatten(2)`), folds the batch in (`x.view(-1)`) or splits the
    channels (a channel shuffle). It used to expand the indices for every
    view/reshape/flatten of a 4D tensor whatever its result, so after
    `x.flatten(2)` the next layer was handed indices up to `C * H * W` and
    failed at rebuild, far from the cause."""
    shape, out = walk.shape(source), walk.shape(node)
    nchw = len(shape) == 4 and walk.channel_dim(source) == -3
    if not nchw or tuple(out) != (shape[0], shape[1] * shape[2] * shape[3]):
        raise NotImplementedError(
            f"{node.name}: reshaping {tuple(shape)} to {tuple(out)} isn't the conv-output-into-classifier-head "
            f"flatten DependencyGraph handles (only [N, C, H, W] -> [N, C*H*W], channels on dim 1, is)"
        )
    walk.forward(node, expand_blocks(idx, int(shape[2]) * int(shape[3])), dim=-1)


# Keyed the way fx records a node's target: the function itself for
# `call_function` (`torch.sub`), the method name for `call_method`
# (`"sub"`), and -- since a `call_module` target is just the submodule's
# name -- the module class (subclasses included) for modules.
# `register_op_propagator(target, propagator)` teaches every
# `DependencyGraph` a new op (re-registering one raises unless
# `overwrite=True`); pass `op_propagators={...}` to one graph instead to
# scope it to that graph.
OP_PROPAGATORS: Registry[OpPropagator] = Registry("op propagator")
register_op_propagator = OP_PROPAGATORS.register

_ELEMENTWISE = (
    torch.add, operator.add, operator.iadd, "__add__", "__iadd__", "add", "add_",
    torch.sub, torch.subtract, operator.sub, operator.isub, "sub", "sub_",
    torch.mul, torch.multiply, operator.mul, operator.imul, "mul", "mul_",
    torch.div, torch.divide, torch.true_divide, operator.truediv, operator.itruediv, "div", "div_",
)
_CHANNELWISE = (
    torch.relu, torch.relu_, torch.sigmoid, torch.tanh, F.relu, F.relu6, F.leaky_relu, F.elu, F.selu,
    F.celu, F.gelu, F.silu, F.mish, F.hardswish, F.hardsigmoid, F.hardtanh, F.softplus,
    F.dropout, F.dropout2d, F.max_pool2d, F.avg_pool2d, F.adaptive_max_pool2d, F.adaptive_avg_pool2d, F.interpolate,
    "relu", "relu_", "sigmoid", "sigmoid_", "tanh", "tanh_", "contiguous", "clone",
)
for _target in dict.fromkeys(_ELEMENTWISE):  # dict.fromkeys: a few of these are one object under two names
    register_op_propagator(_target, propagate_elementwise)
for _target in dict.fromkeys(_CHANNELWISE):
    register_op_propagator(_target, propagate_channelwise)
for _target in (torch.cat, torch.concat):
    register_op_propagator(_target, propagate_cat)
for _target in (torch.flatten, "view", "reshape", "flatten", nn.Flatten):
    register_op_propagator(_target, propagate_flatten)
for _target in ("size", "dim", getattr):
    register_op_propagator(_target, _shape_query)
del _target
