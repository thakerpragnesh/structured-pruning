"""
How a prune passes through an operation that owns no weights -- the op-side
twin of `module_rules.py`. A module rule says how a *layer type* takes part
in a prune; an op propagator says how a prune is carried *through* an
operation between layers (an elementwise add, a `torch.cat`, a flatten).

A propagator is `(walk, source, node, idx) -> None`: a prune of channels
`idx` of `source`'s output has reached `node`. It reads recorded shapes with
`walk.shape`, records what must shrink with `walk.add_output_target` /
`walk.add_input_target`, and continues the walk with `walk.forward` /
`walk.find_producer` -- `graph.Propagation`'s small public interface, which
is all it sees of the walk (not the graph, and not the group being filled).

Built-in propagators cover elementwise `+ - * /` (`propagate_elementwise`),
activations, dropout, pooling and resizing written as functions or tensor
methods (`propagate_channelwise`), `torch.cat` along the channel dimension
(`propagate_cat`), a conv output flattened ahead of a Linear
(`propagate_flatten`), and reading a tensor's shape (`x.size(0)`), which
carries no channels on. `register_op_propagator(torch.maximum,
propagate_elementwise)` teaches every `DependencyGraph` a new op; pass
`op_propagators={...}` to one graph to scope it to that graph. Channels are
dim 1 here, as everywhere in `DependencyGraph`.

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

if TYPE_CHECKING:  # only for annotations; graph.py imports this module
    from .graph import Propagation

OpPropagator = Callable[["Propagation", fx.Node, fx.Node, torch.Tensor], None]  # (walk, source, node, idx)


def _channels_against(source_shape: torch.Size, other_shape: torch.Size) -> int | None:
    """The size of `other`'s dimension that broadcasts against `source`'s
    channel dimension (dim 1), or None if `other` has too few dimensions
    to reach it (a scalar, a `[W]` vector)."""
    offset = len(source_shape) - len(other_shape)
    if offset < 0:
        raise NotImplementedError(
            f"an operand of shape {tuple(other_shape)} has more dimensions than the pruned {tuple(source_shape)}"
        )
    return None if offset > 1 else other_shape[1 - offset]


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
    source_shape = walk.shape(source)
    if walk.shape(node)[1] != source_shape[1]:
        raise NotImplementedError(
            f"{node.name}: the pruned operand {tuple(source_shape)} is broadcast up to "
            f"{tuple(walk.shape(node))}, so its channels aren't the result's"
        )
    others = [n for n in node.all_input_nodes if n is not source]
    if len(others) > 1:
        raise NotImplementedError(
            f"{node.name}: elementwise op with {len(others)} other tensor operands (at most 1 is supported)"
        )
    if others and _channels_against(source_shape, walk.shape(others[0])) not in (None, 1):
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
    before, after = walk.shape(source), walk.shape(node)
    if len(after) != len(before) or after[1] != before[1]:
        raise NotImplementedError(
            f"{node.name}: {tuple(before)} -> {tuple(after)} changes the channel dimension, "
            f"so it can't be passed through as a channel-wise op"
        )
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
    if len(shape) != 4 or tuple(out) != (shape[0], shape[1] * shape[2] * shape[3]):
        raise NotImplementedError(
            f"{node.name}: reshaping {tuple(shape)} to {tuple(out)} isn't the conv-output-into-classifier-head "
            f"flatten DependencyGraph handles (only [N, C, H, W] -> [N, C*H*W] is)"
        )
    walk.forward(node, expand_blocks(idx, int(shape[2]) * int(shape[3])))


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
