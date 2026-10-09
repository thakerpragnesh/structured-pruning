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

Built-in propagators cover `+`/`torch.add` (`propagate_add`), `torch.cat`
along the channel dimension (`propagate_cat`), and a conv output flattened
ahead of a Linear (`propagate_flatten`). `register_op_propagator(torch.sub,
propagate_add)` teaches every `DependencyGraph` a new op; pass
`op_propagators={...}` to one graph to scope it to that graph.

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

from .indices import expand_blocks
from .registry import Registry

if TYPE_CHECKING:  # only for annotations; graph.py imports this module
    from .graph import Propagation

OpPropagator = Callable[["Propagation", fx.Node, fx.Node, torch.Tensor], None]  # (walk, source, node, idx)


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
    walk.add_output_target(producer.target, idx)
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
# `call_function` (`torch.sub`), the method name for `call_method`
# (`"sub"`), and -- since a `call_module` target is just the submodule's
# name -- the module class (subclasses included) for modules.
# `register_op_propagator(target, propagator)` teaches every
# `DependencyGraph` a new op (re-registering one raises unless
# `overwrite=True`); pass `op_propagators={...}` to one graph instead to
# scope it to that graph.
OP_PROPAGATORS: Registry[OpPropagator] = Registry("op propagator")
register_op_propagator = OP_PROPAGATORS.register

for _target in (torch.add, operator.add, operator.iadd, "__add__", "__iadd__", "add", "add_"):
    register_op_propagator(_target, propagate_add)
for _target in (torch.cat, torch.concat):
    register_op_propagator(_target, propagate_cat)
for _target in (torch.flatten, "view", "reshape", "flatten", nn.Flatten):
    register_op_propagator(_target, propagate_flatten)
del _target
