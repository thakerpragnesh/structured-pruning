"""
One-call generic pruning: score a layer, choose what to prune, and resize
everything `graph.DependencyGraph` finds it coupled to.

This is policy on top of mechanism -- `selection.py` decides which channels
go, `graph.py` works out what else has to change, `group.py` applies it --
so it lives apart from all three. `prune_model` used to sit at the bottom of
`graph.py`, which made the dependency tracer import the selection rules it
otherwise has no use for.
"""
from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from .graph import DependencyGraph
from .group import PruningGroup
from .indices import complement_indices
from .selection import Selector, prune_count, select_prune_indices_by_method


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

    At least one channel always survives (`selection.prune_count`), however
    close `prune_fraction` is to 1.

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
    n_to_prune = prune_count(n_out, prune_fraction, min_prune=0)
    prune_idx = select_prune_indices_by_method(module.weight, n_to_prune, method=method, **method_kwargs)

    group = dep.get_pruning_group(layer, complement_indices(n_out, prune_idx))
    group.prune()
    return group
