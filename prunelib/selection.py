"""
Selection rules: given a layer's weight and a budget, which output channels
to prune. This is the one abstraction every caller that *decides* what to
prune depends on -- `vgg.py`, `graph.prune_model` -- so none of them knows
which concrete rules exist.

Two kinds of rule, behind one entry point (`select_prune_indices_by_method`):

- Every saliency scorer in `saliency.SALIENCY_METHODS` (`"max_k"`, `"l1"`,
  `"l2"`, `"random"`, plus anything registered later) is automatically a
  selection rule: score via `compute_score`, then take the lowest
  `prune_amount` via `select_prune_indices` -- the same one-score-one-
  selection path that keeps D1 structurally impossible.
- Rules that aren't a per-channel score at all register a full selector in
  `SELECTION_METHODS` -- `(weight, prune_amount, **kwargs) -> prune_idx`,
  ascending. `clustering.py` registers `"kmeans"` there (whether a channel
  is pruned depends on which others it clusters with). This module never
  imports `clustering`: plugins depend on the abstraction, not the other way
  round, so a new cluster-based rule (KT.md section 7) is a
  `register_selection_method` call, not an edit here.

A callable can be passed as `method=` directly too, with the selector
signature above, for a one-off rule not worth registering.
"""
from __future__ import annotations

from typing import Callable, Union

import torch

from .registry import Registry
from .saliency import SALIENCY_METHODS, compute_score, select_prune_indices

Selector = Callable[..., torch.Tensor]  # (weight, prune_amount, **kwargs) -> prune indices, ascending

SELECTION_METHODS: Registry[Selector] = Registry("selection method")
register_selection_method = SELECTION_METHODS.register


def available_methods() -> list[str]:
    """Every name `select_prune_indices_by_method` accepts."""
    return SELECTION_METHODS.names() + [m for m in SALIENCY_METHODS.names() if m not in SELECTION_METHODS]


def select_prune_indices_by_method(
    weight: torch.Tensor,
    prune_amount: int,
    method: Union[str, Selector] = "max_k",
    **kwargs,
) -> torch.Tensor:
    """Indices of the `prune_amount` output channels of `weight` to prune,
    ascending, under `method`: a name in `SELECTION_METHODS` (e.g.
    `"kmeans"`), a name in `saliency.SALIENCY_METHODS` (score, then take the
    lowest), or a selector callable. `kwargs` are forwarded to the rule
    (e.g. `k=` for `"max_k"`; `metric=`, `n_clusters=`, `seed=` for
    `"kmeans"`). A name registered as both a selector and a scorer resolves
    to the selector."""
    if callable(method):
        return method(weight, prune_amount, **kwargs)
    if method in SELECTION_METHODS:
        return SELECTION_METHODS.get(method)(weight, prune_amount, **kwargs)
    if method in SALIENCY_METHODS:
        return select_prune_indices(compute_score(weight, method=method, **kwargs), prune_amount)
    raise ValueError(f"unknown method {method!r}, expected one of {available_methods()}")
