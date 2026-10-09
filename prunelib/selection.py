"""
Selection rules: given a layer's weight and a budget, which output channels
to prune. This is the one abstraction every caller that *decides* what to
prune depends on -- `vgg.py`, `oneshot.prune_model` -- so none of them knows
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
`select_prune_indices_among` applies any of them to a subset of channels
(the not-yet-masked ones, in an iterative masking loop).

`prune_count` is the other half of "what to prune": how many channels a
`prune_fraction` budget means. `oneshot.prune_model`, `vgg.prune_vgg_layer`
and `vgg.mask_vgg_layer` used to each round and clamp that themselves, three
slightly different ways -- and `prune_model`'s version could round a budget
just under 1.0 up to every channel, leaving a zero-width layer.
"""
from __future__ import annotations

from typing import Callable, Union

import torch

from .registry import Registry
from .saliency import SALIENCY_METHODS, compute_score, select_prune_indices

Selector = Callable[..., torch.Tensor]  # (weight, prune_amount, **kwargs) -> prune indices, ascending

SELECTION_METHODS: Registry[Selector] = Registry("selection method")
register_selection_method = SELECTION_METHODS.register


def prune_count(n: int, fraction: float, *, available: int | None = None, min_prune: int = 1, min_keep: int = 1) -> int:
    """How many of `n` channels a `fraction` budget prunes: `round(n *
    fraction)`, raised to at least `min_prune`, then capped so that at least
    `min_keep` of the `available` candidates (default: all `n`) survive.
    Never negative. Each caller states its own floor and cap through the
    keywords; the rounding itself is the same for everyone."""
    available = n if available is None else available
    return max(0, min(max(min_prune, int(round(n * fraction))), available - min_keep))


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


def select_prune_indices_among(
    weight: torch.Tensor,
    candidates: torch.Tensor,
    prune_amount: int,
    method: Union[str, Selector] = "max_k",
    **kwargs,
) -> torch.Tensor:
    """`select_prune_indices_by_method`, restricted to the output channels
    in `candidates`: the rule only sees those channels' weights, and the
    result is indices into `weight`'s dim 0, ascending.

    Iterative masking needs this for correctness: an already-masked
    channel's weight is zero, the lowest score under every criterion, so
    selecting over all channels would pick it again every iteration and the
    schedule would advance far slower than its budget (KT.md section 3).
    Pass the unmasked channels (`masking.surviving_channels`) as
    `candidates`. This used to live inside `vgg.mask_vgg_layer`, out of
    reach of any other masking loop."""
    positions = select_prune_indices_by_method(
        weight.detach().index_select(0, candidates), prune_amount, method=method, **kwargs
    )
    return candidates[positions].sort().values
