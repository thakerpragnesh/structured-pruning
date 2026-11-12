"""
Redundancy scanners: pairwise distance between channels/heads (for the
Manhattan/Euclidean/Cosine K-Means comparison from Thaker & Mohan, IEEE
Access 2024) and co-activation overlap (for the Transformer extension work).
"""
from __future__ import annotations

from typing import Callable

import torch


DistanceFn = Callable[[torch.Tensor], torch.Tensor]
"""Maps row vectors `[N, D]` to a pairwise distance matrix `[N, N]`."""

_METRICS: dict[str, DistanceFn] = {}


def register_metric(name: str, fn: DistanceFn | None = None, *, overwrite: bool = False):
    """Register a pairwise distance under `name` for
    `pairwise_distance_matrix(..., metric=name)`. Usable directly or as a
    decorator; mirrors `prunelib.saliency.register_saliency`."""

    def _register(f: DistanceFn) -> DistanceFn:
        if name in _METRICS and not overwrite:
            raise ValueError(f"distance metric {name!r} is already registered")
        _METRICS[name] = f
        return f

    return _register(fn) if fn is not None else _register


def _cosine_distance(vectors: torch.Tensor) -> torch.Tensor:
    normed = torch.nn.functional.normalize(vectors, dim=1, eps=1e-12)
    return 1.0 - normed @ normed.T


register_metric("manhattan", lambda v: torch.cdist(v, v, p=1))
register_metric("euclidean", lambda v: torch.cdist(v, v, p=2))
register_metric("cosine", _cosine_distance)


def pairwise_distance_matrix(vectors: torch.Tensor, metric: str = "manhattan") -> torch.Tensor:
    """Pairwise distance between rows of `vectors`, shape [N, D] -> [N, N].

    metric: "manhattan" (L1), "euclidean" (L2), or "cosine" (1 - cosine
    similarity), plus anything added via `register_metric`.
    Manhattan was found to outperform the other two for K-Means-based channel
    selection in the paper above (35.15% param / 49.11% FLOPs reduction on
    VGG16 vs. Euclidean's 31.01%/43.96% and Cosine's 21.93%/32.03%).
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}, expected one of {list(_METRICS)}")
    return _METRICS[metric](vectors)


class CoActivationScanner:
    """Finds units (neurons/heads) that fire together across tokens.

    Units that fire on nearly every token overlap with *everything* by
    construction — a unit that's on 95% of the time will show ~90%+ Jaccard
    similarity with any other frequently-firing unit, which looks like
    redundancy but is really just both units being non-selective. Excluding
    high-firing-rate units before computing similarity removes this class of
    false positive.
    """

    def __init__(self, firing_rate_ceiling: float = 0.9):
        self.firing_rate_ceiling = firing_rate_ceiling

    def firing_rate(self, activation_mask: torch.Tensor) -> torch.Tensor:
        """activation_mask: [num_units, num_tokens] boolean. Returns [num_units]."""
        return activation_mask.float().mean(dim=1)

    def jaccard(self, activation_mask: torch.Tensor) -> torch.Tensor:
        """Pairwise Jaccard similarity among units, with high-firing-rate units
        excluded (their rows/columns are returned as zero rather than dropped,
        so the output shape always matches the input unit count).

        activation_mask: [num_units, num_tokens] boolean
        returns: [num_units, num_units]
        """
        n_units = activation_mask.shape[0]
        rate = self.firing_rate(activation_mask)
        keep = (rate <= self.firing_rate_ceiling).nonzero(as_tuple=True)[0]

        sim = torch.zeros(n_units, n_units, device=activation_mask.device)
        if keep.numel() < 2:
            return sim

        sub = activation_mask.index_select(0, keep).float()
        intersection = sub @ sub.T
        counts = sub.sum(dim=1, keepdim=True)
        union = counts + counts.T - intersection
        jac = torch.where(union > 0, intersection / union, torch.zeros_like(union))

        sim[keep.unsqueeze(1), keep.unsqueeze(0)] = jac
        return sim
