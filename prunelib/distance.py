"""
Distance metrics, shared by redundancy scanning (`scanners.
pairwise_distance_matrix`) and K-Means selection (`clustering.kmeans`).

These are the Manhattan/Euclidean/Cosine metrics compared in Thaker & Mohan,
IEEE Access 2024. They used to be written out twice -- once as an if-chain
in `scanners.py`, once in `clustering._distance` -- so a fourth metric meant
editing both, and nothing stopped the two copies from drifting apart.

Each `DistanceMetric` pairs its distance with the K-Means centroid update
that actually minimizes it (see `clustering.py`'s module docstring for why
Manhattan needs the median, not the mean). Keeping them in one object is
what makes that pairing impossible to get wrong when adding a metric: you
can't register a distance for clustering without also saying how to
average under it. A metric registered with `centroid=None` still works for
scanning, and `kmeans` rejects it with a clear error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .registry import Registry


@dataclass(frozen=True)
class DistanceMetric:
    """`pairwise(x [N, D], y [K, D]) -> [N, K]`, plus `centroid(members [M,
    D]) -> [D]`, the K-Means update step that minimizes this distance (or
    None if the metric is only meant for scanning)."""

    pairwise: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    centroid: Callable[[torch.Tensor], torch.Tensor] | None = None


DISTANCE_METRICS: Registry[DistanceMetric] = Registry("metric")
register_distance_metric = DISTANCE_METRICS.register


def _cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xn = F.normalize(x, dim=1, eps=1e-12)
    yn = F.normalize(y, dim=1, eps=1e-12)
    return 1.0 - xn @ yn.T


register_distance_metric("manhattan", DistanceMetric(
    pairwise=lambda x, y: torch.cdist(x, y, p=1),
    centroid=lambda members: members.median(dim=0).values,  # K-Medians
))
register_distance_metric("euclidean", DistanceMetric(
    pairwise=lambda x, y: torch.cdist(x, y, p=2),
    centroid=lambda members: members.mean(dim=0),  # Lloyd's K-Means
))
register_distance_metric("cosine", DistanceMetric(
    pairwise=_cosine_distance,
    centroid=lambda members: F.normalize(members.mean(dim=0), dim=0, eps=1e-12),  # spherical K-Means
))


def pairwise_distance(x: torch.Tensor, y: torch.Tensor, metric: str | DistanceMetric = "manhattan") -> torch.Tensor:
    """[N, D] x [K, D] -> [N, K] distance from every row of `x` to every row
    of `y`. `metric` is a registered name or a `DistanceMetric`."""
    return DISTANCE_METRICS.resolve(metric).pairwise(x, y)
