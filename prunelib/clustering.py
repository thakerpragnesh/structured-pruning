"""
K-Means clustering-based channel selection: group a layer's output channels
(or neurons, or attention heads) by weight similarity, then, *within each
cluster*, prune the channels with the smallest L1 norm and keep the
strongest one as that cluster's representative.

This is the selection rule behind the Manhattan/Euclidean/Cosine K-Means
comparison in Thaker & Mohan, IEEE Access 2024 (Manhattan: 35.15% param /
49.11% FLOPs reduction on VGG16 at <1% accuracy drop). The intuition: channels
that land in the same cluster compute nearly the same feature, so all but
one of them are redundant -- and among near-duplicates, the one with the
largest L1 norm contributes the most, so it's the one worth keeping.

Unlike `saliency.py`'s scorers, this is not a per-channel score: whether a
channel is pruned depends on which other channels it clusters with, so it
can't be expressed as `compute_score(...) -> select_prune_indices(...)`.
`select_prune_indices_by_method` below is the one dispatcher that hides that
difference from callers (`vgg.py`, `graph.prune_model`), so they accept
`method="kmeans"` alongside `"max_k"`/`"l1"`/`"l2"`/`"random"` without each
growing its own branch.

K-Means is implemented here in plain torch (no scikit-learn dependency).
Each metric uses the centroid update that actually minimizes it, not the
Euclidean mean for all three:

- `"euclidean"`: arithmetic mean (standard Lloyd's K-Means).
- `"manhattan"`: coordinate-wise median (K-Medians) -- the mean minimizes
  squared L2 distance, not L1, so using it with Manhattan assignment would
  not be a descent step for the Manhattan objective.
- `"cosine"`: L2-normalized mean (spherical K-Means).
"""
from __future__ import annotations

import torch

from .saliency import compute_score, select_prune_indices

_METRICS = ("manhattan", "euclidean", "cosine")


def _distance(x: torch.Tensor, centroids: torch.Tensor, metric: str) -> torch.Tensor:
    """[N, D] x [K, D] -> [N, K] distance from every point to every centroid."""
    if metric == "manhattan":
        return torch.cdist(x, centroids, p=1)
    if metric == "euclidean":
        return torch.cdist(x, centroids, p=2)
    if metric == "cosine":
        xn = torch.nn.functional.normalize(x, dim=1, eps=1e-12)
        cn = torch.nn.functional.normalize(centroids, dim=1, eps=1e-12)
        return 1.0 - xn @ cn.T
    raise ValueError(f"unknown metric {metric!r}, expected one of {list(_METRICS)}")


def _update_centroid(members: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "manhattan":
        return members.median(dim=0).values
    centroid = members.mean(dim=0)
    if metric == "cosine":
        centroid = torch.nn.functional.normalize(centroid, dim=0, eps=1e-12)
    return centroid


def _kmeans_plus_plus_init(x: torch.Tensor, n_clusters: int, metric: str, generator: torch.Generator) -> torch.Tensor:
    """K-Means++ seeding under `metric`: each new centroid is sampled with
    probability proportional to its squared distance from the nearest
    centroid chosen so far, which spreads the initial centroids out and
    avoids the degenerate all-seeds-in-one-cluster starts plain random
    sampling can produce."""
    n = x.shape[0]
    first = torch.randint(n, (1,), generator=generator).item()
    chosen = [first]
    nearest = _distance(x, x[first:first + 1], metric).squeeze(1)
    for _ in range(1, n_clusters):
        weights = nearest.clamp_min(0).pow(2)
        if weights.sum() <= 0:
            # Every remaining point coincides with an existing centroid (e.g.
            # duplicate channels); pick any not-yet-chosen index instead.
            taken = set(chosen)
            remaining = [i for i in range(n) if i not in taken]
            nxt = remaining[torch.randint(len(remaining), (1,), generator=generator).item()]
        else:
            nxt = torch.multinomial(weights, 1, generator=generator).item()
        chosen.append(nxt)
        nearest = torch.minimum(nearest, _distance(x, x[nxt:nxt + 1], metric).squeeze(1))
    return x[chosen].clone()


def kmeans(
    vectors: torch.Tensor,
    n_clusters: int,
    metric: str = "manhattan",
    max_iter: int = 100,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster the rows of `vectors` ([N, D]) into `n_clusters` groups.

    Returns `(labels [N], centroids [n_clusters, D])`. Deterministic for a
    given `seed`. Every cluster is guaranteed non-empty: if an update step
    leaves one empty, it's re-seeded with the point farthest from its
    currently-assigned centroid (the standard fix, and what makes
    `kmeans_prune_indices`' "each cluster keeps one representative" rule
    produce exactly the requested number of survivors).
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}, expected one of {list(_METRICS)}")
    if vectors.dim() != 2:
        raise ValueError(f"expected a 2D [N, D] tensor, got shape {tuple(vectors.shape)}")
    n = vectors.shape[0]
    if not 1 <= n_clusters <= n:
        raise ValueError(f"n_clusters must be in [1, {n}], got {n_clusters}")

    x = vectors.detach().to(torch.float32)
    generator = torch.Generator().manual_seed(seed)
    centroids = _kmeans_plus_plus_init(x, n_clusters, metric, generator)
    labels = torch.full((n,), -1, dtype=torch.long)

    for _ in range(max_iter):
        dist = _distance(x, centroids, metric)
        new_labels = dist.argmin(dim=1)

        # Repair empty clusters before the update step, so no centroid is
        # left undefined.
        for c in range(n_clusters):
            if not (new_labels == c).any():
                point_dist = dist.gather(1, new_labels.unsqueeze(1)).squeeze(1)
                # Only steal from clusters that would still have a member left.
                counts = torch.bincount(new_labels, minlength=n_clusters)
                donor_ok = counts[new_labels] > 1
                point_dist = torch.where(donor_ok, point_dist, torch.full_like(point_dist, -1.0))
                new_labels[point_dist.argmax()] = c

        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        centroids = torch.stack([_update_centroid(x[labels == c], metric) for c in range(n_clusters)])

    return labels, centroids


def kmeans_prune_indices(
    weight: torch.Tensor,
    prune_amount: int,
    n_clusters: int | None = None,
    metric: str = "manhattan",
    seed: int = 0,
    max_iter: int = 100,
) -> torch.Tensor:
    """Indices of the output channels to prune, ascending, chosen by
    clustering channels on their flattened weights and pruning the
    lowest-L1-norm channels *within each cluster*.

    weight: Conv2d `[out, in, kh, kw]` or Linear `[out, in]` weight (any
        tensor whose dim 0 is the channel axis works; each channel's slice is
        flattened into one clustering vector)
    prune_amount: how many channels to remove.
    n_clusters: defaults to `out - prune_amount`, the paper's setting --
        exactly one channel (the highest-L1 one) survives per cluster, and
        every other cluster member is pruned. A smaller `n_clusters` groups
        more coarsely: each cluster still keeps its highest-L1 member as a
        protected representative, and the `prune_amount` lowest-L1
        channels among all *non*-representatives are pruned -- so no cluster
        is ever emptied, and within any one cluster the weakest channels
        always go first.

    Raises if `prune_amount` exceeds the number of non-representative
    channels (i.e. `out - n_clusters`), since satisfying it would mean
    emptying a cluster.
    """
    n_out = weight.shape[0]
    prune_amount = max(0, min(int(prune_amount), n_out))
    if prune_amount == 0:
        return torch.empty(0, dtype=torch.long)
    if prune_amount >= n_out:
        raise ValueError(f"prune_amount {prune_amount} would remove every one of the {n_out} channels")
    if n_clusters is None:
        n_clusters = n_out - prune_amount
    if prune_amount > n_out - n_clusters:
        raise ValueError(
            f"can't prune {prune_amount} of {n_out} channels while keeping one representative "
            f"per cluster with n_clusters={n_clusters} (at most {n_out - n_clusters} are prunable)"
        )

    vectors = weight.detach().reshape(n_out, -1)
    labels, _ = kmeans(vectors, n_clusters, metric=metric, max_iter=max_iter, seed=seed)
    l1 = vectors.abs().sum(dim=1).to(torch.float32)

    # The representative of each cluster is its highest-L1 member; it's
    # never a pruning candidate. Everyone else competes on L1 alone.
    protected = torch.zeros(n_out, dtype=torch.bool)
    for c in range(n_clusters):
        members = (labels == c).nonzero(as_tuple=True)[0]
        protected[members[l1[members].argmax()]] = True

    candidate_scores = torch.where(protected, torch.full_like(l1, float("inf")), l1)
    return select_prune_indices(candidate_scores, prune_amount)


def select_prune_indices_by_method(weight: torch.Tensor, prune_amount: int, method: str = "max_k", **kwargs) -> torch.Tensor:
    """One entry point for every selection rule this package has: the
    per-channel saliency scorers (`"max_k"`, `"l1"`, `"l2"`, `"random"` --
    score via `compute_score`, then take the lowest `prune_amount`) and
    cluster-based selection (`"kmeans"` -- see `kmeans_prune_indices`;
    `kwargs` are forwarded to it, e.g. `metric=`, `n_clusters=`, `seed=`).
    """
    if method == "kmeans":
        return kmeans_prune_indices(weight, prune_amount, **kwargs)
    return select_prune_indices(compute_score(weight, method=method, **kwargs), prune_amount)

