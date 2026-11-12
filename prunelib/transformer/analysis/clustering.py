"""Pure K-means clustering: no autograd, no external ML dependency.

Exists as its own module because `ffn_selectors.WeightClusterRedundancySelector`
is the only consumer today, but a from-scratch k-means over normalized
weight vectors is a clustering strategy in its own right, not selector
bookkeeping -- keeping it separate is what lets the selector focus purely
on turning a cluster assignment into a `Selection`.
"""
import torch


def kmeans_assign(x: torch.Tensor, k: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
    """Lloyd's algorithm. Returns a `(n,)` LongTensor of cluster ids.

    `x` is `(n, dim)`. Centroids are seeded from `k` distinct rows chosen
    via a seeded permutation, so results are reproducible independent of
    any global RNG state. An empty cluster keeps its previous centroid
    rather than being reseeded -- rare with well-separated neuron weight
    vectors, and simpler than a reseeding heuristic that would need its
    own justification.
    """
    n = x.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    if k == n:
        return torch.arange(n, device=x.device)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    seed_idx = torch.randperm(n, generator=generator)[:k]
    centroids = x[seed_idx].clone()

    # On the input's device, like `new_assignment` below -- `torch.equal`
    # raises on tensors from two devices, so a CPU-side initial value broke
    # clustering on GPU at the first convergence check.
    assignment = torch.full((n,), -1, dtype=torch.long, device=x.device)
    for _ in range(iters):
        distances = torch.cdist(x, centroids)
        new_assignment = distances.argmin(dim=1)
        if torch.equal(new_assignment, assignment):
            break
        assignment = new_assignment
        for cluster in range(k):
            members = x[assignment == cluster]
            if members.shape[0] > 0:
                centroids[cluster] = members.mean(dim=0)
    return assignment
