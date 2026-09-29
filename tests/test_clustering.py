import pytest
import torch
import torch.nn as nn

from prunelib import compute_score, kmeans, kmeans_prune_indices, prune_model, select_prune_indices
from prunelib.clustering import select_prune_indices_by_method


def _grouped_channels(group_sizes, dim=16, noise=0.01, seed=0):
    """Channels drawn as small perturbations around one well-separated
    center per group. Returns (vectors [N, dim], true_group [N])."""
    g = torch.Generator().manual_seed(seed)
    vecs, groups = [], []
    for gid, size in enumerate(group_sizes):
        center = torch.randn(dim, generator=g) * 10
        for _ in range(size):
            vecs.append(center + noise * torch.randn(dim, generator=g))
            groups.append(gid)
    return torch.stack(vecs), torch.tensor(groups)


def _same_partition(labels_a, labels_b):
    """Two labelings describe the same partition (up to renaming clusters)."""
    n = labels_a.numel()
    same_a = labels_a.unsqueeze(0) == labels_a.unsqueeze(1)
    same_b = labels_b.unsqueeze(0) == labels_b.unsqueeze(1)
    return torch.equal(same_a, same_b) and n == labels_b.numel()


@pytest.mark.parametrize("metric", ["manhattan", "euclidean", "cosine"])
def test_kmeans_recovers_well_separated_groups(metric):
    vectors, truth = _grouped_channels([4, 3, 5, 2])
    labels, centroids = kmeans(vectors, n_clusters=4, metric=metric, seed=0)
    assert centroids.shape == (4, vectors.shape[1])
    assert _same_partition(labels, truth)


def test_kmeans_is_deterministic_for_a_seed():
    vectors = torch.randn(30, 8, generator=torch.Generator().manual_seed(1))
    a, _ = kmeans(vectors, 5, seed=3)
    b, _ = kmeans(vectors, 5, seed=3)
    assert torch.equal(a, b)


def test_kmeans_never_leaves_a_cluster_empty_even_with_duplicate_points():
    """Duplicate channels (e.g. zero-initialized ones) make K-Means++ run out
    of distinct seeds and can starve a centroid -- every cluster must still
    end up with at least one member, or the "one representative per cluster"
    rule would silently keep fewer channels than requested."""
    vectors = torch.cat([torch.zeros(6, 4), torch.ones(2, 4)])
    labels, _ = kmeans(vectors, n_clusters=5, seed=0)
    assert torch.bincount(labels, minlength=5).min() >= 1


def test_manhattan_centroid_is_the_median_not_the_mean():
    """The mean minimizes squared L2, not L1: with one outlier the two differ
    sharply, and only the median is the correct K-Medians update."""
    vectors = torch.tensor([[0.0], [1.0], [2.0], [100.0]])
    _, centroids = kmeans(vectors, n_clusters=1, metric="manhattan")
    assert centroids.item() == pytest.approx(1.0)  # torch's lower median of {0,1,2,100}
    _, centroids = kmeans(vectors, n_clusters=1, metric="euclidean")
    assert centroids.item() == pytest.approx(25.75)


def test_kmeans_rejects_bad_arguments():
    with pytest.raises(ValueError, match="n_clusters"):
        kmeans(torch.randn(3, 2), n_clusters=4)
    with pytest.raises(ValueError, match="metric"):
        kmeans(torch.randn(3, 2), n_clusters=2, metric="hamming")


def test_kmeans_prune_keeps_exactly_the_highest_l1_channel_per_cluster():
    """Default n_clusters = out - prune_amount: each near-duplicate group
    collapses to its single strongest (highest-L1) member."""
    sizes = [3, 2, 4, 1]
    vectors, truth = _grouped_channels(sizes, noise=0.05)
    weight = vectors.reshape(len(truth), 4, 2, 2)  # conv-shaped weight, 16 = 4*2*2
    n_prune = len(truth) - len(sizes)

    pruned = kmeans_prune_indices(weight, n_prune, metric="manhattan")

    l1 = vectors.abs().sum(dim=1)
    expected_keep = sorted(
        int(members[l1[members].argmax()]) for members in (torch.nonzero(truth == g).flatten() for g in range(len(sizes)))
    )
    keep = sorted(set(range(len(truth))) - set(pruned.tolist()))
    assert keep == expected_keep
    assert torch.equal(pruned, pruned.sort().values)


def test_kmeans_prune_with_fewer_clusters_prunes_lowest_l1_non_representatives():
    """Brute-force check of the coarser-clustering rule: every cluster keeps
    its highest-L1 member, and the prune set is the lowest-L1 channels among
    everyone else."""
    torch.manual_seed(0)
    weight = torch.randn(20, 3, 3, 3)
    n_clusters, n_prune = 4, 7

    pruned = kmeans_prune_indices(weight, n_prune, n_clusters=n_clusters, metric="euclidean", seed=5)

    labels, _ = kmeans(weight.reshape(20, -1), n_clusters, metric="euclidean", seed=5)
    l1 = weight.reshape(20, -1).abs().sum(dim=1)
    reps = {int(m[l1[m].argmax()]) for m in (torch.nonzero(labels == c).flatten() for c in range(n_clusters))}
    others = sorted((i for i in range(20) if i not in reps), key=lambda i: l1[i].item())
    assert pruned.tolist() == sorted(others[:n_prune])
    assert reps.isdisjoint(pruned.tolist())


def test_kmeans_prune_refuses_to_empty_a_cluster():
    with pytest.raises(ValueError, match="representative"):
        kmeans_prune_indices(torch.randn(10, 4), prune_amount=8, n_clusters=3)


def test_kmeans_prune_zero_amount_is_a_no_op():
    assert kmeans_prune_indices(torch.randn(6, 4), 0).numel() == 0


def test_kmeans_prune_accepts_linear_weights():
    pruned = kmeans_prune_indices(torch.randn(12, 5), 4, metric="cosine")
    assert pruned.numel() == 4


def test_dispatcher_matches_saliency_path_for_score_methods():
    torch.manual_seed(0)
    weight = torch.randn(16, 8, 3, 3)
    for method in ("max_k", "l1", "l2"):
        expected = select_prune_indices(compute_score(weight, method=method), 5)
        assert torch.equal(select_prune_indices_by_method(weight, 5, method=method), expected)
    assert torch.equal(
        select_prune_indices_by_method(weight, 5, method="kmeans", metric="manhattan"),
        kmeans_prune_indices(weight, 5, metric="manhattan"),
    )


def test_prune_model_with_kmeans_matches_zeroing_the_same_channels():
    """End to end through DependencyGraph: pruning with method="kmeans" must
    give the same output as the unpruned model with exactly those channels'
    contribution removed."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.Conv2d(12, 6, 3, padding=1)).eval()
    x = torch.randn(2, 3, 8, 8)

    expected_pruned = kmeans_prune_indices(model[0].weight, 4, metric="manhattan")
    reference = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.Conv2d(12, 6, 3, padding=1)).eval()
    reference.load_state_dict(model.state_dict())
    with torch.no_grad():
        reference[0].weight[expected_pruned] = 0
        reference[0].bias[expected_pruned] = 0
        ref_out = reference(x)

    prune_model(model, x, "0", prune_fraction=4 / 12, method="kmeans", metric="manhattan")

    assert model[0].out_channels == 8
    assert model[2].in_channels == 8
    with torch.no_grad():
        assert torch.allclose(model(x), ref_out, atol=1e-5)
