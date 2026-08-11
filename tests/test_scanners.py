import torch

from prunelib.scanners import CoActivationScanner, pairwise_distance_matrix


def test_manhattan_matches_manual_computation():
    vectors = torch.tensor([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
    dist = pairwise_distance_matrix(vectors, metric="manhattan")
    # |0-3|+|0-4| = 7, |0-1|+|0-1| = 2, |3-1|+|4-1| = 5
    expected = torch.tensor([[0.0, 7.0, 2.0], [7.0, 0.0, 5.0], [2.0, 5.0, 0.0]])
    assert torch.allclose(dist, expected)


def test_euclidean_matches_manual_computation():
    vectors = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    dist = pairwise_distance_matrix(vectors, metric="euclidean")
    assert torch.isclose(dist[0, 1], torch.tensor(5.0))


def test_cosine_identical_vectors_have_zero_distance():
    vectors = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 0.0]])
    dist = pairwise_distance_matrix(vectors, metric="cosine")
    # rows 0 and 1 are parallel (scaled copies) -> cosine distance ~0
    assert torch.isclose(dist[0, 1], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(torch.diagonal(dist), torch.zeros(3), atol=1e-5)


def test_coactivation_excludes_high_firing_rate_units():
    # unit 0 fires on every token (should be excluded), units 1 and 2 fire
    # identically on a minority of tokens (should show similarity 1.0)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    scanner = CoActivationScanner(firing_rate_ceiling=0.9)
    sim = scanner.jaccard(mask)

    assert sim[0, 1] == 0.0 and sim[0, 2] == 0.0  # unit 0 excluded entirely
    assert torch.isclose(sim[1, 2], torch.tensor(1.0))  # identical firing pattern
