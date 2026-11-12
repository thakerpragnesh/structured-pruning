import pytest
import torch

from prunelib.transformer import JaccardTwinFinder


def test_dead_neurons_are_reported():
    """Regression: dead neurons were previously undiscoverable.

    With no firings, intersection and union are both 0; the divide-by-zero
    guard turned that into a similarity of 0, which never cleared the
    threshold, so the `dead` list was always empty.
    """
    fires = torch.zeros(10, 4)
    fires[:, 1] = 1.0
    fires[:5, 2] = 1.0
    twins, dead = JaccardTwinFinder().find(fires, threshold=0.95)
    assert dead == [0, 3]
    assert twins == []


def test_identical_neurons_are_twins_with_overlap_one():
    fires = torch.zeros(20, 3)
    pattern = torch.randint(0, 2, (20,)).float()
    fires[:, 0] = pattern
    fires[:, 2] = pattern
    fires[:, 1] = 1.0 - pattern
    twins, dead = JaccardTwinFinder().find(fires, threshold=0.95)
    assert dead == []
    assert len(twins) == 1
    i, j, overlap = twins[0]
    assert (i, j) == (0, 2)
    assert overlap == pytest.approx(1.0)


def test_jaccard_is_intersection_over_union():
    # A fires on tokens {0,1,2}, B on {1,2,3} -> |∩|=2, |∪|=4 -> 0.5
    fires = torch.tensor([
        [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0],
    ])
    twins, _ = JaccardTwinFinder().find(fires, threshold=0.5)
    assert twins[0][2] == pytest.approx(0.5)
    # The old (incorrect) formula intersection/(|A|+|B|) would give 2/6.
    assert twins[0][2] != pytest.approx(2 / 6)


def test_pairs_are_upper_triangular_and_sorted_by_overlap():
    fires = torch.randint(0, 2, (50, 8)).float()
    twins, _ = JaccardTwinFinder().find(fires, threshold=0.01)
    assert all(i < j for i, j, _ in twins)
    overlaps = [o for _, _, o in twins]
    assert overlaps == sorted(overlaps, reverse=True)


def test_accepts_bool_input_and_caps_pairs():
    fires = torch.randint(0, 2, (30, 6)).to(torch.bool)
    twins, _ = JaccardTwinFinder().find(fires, threshold=0.01, max_pairs=2)
    assert len(twins) <= 2


def test_rejects_bad_shape_and_threshold():
    with pytest.raises(ValueError, match="tokens, neurons"):
        JaccardTwinFinder().find(torch.zeros(5), threshold=0.9)
    with pytest.raises(ValueError, match="threshold"):
        JaccardTwinFinder().find(torch.zeros(5, 2), threshold=0.0)
