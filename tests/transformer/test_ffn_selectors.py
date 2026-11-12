import pytest
import torch
import torch.nn as nn

from prunelib.transformer import (
    ActivationAwareSelector,
    CalibrationStats,
    FFNContext,
    InOutNormSelector,
    Max3SaliencyScorer,
    SaliencySelector,
    TwinRedundancySelector,
    WeightClusterRedundancySelector,
)


def make_ctx(neurons=10, hidden=4, stats=None, cross_moments=None):
    torch.manual_seed(0)
    intermediate = nn.Linear(hidden, neurons)
    output = nn.Linear(neurons, hidden)
    return FFNContext.from_layers(intermediate, output, stats=stats, cross_moments=cross_moments)


def test_saliency_selector_keeps_the_strongest():
    ctx = make_ctx(neurons=10)
    keep = SaliencySelector(Max3SaliencyScorer(), prune_percent=40).select_keep_indices(ctx)
    assert len(keep) == 6
    scores = Max3SaliencyScorer().score(ctx.intermediate_weight)
    assert set(keep) == set(torch.topk(scores, 6).indices.tolist())


def test_keep_count_rounds_rather_than_truncates():
    """int() truncates toward zero, so it prunes one neuron more than asked
    whenever n * (1 - p/100) isn't whole -- a one-directional bias.
    """
    assert SaliencySelector(Max3SaliencyScorer(), prune_percent=40).n_keep(768) == 461
    assert int(768 * 0.6) == 460  # what the old code kept
    assert SaliencySelector(Max3SaliencyScorer(), prune_percent=30).n_keep(1024) == 717
    assert int(1024 * 0.7) == 716
    assert SaliencySelector(Max3SaliencyScorer(), prune_percent=40).n_keep(10) == 6


def test_a_layer_is_never_pruned_out_of_existence():
    selector = SaliencySelector(Max3SaliencyScorer(), prune_percent=99.9, min_keep=1)
    assert selector.n_keep(10) == 1
    ctx = make_ctx(neurons=10)
    assert len(selector.select(ctx).keep_indices) == 1


def test_prune_percent_is_validated():
    with pytest.raises(ValueError, match="must be in"):
        SaliencySelector(Max3SaliencyScorer(), prune_percent=100)
    with pytest.raises(ValueError, match="must be in"):
        SaliencySelector(Max3SaliencyScorer(), prune_percent=-1)


def test_in_out_norm_sinks_a_neuron_with_a_dead_output_column():
    """The point of the product form: a strongly-driven neuron nothing
    reads is worthless, and a W_in-only criterion cannot see that.
    """
    hidden, neurons = 4, 6
    intermediate = nn.Linear(hidden, neurons)
    output = nn.Linear(neurons, hidden)
    with torch.no_grad():
        intermediate.weight[2] *= 50.0   # loudest neuron by any W_in measure
        output.weight[:, 2] = 0.0        # ...but nothing downstream reads it
    ctx = FFNContext.from_layers(intermediate, output)

    weight_only = SaliencySelector(Max3SaliencyScorer(), prune_percent=50).select_keep_indices(ctx)
    in_out = InOutNormSelector(prune_percent=50).select_keep_indices(ctx)

    assert 2 in weight_only
    assert 2 not in in_out


def test_activation_aware_drops_a_neuron_that_never_fires():
    ctx_stats = CalibrationStats(
        mean=torch.ones(6), mean_abs=torch.ones(6), rms=torch.ones(6), tokens=100,
    )
    ctx_stats.mean_abs[4] = 0.0  # neuron 4 is well-wired but inert on real data
    ctx = make_ctx(neurons=6, stats=ctx_stats)
    keep = ActivationAwareSelector(prune_percent=20).select_keep_indices(ctx)
    assert 4 not in keep


def test_activation_aware_explains_missing_calibration():
    ctx = make_ctx(neurons=6, stats=None)
    with pytest.raises(ValueError, match="needs calibration statistics"):
        ActivationAwareSelector(prune_percent=20).select_keep_indices(ctx)


def test_stats_for_the_wrong_layer_are_caught():
    wrong = CalibrationStats(torch.ones(3), torch.ones(3), torch.ones(3), tokens=10)
    ctx = make_ctx(neurons=6, stats=wrong)
    with pytest.raises(ValueError, match="stats for 3 neurons"):
        ActivationAwareSelector(prune_percent=20).select_keep_indices(ctx)


def test_activation_aware_statistic_is_validated():
    with pytest.raises(ValueError, match="mean_abs"):
        ActivationAwareSelector(prune_percent=10, statistic="variance")


def test_twin_selector_drops_the_higher_index_of_each_pair():
    ctx = make_ctx(neurons=8)
    selection = TwinRedundancySelector([(1, 4), (2, 6)]).select(ctx)
    assert selection.keep_indices == [0, 1, 2, 3, 5, 7]
    assert selection.merge_map is None


def test_twin_chains_resolve_to_a_single_survivor():
    """Given (1,4) and (4,6), neuron 6 must be merged into 1 -- merging it
    into 4 would point at a neuron that is itself being removed.
    """
    ctx = make_ctx(neurons=8)
    selector = TwinRedundancySelector([(1, 4), (4, 6)], merge=True)
    selection = selector.select(ctx)
    assert selection.keep_indices == [0, 1, 2, 3, 5, 7]
    assert selection.merge_map[4][0] == 1
    assert selection.merge_map[6][0] == 1


def test_twin_chains_resolve_regardless_of_pair_order():
    """`JaccardTwinFinder` sorts pairs by overlap, not by index, so a chain
    can arrive as (4, 6) before (1, 4). Resolving each pair only upward from
    its lower index used to map 6 into 4 and then drop 4, leaving a merge
    into a removed neuron (a ValueError from `Selection.of`).
    """
    ctx = make_ctx(neurons=8)
    selection = TwinRedundancySelector([(4, 6), (1, 4)], merge=True).select(ctx)
    assert selection.keep_indices == [0, 1, 2, 3, 5, 7]
    assert selection.merge_map[4][0] == 1
    assert selection.merge_map[6][0] == 1


def test_twin_pairs_sharing_a_neuron_collapse_to_one_survivor():
    """(2, 5) and (1, 5) put 1, 2 and 5 in one twin group: only the lowest
    index survives. Overwriting 5's survivor used to leave 2 in the model."""
    ctx = make_ctx(neurons=6)
    selection = TwinRedundancySelector([(2, 5), (1, 5)], merge=True).select(ctx)
    assert selection.keep_indices == [0, 1, 3, 4]
    assert {d: s for d, (s, _) in selection.merge_map.items()} == {2: 1, 5: 1}


def test_twin_merge_scale_uses_calibration_when_available():
    stats = CalibrationStats(
        mean=torch.tensor([1.0, 2.0, 1.0, 1.0]), mean_abs=torch.ones(4),
        rms=torch.ones(4), tokens=10,
    )
    ctx = make_ctx(neurons=4, stats=stats)
    selection = TwinRedundancySelector([(0, 1)], merge=True).select(ctx)
    # Neuron 1 fires twice as hard as neuron 0, so its column scales by 2.
    assert selection.merge_map[1] == (0, 2.0)


def test_twin_merge_scale_falls_back_to_one_without_calibration():
    ctx = make_ctx(neurons=4, stats=None)
    selection = TwinRedundancySelector([(0, 1)], merge=True).select(ctx)
    assert selection.merge_map[1] == (0, 1.0)


def test_twin_merge_scale_prefers_least_squares_when_cross_moments_given():
    """s* = E[a_1 * a_0] / E[a_0^2] minimizes E[(a_1 - s*a_0)^2] -- the
    reconstruction error the merge is actually trying to minimize -- unlike
    the mean-matching ratio, which only guarantees the *average* residual is
    zero (that's `_bias_compensation`'s job, not the scale's).
    """
    stats = CalibrationStats(
        mean=torch.tensor([1.0, 2.0, 1.0, 1.0]), mean_abs=torch.ones(4),
        rms=torch.tensor([2.0, 3.0, 1.0, 1.0]), tokens=10,
    )
    cross_moments = {(0, 1): 5.0}  # E[a_0 * a_1]
    ctx = make_ctx(neurons=4, stats=stats, cross_moments=cross_moments)

    selection = TwinRedundancySelector([(0, 1)], merge=True).select(ctx)
    assert selection.merge_map[1] == (0, pytest.approx(5.0 / 2.0 ** 2))  # E[a0*a1] / E[a0^2]
    # Not the mean-matching answer (1.0 / 2.0) -- proves cross_moments, not
    # mean, actually drove the scale.
    assert selection.merge_map[1][1] != pytest.approx(1.0 / 2.0)


def test_twin_merge_scale_is_keyed_regardless_of_dropped_survivor_order():
    """`cross_moments` is keyed `(min, max)`. `TwinRedundancySelector` always
    drops the *higher*-indexed neuron of a pair (dropped > survivor), but
    `WeightClusterRedundancySelector` picks the highest-L1-norm cluster
    member as survivor regardless of index (dropped can be lower), so both
    orderings need covering.
    """
    stats = CalibrationStats(
        mean=torch.tensor([1.0, 2.0, 1.0, 1.0]), mean_abs=torch.ones(4),
        rms=torch.tensor([2.0, 3.0, 1.0, 1.0]), tokens=10,
    )
    cross_moments = {(0, 1): 5.0}

    twin_ctx = make_ctx(neurons=4, stats=stats, cross_moments=cross_moments)
    dropped_is_higher = TwinRedundancySelector([(0, 1)], merge=True).select(twin_ctx)
    assert dropped_is_higher.merge_map[1] == (0, pytest.approx(5.0 / 2.0 ** 2))  # E[a0*a1]/E[a0^2]

    # Same setup as test_weight_cluster_selector_merge_uses_calibration_scale:
    # neuron 0 (same direction as 1, but not the louder one) is the one
    # dropped, into survivor 1 -- the reverse index order from the twin case.
    intermediate = nn.Linear(4, 6)
    output = nn.Linear(6, 4)
    with torch.no_grad():
        intermediate.weight[1] = intermediate.weight[0] * 3.0
    cluster_stats = CalibrationStats(
        mean=torch.ones(6), mean_abs=torch.ones(6), rms=torch.tensor([2.0, 3.0, 1.0, 1.0, 1.0, 1.0]),
        tokens=10,
    )
    cluster_ctx = FFNContext.from_layers(
        intermediate, output, stats=cluster_stats, cross_moments=cross_moments,
    )
    dropped_is_lower = WeightClusterRedundancySelector(prune_percent=20, merge=True).select(cluster_ctx)
    assert dropped_is_lower.merge_map[0] == (1, pytest.approx(5.0 / 3.0 ** 2))  # E[a0*a1]/E[a1^2]


def test_twin_merge_scale_falls_back_to_mean_when_pair_missing_from_cross_moments():
    """`cross_moments` was collected for *some* pairs, just not this one --
    still a graceful fallback to the mean-matching scale, not 1.0.
    """
    stats = CalibrationStats(
        mean=torch.tensor([1.0, 2.0, 1.0, 1.0]), mean_abs=torch.ones(4),
        rms=torch.ones(4), tokens=10,
    )
    ctx = make_ctx(neurons=4, stats=stats, cross_moments={(2, 3): 1.0})
    selection = TwinRedundancySelector([(0, 1)], merge=True).select(ctx)
    assert selection.merge_map[1] == (0, 2.0)  # unchanged from the no-cross-moments behavior


def test_twin_merge_scale_falls_back_to_mean_when_least_squares_denominator_is_near_zero():
    """`E[a_survivor^2]` (rms^2) near zero is the least-squares path's own
    guard -- distinct from the mean-ratio guard, and checked first.
    """
    stats = CalibrationStats(
        mean=torch.tensor([1.0, 2.0, 1.0, 1.0]), mean_abs=torch.ones(4),
        rms=torch.tensor([1e-9, 1.0, 1.0, 1.0]), tokens=10,
    )
    ctx = make_ctx(neurons=4, stats=stats, cross_moments={(0, 1): 5.0})
    selection = TwinRedundancySelector([(0, 1)], merge=True).select(ctx)
    assert selection.merge_map[1] == (0, 2.0)  # falls back to E[a1]/E[a0] = 2.0/1.0


def test_weight_cluster_selector_validates_prune_percent():
    with pytest.raises(ValueError, match="must be in"):
        WeightClusterRedundancySelector(prune_percent=100)


def test_weight_cluster_selector_keeps_everyone_at_zero_percent():
    ctx = make_ctx(neurons=6)
    selection = WeightClusterRedundancySelector(prune_percent=0).select(ctx)
    assert selection.keep_indices == list(range(6))
    assert selection.merge_map is None


def test_weight_cluster_selector_drops_a_direction_duplicate():
    """Neuron 1 points the same way as neuron 0 (just louder) -- after
    L2-normalizing rows before clustering, the two are identical points
    and must land in the same cluster regardless of centroid init.
    """
    intermediate = nn.Linear(4, 6)
    output = nn.Linear(6, 4)
    with torch.no_grad():
        intermediate.weight[1] = intermediate.weight[0] * 3.0
    ctx = FFNContext.from_layers(intermediate, output)

    selection = WeightClusterRedundancySelector(prune_percent=20).select(ctx)
    assert len(selection.keep_indices) == 5
    assert len({0, 1} & set(selection.keep_indices)) == 1


def test_weight_cluster_selector_keeps_the_requested_count_with_duplicate_directions():
    """Eight neurons pointing in only two directions, asked to keep four.
    K-Means can only form two non-empty clusters here, and keeping one
    neuron per non-empty cluster used to keep two -- twice the cut asked for.
    """
    intermediate = nn.Linear(6, 8)
    output = nn.Linear(8, 3)
    with torch.no_grad():
        intermediate.weight[:4] = torch.ones(4, 6) * torch.arange(1, 5).unsqueeze(1)
        intermediate.weight[4:] = -torch.ones(4, 6) * torch.arange(1, 5).unsqueeze(1)
    ctx = FFNContext.from_layers(intermediate, output)

    selection = WeightClusterRedundancySelector(prune_percent=50, merge=True).select(ctx)
    assert len(selection.keep_indices) == 4
    # Each direction keeps its loudest neuron; the extra slots go to the next-loudest.
    assert {3, 7} <= set(selection.keep_indices)
    assert set(selection.merge_map).isdisjoint(selection.keep_indices)


def test_weight_cluster_selector_keeps_the_higher_l1_norm_representative():
    intermediate = nn.Linear(4, 6)
    output = nn.Linear(6, 4)
    with torch.no_grad():
        intermediate.weight[1] = intermediate.weight[0] * 3.0  # same direction, louder

    ctx = FFNContext.from_layers(intermediate, output)
    selection = WeightClusterRedundancySelector(prune_percent=20).select(ctx)
    assert 1 in selection.keep_indices
    assert 0 not in selection.keep_indices


def test_weight_cluster_selector_merge_uses_calibration_scale():
    intermediate = nn.Linear(4, 6)
    output = nn.Linear(6, 4)
    with torch.no_grad():
        intermediate.weight[1] = intermediate.weight[0] * 3.0
    stats = CalibrationStats(
        mean=torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
        mean_abs=torch.ones(6), rms=torch.ones(6), tokens=10,
    )
    ctx = FFNContext.from_layers(intermediate, output, stats=stats)

    selection = WeightClusterRedundancySelector(prune_percent=20, merge=True).select(ctx)
    # Neuron 0 is dropped into survivor 1; E[a_0] / E[a_1] = 1.0 / 2.0.
    assert selection.merge_map[0] == (1, 0.5)


def test_twin_merge_scale_guards_a_near_zero_denominator():
    """A GELU neuron's signed mean can sit near zero while it is highly
    active; an unguarded ratio would explode the merged column.
    """
    stats = CalibrationStats(
        mean=torch.tensor([1e-12, 5.0, 1.0, 1.0]), mean_abs=torch.ones(4),
        rms=torch.ones(4), tokens=10,
    )
    ctx = make_ctx(neurons=4, stats=stats)
    selection = TwinRedundancySelector([(0, 1)], merge=True).select(ctx)
    assert selection.merge_map[1] == (0, 1.0)


def test_merge_pairs_name_the_folds_not_the_twin_list():
    """A chain (1, 4), (4, 6) folds 6 into 1 -- the pair cross moments must
    be collected for, though no twin pair says so."""
    selector = TwinRedundancySelector([(4, 6), (1, 4)], merge=True)
    assert selector.merge_pairs == [(4, 1), (6, 1)]
