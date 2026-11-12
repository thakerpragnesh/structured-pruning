import pytest
import torch
import torch.nn as nn

from prunelib.transformer import (
    ActivationAwareHeadSelector,
    GradientHeadSelector,
    HeadContext,
    HeadStats,
    OVNormHeadSelector,
    RedundantHeadSelector,
    ov_norms,
    ov_similarity,
)

HIDDEN, HEADS, D = 16, 4, 4


def make_ctx(stats=None, gradient_importance=None, seed=0):
    torch.manual_seed(seed)
    query, key, value, output = (nn.Linear(HIDDEN, HIDDEN) for _ in range(4))
    return HeadContext.from_layers(
        query, key, value, output, HEADS, stats=stats, gradient_importance=gradient_importance
    ), (query, key, value, output)


def fake_stats(rms, std=None):
    rms = torch.tensor(rms, dtype=torch.float32)
    return HeadStats(
        context_mean=torch.zeros(len(rms), D), contribution_rms=rms,
        contribution_std=rms if std is None else torch.tensor(std, dtype=torch.float32), tokens=10,
    )


# --- OV-circuit math --------------------------------------------------------

def test_ov_norms_match_the_explicit_ov_product():
    ctx, (_, _, value, output) = make_ctx()
    norms = ov_norms(value.weight, output.weight, HEADS)
    for h in range(HEADS):
        ov = output.weight[:, h * D:(h + 1) * D] @ value.weight[h * D:(h + 1) * D]
        assert norms[h].item() == pytest.approx(torch.linalg.matrix_norm(ov).item(), rel=1e-5)


def test_ov_similarity_matches_the_explicit_cosine():
    _, (_, _, value, output) = make_ctx()
    sim = ov_similarity(value.weight, output.weight, HEADS)
    flat = torch.stack([
        (output.weight[:, h * D:(h + 1) * D] @ value.weight[h * D:(h + 1) * D]).reshape(-1)
        for h in range(HEADS)
    ])
    unit = torch.nn.functional.normalize(flat, dim=1)
    assert torch.allclose(sim, unit @ unit.t(), atol=1e-5)


def test_heads_with_the_same_ov_circuit_have_similarity_one():
    _, (_, _, value, output) = make_ctx()
    with torch.no_grad():
        value.weight[D:2 * D] = value.weight[:D]
        output.weight[:, D:2 * D] = output.weight[:, :D] * 3.0  # same direction, louder
    sim = ov_similarity(value.weight, output.weight, HEADS)
    assert sim[0, 1].item() == pytest.approx(1.0, abs=1e-5)


# --- importance selectors ---------------------------------------------------

def test_ov_norm_selector_drops_the_head_that_cannot_write():
    _, (query, key, value, output) = make_ctx()
    with torch.no_grad():
        output.weight[:, 2 * D:3 * D] *= 1e-4
    ctx = HeadContext.from_layers(query, key, value, output, HEADS)
    assert OVNormHeadSelector(prune_percent=25).select(ctx) == [0, 1, 3]


def test_activation_aware_selector_follows_the_measured_contribution():
    ctx, _ = make_ctx(stats=fake_stats([1.0, 0.1, 2.0, 3.0]))
    assert ActivationAwareHeadSelector(prune_percent=25).select(ctx) == [0, 2, 3]


def test_std_statistic_spares_a_loud_but_constant_head():
    """Head 3 writes a large, almost constant vector: deleting it outright
    hurts (high rms), but bias compensation restores it (low std)."""
    stats = fake_stats(rms=[1.0, 1.1, 1.2, 5.0], std=[1.0, 1.1, 1.2, 0.01])
    ctx, _ = make_ctx(stats=stats)
    assert ActivationAwareHeadSelector(prune_percent=25, statistic="rms").select(ctx) == [1, 2, 3]
    assert ActivationAwareHeadSelector(prune_percent=25, statistic="std").select(ctx) == [0, 1, 2]


def test_activation_aware_selector_needs_matching_stats():
    ctx, _ = make_ctx()
    with pytest.raises(ValueError, match="HeadCalibrator"):
        ActivationAwareHeadSelector(prune_percent=25).select(ctx)
    ctx, _ = make_ctx(stats=fake_stats([1.0, 2.0]))
    with pytest.raises(ValueError, match="2 heads but this layer has 4"):
        ActivationAwareHeadSelector(prune_percent=25).select(ctx)


def test_gradient_selector_uses_the_supplied_importance():
    ctx, _ = make_ctx(gradient_importance=torch.tensor([0.5, 0.0, 0.2, 0.9]))
    assert GradientHeadSelector(prune_percent=50).select(ctx) == [0, 3]


def test_gradient_selector_explains_a_missing_or_mismatched_importance():
    ctx, _ = make_ctx()
    with pytest.raises(ValueError, match="collect_gradient_importance"):
        GradientHeadSelector(prune_percent=25).select(ctx)
    ctx, _ = make_ctx(gradient_importance=torch.ones(3))
    with pytest.raises(ValueError, match="3 heads"):
        GradientHeadSelector(prune_percent=25).select(ctx)


def test_budget_rounds_and_floors_like_the_ffn_selectors():
    ctx, _ = make_ctx(gradient_importance=torch.tensor([4.0, 3.0, 2.0, 1.0]))
    assert len(GradientHeadSelector(prune_percent=40).select(ctx)) == 2  # round(2.4)
    assert len(GradientHeadSelector(prune_percent=30).select(ctx)) == 3  # round(2.8); int() gives 2
    assert GradientHeadSelector(prune_percent=99, min_keep=1).select(ctx) == [0]
    with pytest.raises(ValueError, match="prune_percent"):
        GradientHeadSelector(prune_percent=100)


# --- redundancy selector ----------------------------------------------------

def duplicate_head(value, output, src, dst, scale=1.0):
    with torch.no_grad():
        value.weight[dst * D:(dst + 1) * D] = value.weight[src * D:(src + 1) * D]
        output.weight[:, dst * D:(dst + 1) * D] = output.weight[:, src * D:(src + 1) * D] * scale


def test_redundant_selector_drops_the_weaker_of_a_duplicated_pair():
    _, (query, key, value, output) = make_ctx()
    duplicate_head(value, output, src=1, dst=3, scale=0.5)  # head 3 = quieter copy of head 1
    ctx = HeadContext.from_layers(query, key, value, output, HEADS)
    assert RedundantHeadSelector(prune_percent=25).select(ctx) == [0, 1, 2]


def test_redundant_selector_keeps_one_of_three_identical_heads():
    """Re-picking from survivors after each drop: a cluster of three
    duplicates loses two, never all three."""
    _, (query, key, value, output) = make_ctx()
    duplicate_head(value, output, src=0, dst=1)
    duplicate_head(value, output, src=0, dst=2)
    ctx = HeadContext.from_layers(query, key, value, output, HEADS)
    keep = RedundantHeadSelector(prune_percent=50).select(ctx)
    assert len(keep) == 2 and 3 in keep and len({0, 1, 2} & set(keep)) == 1


def test_threshold_stops_before_the_budget_when_nothing_is_redundant():
    ctx, _ = make_ctx()  # random heads: OV cosines near 0
    assert RedundantHeadSelector(prune_percent=50, threshold=0.9).select(ctx) == [0, 1, 2, 3]


def test_query_similarity_mode_reads_only_the_query_weight():
    _, (query, key, value, output) = make_ctx()
    with torch.no_grad():
        query.weight[2 * D:3 * D] = query.weight[:D] * 0.1  # head 2 = quiet copy of head 0
    ctx = HeadContext.from_layers(query, key, value, output, HEADS)
    assert RedundantHeadSelector(prune_percent=25, similarity="query").select(ctx) == [0, 1, 3]


def test_redundant_selector_validates_its_arguments():
    with pytest.raises(ValueError, match="similarity"):
        RedundantHeadSelector(prune_percent=25, similarity="value")
    with pytest.raises(ValueError, match="threshold"):
        RedundantHeadSelector(prune_percent=25, threshold=2.0)


def test_head_context_rejects_an_impossible_layout():
    query, key, value, output = (nn.Linear(HIDDEN, HIDDEN) for _ in range(4))
    with pytest.raises(ValueError, match="not divisible"):
        HeadContext.from_layers(query, key, value, output, num_heads=3)
    with pytest.raises(ValueError, match="inconsistent"):
        HeadContext.from_layers(query, key, value, nn.Linear(8, HIDDEN), num_heads=4)
