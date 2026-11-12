import pytest
import torch
import torch.nn as nn

from prunelib.transformer import CalibrationStats, FFNSurgeon, Selection


def make_pair(hidden=6, neurons=8, bias=True, dtype=torch.float32):
    torch.manual_seed(0)
    intermediate = nn.Linear(hidden, neurons, bias=bias, dtype=dtype)
    output = nn.Linear(neurons, hidden, bias=bias, dtype=dtype)
    return intermediate, output


def test_resize_keeps_selected_rows_and_columns():
    intermediate, output = make_pair()
    keep = [0, 2, 5]
    new_i, new_o = FFNSurgeon().resize(intermediate, output, keep)

    assert new_i.out_features == 3 and new_o.in_features == 3
    assert torch.equal(new_i.weight, intermediate.weight[keep])
    assert torch.equal(new_i.bias, intermediate.bias[keep])
    assert torch.equal(new_o.weight, output.weight[:, keep])
    assert torch.equal(new_o.bias, output.bias)


def test_keep_indices_are_sorted_so_neuron_order_is_stable():
    """Unsorted indices would silently permute the layer."""
    intermediate, output = make_pair()
    shuffled, _ = FFNSurgeon().resize(intermediate, output, [5, 0, 2])
    ordered, _ = FFNSurgeon().resize(intermediate, output, [0, 2, 5])
    assert torch.equal(shuffled.weight, ordered.weight)


def test_dtype_is_preserved():
    """Regression: a bare nn.Linear defaults to float32, which silently
    turned a half-precision layer into an fp32 one that then mismatched
    every other layer at forward time.
    """
    intermediate, output = make_pair(dtype=torch.float16)
    new_i, new_o = FFNSurgeon().resize(intermediate, output, [0, 1, 2])
    assert new_i.weight.dtype == torch.float16
    assert new_o.weight.dtype == torch.float16


def test_biasless_layers_survive_surgery():
    intermediate, output = make_pair(bias=False)
    new_i, new_o = FFNSurgeon().resize(intermediate, output, [1, 3])
    assert new_i.bias is None and new_o.bias is None


def test_requires_grad_is_preserved():
    intermediate, output = make_pair()
    intermediate.weight.requires_grad_(False)
    new_i, new_o = FFNSurgeon().resize(intermediate, output, [0, 1])
    assert new_i.weight.requires_grad is False
    assert new_o.weight.requires_grad is True


def test_merging_identical_neurons_is_output_preserving():
    """The strongest guarantee merging can offer: if two neurons are
    genuinely interchangeable, folding one into the other changes the
    layer's output on *every* input, not merely on average.
    """
    intermediate, output = make_pair(hidden=6, neurons=8)
    with torch.no_grad():
        intermediate.weight[3] = intermediate.weight[1]
        intermediate.bias[3] = intermediate.bias[1]

    x = torch.randn(10, 6)
    acts = torch.nn.functional.gelu(intermediate(x))
    before = output(acts)

    keep = [i for i in range(8) if i != 3]
    selection = Selection.of(keep, merge_map={3: (1, 1.0)}, num_neurons=8)
    new_i, new_o = FFNSurgeon().resize(intermediate, output, selection)
    after = new_o(torch.nn.functional.gelu(new_i(x)))

    assert torch.allclose(before, after, atol=1e-5)


def test_merging_without_merge_map_loses_the_neuron():
    """Control for the test above: plain deletion is *not* output-preserving."""
    intermediate, output = make_pair(hidden=6, neurons=8)
    with torch.no_grad():
        intermediate.weight[3] = intermediate.weight[1]
        intermediate.bias[3] = intermediate.bias[1]

    x = torch.randn(10, 6)
    before = output(torch.nn.functional.gelu(intermediate(x)))
    new_i, new_o = FFNSurgeon().resize(intermediate, output, [i for i in range(8) if i != 3])
    after = new_o(torch.nn.functional.gelu(new_i(x)))

    assert not torch.allclose(before, after, atol=1e-3)


def test_bias_compensation_preserves_the_mean_output():
    intermediate, output = make_pair(hidden=6, neurons=8)
    x = torch.randn(200, 6)
    acts = torch.nn.functional.gelu(intermediate(x)).detach()
    stats = CalibrationStats(
        mean=acts.mean(dim=0), mean_abs=acts.abs().mean(dim=0),
        rms=acts.square().mean(dim=0).sqrt(), tokens=200,
    )
    mean_before = output(acts).mean(dim=0)

    keep = [0, 1, 2, 3]
    uncompensated_i, uncompensated_o = FFNSurgeon().resize(intermediate, output, keep)
    compensated_i, compensated_o = FFNSurgeon().resize(
        intermediate, output, keep, stats=stats, compensate_bias=True
    )

    mean_plain = uncompensated_o(torch.nn.functional.gelu(uncompensated_i(x))).mean(dim=0)
    mean_fixed = compensated_o(torch.nn.functional.gelu(compensated_i(x))).mean(dim=0)

    assert torch.allclose(mean_fixed, mean_before, atol=1e-5)
    # And it genuinely helped -- the plain cut shifted the mean.
    assert (mean_plain - mean_before).abs().max() > (mean_fixed - mean_before).abs().max()


def test_least_squares_merge_scale_reduces_pointwise_error_versus_mean_matching():
    """Both scales make bias compensation preserve the *mean* output exactly
    (that's `_bias_compensation`'s job, not the scale's) -- the least-squares
    scale `s* = E[a_dropped*a_survivor]/E[a_survivor^2]`'s actual payoff is a
    smaller *pointwise* (per-token) reconstruction error, which comparing
    means alone can't distinguish. See `ffn_selectors._merge_scale`.
    """
    intermediate, output = make_pair(hidden=6, neurons=8)
    x = torch.randn(500, 6)
    acts = torch.nn.functional.gelu(intermediate(x)).detach()

    survivor, dropped = 1, 3
    mean = acts.mean(dim=0)
    rms = acts.square().mean(dim=0).sqrt()
    stats = CalibrationStats(mean=mean, mean_abs=acts.abs().mean(dim=0), rms=rms, tokens=500)
    cross = (acts[:, survivor] * acts[:, dropped]).mean().item()

    mean_scale = float(mean[dropped] / mean[survivor])
    least_squares_scale = cross / float(rms[survivor]) ** 2
    assert mean_scale != pytest.approx(least_squares_scale)  # the two genuinely differ here

    keep = [i for i in range(8) if i != dropped]
    before = output(acts)

    def merged_output(scale):
        selection = Selection.of(keep, merge_map={dropped: (survivor, scale)}, num_neurons=8)
        new_i, new_o = FFNSurgeon().resize(intermediate, output, selection, stats=stats, compensate_bias=True)
        return new_o(torch.nn.functional.gelu(new_i(x)))

    mean_matched = merged_output(mean_scale)
    least_squares = merged_output(least_squares_scale)

    # Both preserve the mean output...
    assert torch.allclose(mean_matched.mean(dim=0), before.mean(dim=0), atol=1e-4)
    assert torch.allclose(least_squares.mean(dim=0), before.mean(dim=0), atol=1e-4)
    # ...but least-squares is the one actually minimizing per-token error.
    mean_matched_err = (mean_matched - before).square().sum(dim=1).mean()
    least_squares_err = (least_squares - before).square().sum(dim=1).mean()
    assert least_squares_err < mean_matched_err


def test_compensation_and_merge_do_not_double_count():
    """A merge already replays the dropped neuron's contribution, so
    compensation must only add the residual -- zero when the merge scale
    was E[a_j]/E[a_i].
    """
    intermediate, output = make_pair(hidden=6, neurons=8)
    x = torch.randn(200, 6)
    acts = torch.nn.functional.gelu(intermediate(x)).detach()
    stats = CalibrationStats(
        mean=acts.mean(dim=0), mean_abs=acts.abs().mean(dim=0),
        rms=acts.square().mean(dim=0).sqrt(), tokens=200,
    )
    mean_before = output(acts).mean(dim=0)

    keep = [i for i in range(8) if i != 3]
    scale = float(stats.mean[3] / stats.mean[1])
    selection = Selection.of(keep, merge_map={3: (1, scale)}, num_neurons=8)
    new_i, new_o = FFNSurgeon().resize(
        intermediate, output, selection, stats=stats, compensate_bias=True
    )
    mean_after = new_o(torch.nn.functional.gelu(new_i(x))).mean(dim=0)
    assert torch.allclose(mean_after, mean_before, atol=1e-5)


def test_compensation_requires_stats_and_a_bias():
    intermediate, output = make_pair()
    with pytest.raises(ValueError, match="requires calibration stats"):
        FFNSurgeon().resize(intermediate, output, [0, 1], compensate_bias=True)

    biasless_i, biasless_o = make_pair(bias=False)
    stats = CalibrationStats(torch.zeros(8), torch.zeros(8), torch.zeros(8), tokens=1)
    with pytest.raises(ValueError, match="no bias"):
        FFNSurgeon().resize(biasless_i, biasless_o, [0, 1], stats=stats, compensate_bias=True)


def test_mismatched_stats_are_rejected():
    intermediate, output = make_pair(neurons=8)
    stats = CalibrationStats(torch.zeros(5), torch.zeros(5), torch.zeros(5), tokens=1)
    with pytest.raises(ValueError, match="cover 5 neurons"):
        FFNSurgeon().resize(intermediate, output, [0, 1], stats=stats, compensate_bias=True)


def test_out_of_range_and_empty_selections_are_rejected():
    intermediate, output = make_pair(neurons=8)
    with pytest.raises(ValueError, match="out-of-range"):
        FFNSurgeon().resize(intermediate, output, [0, 99])
    with pytest.raises(ValueError, match="zero neurons"):
        FFNSurgeon().resize(intermediate, output, [])


def test_mismatched_ffn_pair_is_rejected():
    intermediate = nn.Linear(6, 8)
    wrong_output = nn.Linear(5, 6)
    with pytest.raises(ValueError, match="inconsistent"):
        FFNSurgeon().resize(intermediate, wrong_output, [0, 1])


def test_selection_validates_merge_targets():
    with pytest.raises(ValueError, match="must itself survive"):
        Selection.of([0, 1], merge_map={2: (3, 1.0)}, num_neurons=4)
    with pytest.raises(ValueError, match="cannot both survive"):
        Selection.of([0, 1, 2], merge_map={2: (0, 1.0)}, num_neurons=4)
