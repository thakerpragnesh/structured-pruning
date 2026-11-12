import pytest
import torch

from prunelib.transformer import ActivationRecorder, BertLayerAdapter, FFNCalibrator


def test_stats_match_a_hand_computed_reference(model, batches):
    stats = FFNCalibrator(model).collect(batches, layer_idx=0)

    # Recompute from the same tokens, bypassing the hook entirely.
    adapter = BertLayerAdapter(model)
    captured = []
    handle = adapter.get_activation_module(0).register_forward_hook(
        lambda m, i, out: captured.append(out.detach())
    )
    with torch.no_grad():
        for batch in batches:
            model(**batch)
    handle.remove()

    valid = torch.cat([
        acts.reshape(-1, acts.shape[-1])[b["attention_mask"].reshape(-1).bool()]
        for acts, b in zip(captured, batches)
    ])
    assert stats.tokens == valid.shape[0]
    assert torch.allclose(stats.mean, valid.mean(dim=0), atol=1e-5)
    assert torch.allclose(stats.mean_abs, valid.abs().mean(dim=0), atol=1e-5)
    assert torch.allclose(stats.rms, valid.square().mean(dim=0).sqrt(), atol=1e-5)


def test_cross_moments_match_a_hand_computed_reference(model, batches):
    pairs = [(0, 3), (1, 2)]
    cross = FFNCalibrator(model).collect_cross_moments(batches, layer_idx=0, pairs=pairs)

    adapter = BertLayerAdapter(model)
    captured = []
    handle = adapter.get_activation_module(0).register_forward_hook(
        lambda m, i, out: captured.append(out.detach())
    )
    with torch.no_grad():
        for batch in batches:
            model(**batch)
    handle.remove()

    valid = torch.cat([
        acts.reshape(-1, acts.shape[-1])[b["attention_mask"].reshape(-1).bool()]
        for acts, b in zip(captured, batches)
    ])
    for i, j in pairs:
        expected = (valid[:, i] * valid[:, j]).mean().item()
        assert cross[(i, j)] == pytest.approx(expected, abs=1e-5)


def test_cross_moments_diagonal_matches_rms_squared(model, batches):
    """`_merge_scale`'s least-squares denominator is `E[a_i^2] = rms_i^2` --
    pin that a same-neuron "pair" agrees with ordinary `collect()`, since
    the two are computed by entirely separate code paths.
    """
    stats = FFNCalibrator(model).collect(batches, layer_idx=0)
    cross = FFNCalibrator(model).collect_cross_moments(batches, layer_idx=0, pairs=[(2, 2)])
    assert cross[(2, 2)] == pytest.approx(float(stats.rms[2]) ** 2, abs=1e-5)


def test_cross_moments_deduplicates_and_ignores_pair_order(model, batches):
    calibrator = FFNCalibrator(model)
    deduped = calibrator.collect_cross_moments(batches, layer_idx=0, pairs=[(0, 1), (1, 0), (0, 1)])
    assert set(deduped) == {(0, 1)}


def test_cross_moments_empty_pairs_returns_empty_dict(model, batches):
    assert FFNCalibrator(model).collect_cross_moments(batches, layer_idx=0, pairs=[]) == {}


def test_cross_moments_empty_calibration_is_an_explicit_error(model):
    with pytest.raises(ValueError, match="no tokens"):
        FFNCalibrator(model).collect_cross_moments([], layer_idx=0, pairs=[(0, 1)])


def test_padding_tokens_are_excluded(model, batches):
    """Counting [PAD] positions would drag every mean toward whatever the
    model emits on padding -- an artifact of batching, not of the data.
    """
    masked = FFNCalibrator(model).collect(batches, layer_idx=0)
    unmasked_batches = [{"input_ids": b["input_ids"]} for b in batches]
    unmasked = FFNCalibrator(model).collect(unmasked_batches, layer_idx=0)

    assert masked.tokens == 9       # 3 + 2 + 4 real tokens
    assert unmasked.tokens == 12    # 4 + 4 + 4 padded positions
    assert not torch.allclose(masked.mean, unmasked.mean)


def test_calibration_covers_every_neuron(model, batches):
    stats = FFNCalibrator(model).collect(batches, layer_idx=0)
    assert stats.num_neurons == model.config.intermediate_size


def test_max_batches_limits_the_pass(model, batches):
    stats = FFNCalibrator(model).collect(batches, layer_idx=0, max_batches=1)
    assert stats.tokens == 5  # first batch only


def test_empty_calibration_is_an_explicit_error(model):
    with pytest.raises(ValueError, match="no tokens"):
        FFNCalibrator(model).collect([], layer_idx=0)


def test_hook_is_removed_even_when_forward_raises(model):
    adapter = BertLayerAdapter(model)
    module = adapter.get_activation_module(0)
    before = len(module._forward_hooks)
    bad = [{"input_ids": torch.tensor([[999999]])}]  # out-of-range embedding index
    with pytest.raises(Exception):
        FFNCalibrator(model).collect(bad, layer_idx=0)
    assert len(module._forward_hooks) == before


def test_calibration_restores_training_mode(model, batches):
    model.train()
    FFNCalibrator(model).collect(batches, layer_idx=0)
    assert model.training is True


def test_stats_select_subsets_neurons(model, batches):
    stats = FFNCalibrator(model).collect(batches, layer_idx=0)
    subset = stats.select([0, 3, 5])
    assert subset.num_neurons == 3
    assert torch.equal(subset.mean, stats.mean[[0, 3, 5]])
    assert subset.tokens == stats.tokens


def test_recorder_returns_a_bool_firing_matrix(model, batches):
    fires = ActivationRecorder(model, tokenizer=None).record_batches(batches, layer_idx=0)
    assert fires.dtype == torch.bool
    assert fires.shape == (9, model.config.intermediate_size)


def test_recorder_masks_padding(model, batches):
    """Padding rows would make every neuron look co-active with every
    other on [PAD], manufacturing twin pairs that don't exist.
    """
    masked = ActivationRecorder(model, tokenizer=None).record_batches(batches, layer_idx=0)
    unmasked_batches = [{"input_ids": b["input_ids"]} for b in batches]
    unmasked = ActivationRecorder(model, tokenizer=None).record_batches(unmasked_batches, layer_idx=0)
    assert masked.shape[0] == 9
    assert unmasked.shape[0] == 12


def test_recorder_agrees_with_calibration_on_which_neurons_fire(model, batches):
    """Firing is read post-activation now; for GELU that agrees with the
    pre-activation sign, so the criterion is unchanged -- this pins that.
    """
    fires = ActivationRecorder(model, tokenizer=None).record_batches(batches, layer_idx=0)
    adapter = BertLayerAdapter(model)
    captured = []
    handle = adapter.get_ffn(0)[0].register_forward_hook(
        lambda m, i, out: captured.append(out.detach())
    )
    with torch.no_grad():
        for batch in batches:
            model(**batch)
    handle.remove()
    pre = torch.cat([
        a.reshape(-1, a.shape[-1])[b["attention_mask"].reshape(-1).bool()]
        for a, b in zip(captured, batches)
    ])
    assert torch.equal(fires, pre > 0)


def test_recorder_is_reusable_across_layers(model, batches):
    recorder = ActivationRecorder(model, tokenizer=None)
    first = recorder.record_batches(batches, layer_idx=0)
    second = recorder.record_batches(batches, layer_idx=1)
    # Regression: activations used to accumulate on the instance, so a
    # second call returned the first call's rows concatenated on.
    assert first.shape == second.shape


def test_collect_many_matches_collect_per_layer_in_one_pass(model, batches):
    calibrator = FFNCalibrator(model)
    calls = {"n": 0}
    original_forward = model.forward

    def counting_forward(*args, **kwargs):
        calls["n"] += 1
        return original_forward(*args, **kwargs)

    model.forward = counting_forward
    many = calibrator.collect_many(batches)
    assert calls["n"] == len(batches)  # one pass for all three layers
    model.forward = original_forward

    assert sorted(many) == [0, 1, 2]
    for idx in range(3):
        single = calibrator.collect(batches, layer_idx=idx)
        assert torch.allclose(many[idx].mean, single.mean)
        assert torch.allclose(many[idx].rms, single.rms)
        assert many[idx].tokens == single.tokens
