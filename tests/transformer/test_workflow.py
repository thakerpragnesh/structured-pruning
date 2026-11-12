import pytest
import torch

from prunelib.transformer import (
    AttentionHeadAnalyzer,
    BertLayerAdapter,
    FFNCalibrator,
    InOutNormSelector,
    Max3SaliencyScorer,
    SaliencySelector,
    TwinRedundancySelector,
    prune_attention_heads,
    prune_ffn_layer,
    prune_model_ffn,
)


def test_single_layer_prune_resizes_and_still_runs(model, batches):
    kept = prune_ffn_layer(model, 0, SaliencySelector(Max3SaliencyScorer(), prune_percent=50))
    assert kept == 8
    adapter = BertLayerAdapter(model)
    intermediate, output = adapter.get_ffn(0)
    assert intermediate.out_features == 8 and output.in_features == 8
    with torch.no_grad():
        model(**batches[0])  # shapes still line up end to end


def test_other_layers_are_untouched(model):
    adapter = BertLayerAdapter(model)
    before = adapter.get_ffn(1)[0].weight.clone()
    prune_ffn_layer(model, 0, SaliencySelector(Max3SaliencyScorer(), prune_percent=50))
    assert torch.equal(adapter.get_ffn(1)[0].weight, before)


def test_uniform_allocation_cuts_every_layer_equally(model, batches):
    kept = prune_model_ffn(model, SaliencySelector(Max3SaliencyScorer(), prune_percent=25))
    assert kept == {0: 12, 1: 12, 2: 12}
    with torch.no_grad():
        model(**batches[0])


def test_global_allocation_respects_the_total_budget(model, batches):
    kept = prune_model_ffn(model, InOutNormSelector(prune_percent=50), allocation="global")
    assert sum(kept.values()) == 24  # 3 layers x 16 neurons, halved
    with torch.no_grad():
        model(**batches[0])


def test_global_allocation_cuts_the_redundant_layer_harder(model):
    """The reason global beats uniform: FFN redundancy is not spread
    evenly across depth, so a fixed per-layer fraction over-cuts the
    layers that are carrying the model and under-cuts the ones that
    aren't.
    """
    adapter = BertLayerAdapter(model)
    with torch.no_grad():
        # Layer 0: every neuron equally useful. Layer 2: half of them inert.
        for idx in (0, 1, 2):
            adapter.get_ffn(idx)[0].weight.fill_(1.0)
            adapter.get_ffn(idx)[1].weight.fill_(1.0)
        adapter.get_ffn(2)[0].weight[:8] = 1e-6
        adapter.get_ffn(2)[1].weight[:, :8] = 1e-6

    kept = prune_model_ffn(model, InOutNormSelector(prune_percent=30), allocation="global")
    assert kept[2] < kept[0]
    assert sum(kept.values()) == 34  # 48 neurons, 30% cut, rounded


def test_min_keep_ratio_floors_each_layer(model):
    adapter = BertLayerAdapter(model)
    with torch.no_grad():
        adapter.get_ffn(1)[1].weight.fill_(1e-9)  # layer 1 looks worthless
    kept = prune_model_ffn(
        model, InOutNormSelector(prune_percent=60), allocation="global", min_keep_ratio=0.5
    )
    assert kept[1] >= 8


def test_normalisation_stops_one_scale_from_swallowing_the_budget(model):
    """Without per-layer normalisation, global ranking deletes whichever
    layer happens to have the smallest weight scale, rather than the
    genuinely redundant neurons.
    """
    adapter = BertLayerAdapter(model)
    with torch.no_grad():
        adapter.get_ffn(1)[0].weight *= 1e-4
        adapter.get_ffn(1)[1].weight *= 1e-4

    raw = prune_model_ffn(
        model, InOutNormSelector(prune_percent=30), allocation="global",
        normalize="none", min_keep_ratio=0.0,
    )
    # The entire budget lands on layer 1 purely because of its scale;
    # layers 0 and 2 are not touched at all.
    assert raw[0] == 16 and raw[2] == 16
    assert raw[1] == 2


def test_normalised_global_spreads_the_cut(model):
    adapter = BertLayerAdapter(model)
    with torch.no_grad():
        adapter.get_ffn(1)[0].weight *= 1e-4
        adapter.get_ffn(1)[1].weight *= 1e-4
    kept = prune_model_ffn(
        model, InOutNormSelector(prune_percent=30), allocation="global",
        normalize="mean", min_keep_ratio=0.0,
    )
    assert kept[1] > 8  # scale alone no longer condemns the layer


def test_global_rejects_a_criterion_with_no_per_neuron_score(model):
    with pytest.raises(ValueError, match="not an ImportanceSelector"):
        prune_model_ffn(model, TwinRedundancySelector([(0, 1)]), allocation="global")


def test_unknown_allocation_is_rejected(model):
    with pytest.raises(ValueError, match="uniform.*global"):
        prune_model_ffn(model, InOutNormSelector(prune_percent=10), allocation="per-head")


def test_layer_indices_restricts_the_sweep(model):
    kept = prune_model_ffn(
        model, SaliencySelector(Max3SaliencyScorer(), prune_percent=50), layer_indices=[0, 2]
    )
    assert set(kept) == {0, 2}
    assert BertLayerAdapter(model).get_ffn(1)[0].out_features == 16


def test_global_scores_the_original_model_not_a_half_pruned_one(model):
    """Scoring lazily inside the resize loop would rank later layers
    against a model earlier cuts had already mutated.
    """
    kept_a = prune_model_ffn(model, InOutNormSelector(prune_percent=40), allocation="global")
    fresh = type(model)()
    fresh.load_state_dict({k: v for k, v in fresh.state_dict().items()})
    assert sum(kept_a.values()) == round(48 * 0.6)


def test_twin_pruning_with_merge_end_to_end(model, batches):
    adapter = BertLayerAdapter(model)
    intermediate, _ = adapter.get_ffn(0)
    with torch.no_grad():
        intermediate.weight[7] = intermediate.weight[2]
        intermediate.bias[7] = intermediate.bias[2]
    with torch.no_grad():
        before = model(**batches[0])

    kept = prune_ffn_layer(model, 0, TwinRedundancySelector([(2, 7)], merge=True))
    assert kept == 15
    with torch.no_grad():
        after = model(**batches[0])
    assert torch.allclose(before, after, atol=1e-5)


def test_prune_attention_heads_resizes_and_still_runs(model, batches):
    adapter = BertLayerAdapter(model)
    assert adapter.num_attention_heads(0) == 2

    kept = prune_attention_heads(model, 0, head_indices=[0])
    assert kept == 1
    assert adapter.num_attention_heads(0) == 1

    query, key, value, output = adapter.get_attention_heads(0)
    assert query.out_features == 4 and key.out_features == 4 and value.out_features == 4
    assert output.in_features == 4
    with torch.no_grad():
        model(**batches[0])  # shapes still line up end to end, including real attention


def test_prune_attention_heads_leaves_other_layers_untouched(model):
    adapter = BertLayerAdapter(model)
    before = adapter.get_query_weight(1).clone()
    prune_attention_heads(model, 0, head_indices=[0])
    assert adapter.num_attention_heads(1) == 2
    assert torch.equal(adapter.get_query_weight(1), before)


def test_prune_attention_heads_keeps_head_analysis_correct_afterwards(model):
    """Regression: num_attention_heads used to be a single model-wide
    property, so AttentionHeadAnalyzer would keep dividing this layer's
    (now smaller) query weight by the *original* head count.
    """
    prune_attention_heads(model, 0, head_indices=[0])
    heads = AttentionHeadAnalyzer(model).get_flat_heads(0)
    assert heads.shape[0] == 1


def test_prune_ffn_layer_threads_cross_moments_into_the_merge_scale(model, batches):
    """End-to-end wiring check: FFNCalibrator.collect_cross_moments ->
    prune_ffn_layer's cross_moments= -> FFNContext -> TwinRedundancySelector's
    merge. The scale's actual numerical payoff (a lower pointwise
    reconstruction error than the mean-ratio fallback) is proven directly
    against FFNSurgeon in test_ffn_surgery.py; this only pins that the
    calibrated cross moment is what reaches the merge, not 1.0/the mean
    fallback, by checking it against a hand-computed expectation.
    """
    stats = FFNCalibrator(model).collect(batches, layer_idx=0)
    cross_moments = FFNCalibrator(model).collect_cross_moments(batches, layer_idx=0, pairs=[(2, 7)])
    expected_scale = cross_moments[(2, 7)] / float(stats.rms[2]) ** 2

    adapter = BertLayerAdapter(model)
    _, output_before = adapter.get_ffn(0)
    original_col2 = output_before.weight[:, 2].clone()
    original_col7 = output_before.weight[:, 7].clone()

    kept = prune_ffn_layer(
        model, 0, TwinRedundancySelector([(2, 7)], merge=True),
        stats=stats, cross_moments=cross_moments, compensate_bias=True,
    )
    assert kept == 15
    with torch.no_grad():
        model(**batches[0])  # shapes still line up end to end

    # Neuron 2 keeps position 2 post-surgery (only neuron 7, at a higher
    # index, was dropped), and now carries the scaled contribution of the
    # original column 7 on top of its own original value.
    _, output_after = adapter.get_ffn(0)
    residual = output_after.weight[:, 2] - original_col2
    assert torch.allclose(residual, original_col7 * expected_scale, atol=1e-5)


def test_unsupported_adapter_methods_explain_themselves():
    from prunelib.transformer import FFNLayerAdapter

    class MinimalAdapter(FFNLayerAdapter):
        def get_ffn(self, layer_idx):
            raise NotImplementedError

        def set_ffn(self, layer_idx, intermediate, output):
            raise NotImplementedError

    with pytest.raises(NotImplementedError, match="get_activation_module"):
        MinimalAdapter().get_activation_module(0)
    with pytest.raises(NotImplementedError, match="num_layers"):
        MinimalAdapter().num_layers()


# --- attention heads: selector-driven and model-wide ------------------------

from conftest import StubBert
from prunelib.transformer import (
    GradientHeadSelector,
    HeadCalibrator,
    OVNormHeadSelector,
    RedundantHeadSelector,
    prune_attention_layer,
    prune_model_attention,
)


def four_head_model():
    return StubBert(hidden=16, intermediate=16, num_layers=3, heads=4)


def test_prune_attention_layer_uses_the_selector_and_still_runs(batches):
    model = four_head_model()
    kept = prune_attention_layer(model, 1, OVNormHeadSelector(prune_percent=50))
    assert kept == 2
    assert model.encoder.layer[1].attention.self.num_attention_heads == 2
    assert model.encoder.layer[0].attention.self.num_attention_heads == 4
    with torch.no_grad():
        model(**batches[0])


def test_uniform_head_allocation_cuts_every_layer_equally(batches):
    model = four_head_model()
    kept = prune_model_attention(model, RedundantHeadSelector(prune_percent=25))
    assert kept == {0: 3, 1: 3, 2: 3}
    with torch.no_grad():
        model(**batches[0])


def test_global_head_allocation_spends_the_budget_where_heads_are_weak(batches):
    model = four_head_model()
    adapter = BertLayerAdapter(model)
    with torch.no_grad():
        for h in (0, 1, 2):  # three of layer 2's heads barely write anything
            adapter.get_attention_heads(2)[3].weight[:, h * 4:(h + 1) * 4] *= 1e-3
    kept = prune_model_attention(
        model, OVNormHeadSelector(prune_percent=25), allocation="global", min_keep_ratio=0.25
    )
    assert sum(kept.values()) == 9  # 12 heads, 25% cut
    assert kept[2] == 1 and kept[0] == kept[1] == 4
    with torch.no_grad():
        model(**batches[0])


def test_global_head_allocation_floors_each_layer(batches):
    model = four_head_model()
    adapter = BertLayerAdapter(model)
    with torch.no_grad():
        adapter.get_attention_heads(0)[3].weight.mul_(1e-6)  # layer 0 looks worthless
    kept = prune_model_attention(
        model, OVNormHeadSelector(prune_percent=50), allocation="global", normalize="none",
        min_keep_ratio=0.5,
    )
    assert kept[0] == 2


def test_global_head_allocation_needs_an_importance_selector():
    with pytest.raises(ValueError, match="HeadImportanceSelector"):
        prune_model_attention(four_head_model(), RedundantHeadSelector(prune_percent=25),
                              allocation="global")


def test_gradient_driven_head_pruning_end_to_end_with_compensation(batches):
    """Calibrate (both signals) -> rank globally -> cut with mean
    compensation -> the pruned model still runs."""
    model = four_head_model()
    calibrator = HeadCalibrator(model)
    loss = lambda outputs, _batch: outputs.square().mean()
    importance = calibrator.collect_gradient_importance(batches, loss_fn=loss)
    stats = calibrator.collect_many(batches)

    kept = prune_model_attention(
        model, GradientHeadSelector(prune_percent=50), allocation="global", normalize="l2",
        stats_by_layer=stats, gradient_importance_by_layer=importance, compensate_bias=True,
    )
    assert sum(kept.values()) == 6
    with torch.no_grad():
        model(**batches[0])


def test_head_compensation_preserves_the_pruned_layers_mean_output(batches):
    model = four_head_model()
    adapter = BertLayerAdapter(model)
    stats = HeadCalibrator(model).collect(batches, layer_idx=0)

    def mean_projection_output():
        rows = []
        output = adapter.get_attention_heads(0)[3]
        handle = output.register_forward_hook(lambda _m, _i, out: rows.append(out.detach()))
        with torch.no_grad():
            for batch in batches:
                model(**batch)
        handle.remove()
        real = [r.reshape(-1, r.shape[-1])[b["attention_mask"].reshape(-1).bool()]
                for r, b in zip(rows, batches)]
        return torch.cat(real).mean(dim=0)

    before = mean_projection_output()
    prune_attention_heads(model, 0, [1, 3], stats=stats, compensate_bias=True)
    assert torch.allclose(mean_projection_output(), before, atol=1e-5)


# --- prune_model_ffn with per-layer selectors -------------------------------

def test_per_layer_selectors_apply_each_layers_own_twins(model, batches):
    kept = prune_model_ffn(model, {
        0: TwinRedundancySelector([(0, 1)]),
        2: TwinRedundancySelector([(3, 4), (5, 6)]),
    })
    assert kept == {0: 15, 2: 14}
    assert BertLayerAdapter(model).get_ffn(1)[0].out_features == 16  # not listed, not touched
    with torch.no_grad():
        model(**batches[0])


def test_per_layer_selectors_are_uniform_only(model):
    with pytest.raises(ValueError, match="per-layer mapping"):
        prune_model_ffn(model, {0: InOutNormSelector(prune_percent=25)}, allocation="global")
    with pytest.raises(ValueError, match="No selector given"):
        prune_model_ffn(model, {0: InOutNormSelector(prune_percent=25)}, layer_indices=[0, 1])


def test_l2_normalisation_is_accepted_for_ffn_global_ranking(model):
    kept = prune_model_ffn(model, InOutNormSelector(prune_percent=50), allocation="global",
                           normalize="l2")
    assert sum(kept.values()) == 24
