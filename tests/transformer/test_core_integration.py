"""The seams between core `prunelib` and `prunelib.transformer`: shared
scores and metrics must agree, and core criteria must be usable here."""
import pytest
import torch

import prunelib
from prunelib.transformer import (
    AttentionHeadAnalyzer,
    CSDScorer,
    NetworkSaliencyScanner,
    RegisteredScorer,
    SaliencySelector,
    discover_layers,
    prune_ffn_layer,
)


def test_csd_is_one_implementation_on_both_sides():
    w = torch.randn(12, 7)
    assert torch.equal(CSDScorer().score(w), prunelib.compute_score(w, method="csd"))


def test_csd_drives_cnn_masking():
    conv = torch.nn.Conv2d(3, 8, 3)
    scores = prunelib.compute_score(conv.weight, method="csd")
    assert scores.shape == (8,)


def test_registered_scorer_bridges_core_methods(model):
    w = model.encoder.layer[0].intermediate.dense.weight
    assert torch.equal(RegisteredScorer("l2").score(w), prunelib.l2_saliency(w))

    kept = prune_ffn_layer(model, 0, SaliencySelector(RegisteredScorer("l1"), prune_percent=50))
    assert kept == 8 and model.encoder.layer[0].output.dense.in_features == 8


def test_registered_scorer_sees_user_registered_methods(model):
    prunelib.register_saliency_method("row_max", lambda w: w.detach().abs().amax(dim=1))
    try:
        layers = discover_layers(model, include_conv=False, include_linear=True, linear_name_filter="intermediate")
        weakest = NetworkSaliencyScanner(layers, RegisteredScorer("row_max")).weakest_units(layers[0].name, n=2)
        assert len(weakest) == 2
    finally:
        prunelib.SALIENCY_METHODS.unregister("row_max")


def test_head_distance_accepts_core_metrics(model):
    analyzer = AttentionHeadAnalyzer(model)
    manhattan = analyzer.compute_distance_matrix(0, metric="manhattan")
    cosine = analyzer.compute_distance_matrix(0, metric="cosine")
    heads = analyzer.get_flat_heads(0)
    assert manhattan[0, 1] == pytest.approx(float((heads[0] - heads[1]).abs().sum()), rel=1e-5)
    assert cosine.shape == manhattan.shape
