import numpy as np
import pytest
import torch

from prunelib.transformer import AttentionHeadAnalyzer, BertLayerAdapter


def test_flat_heads_partition_the_query_rows(model):
    heads = AttentionHeadAnalyzer(model).get_flat_heads(0)
    hidden, num_heads = model.config.hidden_size, model.config.num_attention_heads
    assert heads.shape == (num_heads, (hidden // num_heads) * hidden)
    w_q = BertLayerAdapter(model).get_query_weight(0)
    assert torch.allclose(heads[0], w_q[: hidden // num_heads].reshape(-1))


def test_similarity_is_symmetric_with_unit_diagonal(model):
    sim = AttentionHeadAnalyzer(model).compute_similarity(0)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-5)
    assert np.allclose(sim, sim.T, atol=1e-6)


def test_identical_heads_have_similarity_one(model):
    w_q = BertLayerAdapter(model).get_query_weight(0)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    with torch.no_grad():
        w_q[head_dim:head_dim * 2] = w_q[:head_dim]
    sim = AttentionHeadAnalyzer(model).compute_similarity(0)
    assert sim[0, 1] == pytest.approx(1.0, abs=1e-5)


def test_distance_matrix_has_a_zero_diagonal(model):
    dist = AttentionHeadAnalyzer(model).compute_distance_matrix(0, metric="euclidean")
    assert np.allclose(np.diag(dist), 0.0)


def test_manhattan_and_euclidean_differ(model):
    analyzer = AttentionHeadAnalyzer(model)
    l1 = analyzer.compute_distance_matrix(0, metric="manhattan")
    l2 = analyzer.compute_distance_matrix(0, metric="euclidean")
    assert not np.allclose(l1, l2)
    assert (l1 >= l2 - 1e-5).all()  # L1 dominates L2


def test_normalising_removes_magnitude(model):
    analyzer = AttentionHeadAnalyzer(model)
    raw = analyzer.compute_distance_matrix(0, metric="euclidean", normalize=False)
    w_q = BertLayerAdapter(model).get_query_weight(0)
    with torch.no_grad():
        w_q *= 10.0
    scaled_raw = analyzer.compute_distance_matrix(0, metric="euclidean", normalize=False)
    scaled_norm = analyzer.compute_distance_matrix(0, metric="euclidean", normalize=True)
    assert not np.allclose(raw, scaled_raw)
    # Direction is unchanged by a uniform rescale, so normalised distance is.
    assert np.allclose(
        scaled_norm,
        AttentionHeadAnalyzer(model).compute_distance_matrix(0, metric="euclidean", normalize=True),
    )


def test_unknown_metric_is_rejected(model):
    with pytest.raises(ValueError, match="(?i)unknown metric"):
        AttentionHeadAnalyzer(model).compute_distance_matrix(0, metric="not_a_metric")


def test_indivisible_head_layout_is_rejected(model):
    model.encoder.layer[0].attention.self.num_attention_heads = 3  # 8 is not divisible by 3
    with pytest.raises(ValueError, match="not divisible"):
        AttentionHeadAnalyzer(model).get_flat_heads(0)


def test_requires_a_model_or_an_adapter():
    with pytest.raises(ValueError, match="requires either"):
        AttentionHeadAnalyzer()


def test_ov_similarity_and_norms_from_the_analyzer(model):
    analyzer = AttentionHeadAnalyzer(model)
    sim = analyzer.compute_ov_similarity(0)
    norms = analyzer.compute_ov_norms(0)
    heads = model.config.num_attention_heads
    assert sim.shape == (heads, heads) and norms.shape == (heads,)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-5)
    assert (norms > 0).all()
