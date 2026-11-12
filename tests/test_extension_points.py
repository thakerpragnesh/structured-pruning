"""Tests for the open/closed extension points: new saliency criteria and
distance metrics plug in via registration, without editing prunelib."""
import pytest
import torch
import torch.nn as nn

from prunelib import (
    available_methods,
    compute_score,
    keep_indices,
    pairwise_distance_matrix,
    prune_conv_bn,
    prune_count,
    register_metric,
    register_saliency,
)
from prunelib import saliency as saliency_module
from prunelib import scanners as scanners_module


@pytest.fixture
def isolated_registries(monkeypatch):
    monkeypatch.setattr(saliency_module, "_METHODS", dict(saliency_module._METHODS))
    monkeypatch.setattr(scanners_module, "_METRICS", dict(scanners_module._METRICS))


def test_registered_saliency_is_dispatched_by_compute_score(isolated_registries):
    @register_saliency("row_max")
    def row_max(weight):
        return weight.detach().reshape(weight.shape[0], -1).abs().amax(dim=1)

    w = torch.randn(6, 3, 3, 3)
    assert "row_max" in available_methods()
    assert torch.equal(compute_score(w, method="row_max"), row_max(w))


def test_register_saliency_refuses_silent_overwrite(isolated_registries):
    with pytest.raises(ValueError, match="already registered"):
        register_saliency("l1", lambda w: w)
    register_saliency("l1", lambda w: w.sum(), overwrite=True)  # explicit opt-in is allowed


def test_max_k_default_matches_paper_setting_without_kwargs():
    w = torch.randn(8, 4, 3, 3)
    assert torch.equal(compute_score(w, method="max_k"), compute_score(w, method="max_k", k=3))


def test_registered_metric_is_dispatched(isolated_registries):
    register_metric("chebyshev", lambda v: torch.cdist(v, v, p=float("inf")))
    v = torch.randn(5, 7)
    assert torch.allclose(pairwise_distance_matrix(v, metric="chebyshev"), torch.cdist(v, v, p=float("inf")))


def test_keep_indices_is_sorted_complement():
    scores = torch.tensor([5.0, 1.0, 4.0, 0.5, 3.0])
    assert keep_indices(scores, 2).tolist() == [0, 2, 4]
    assert keep_indices(scores, 0).tolist() == [0, 1, 2, 3, 4]
    assert keep_indices(scores, 99).tolist() == []


@pytest.mark.parametrize("n,fraction,expected", [(64, 0.05, 3), (10, 0.01, 1), (32, 0.5, 16)])
def test_prune_count_rounds_and_is_at_least_one(n, fraction, expected):
    assert prune_count(n, fraction) == expected


def test_prune_conv_bn_rejects_grouped_conv():
    conv = nn.Conv2d(8, 8, 3, groups=4)
    with pytest.raises(ValueError, match="grouped"):
        prune_conv_bn(conv, torch.arange(4))


def test_surgery_preserves_dtype():
    conv = nn.Conv2d(3, 8, 3).double()
    bn = nn.BatchNorm2d(8).double()
    nxt = nn.Conv2d(8, 4, 3).double()
    new_conv, new_bn, new_next = prune_conv_bn(conv, torch.arange(4), bn=bn, next_conv=nxt)
    assert new_conv.weight.dtype == new_bn.weight.dtype == new_next.weight.dtype == torch.float64
