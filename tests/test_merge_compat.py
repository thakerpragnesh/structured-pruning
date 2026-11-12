"""Guarantees `main` had before the generic-pruning-quantization merge, kept
true on the unified API: the names `main` published (`register_saliency`,
`register_metric`) still work, CSD is one scorer on both sides, and the
surgery guarantees `main` tested (grouped convs refused, dtype kept) hold
for the branch's `slice_*` implementation too."""
import pytest
import torch
import torch.nn as nn

from prunelib import (
    available_methods,
    compute_score,
    csd_saliency,
    keep_indices,
    pairwise_distance_matrix,
    prune_conv_bn,
    prune_count,
    register_metric,
    register_saliency,
    select_prune_indices_by_method,
)
from prunelib.distance import DISTANCE_METRICS
from prunelib.saliency import SALIENCY_METHODS


@pytest.fixture
def cleanup():
    added = []
    yield added
    for registry, name in added:
        registry.unregister(name)


def test_register_saliency_alias_reaches_compute_score_and_selection(cleanup):
    @register_saliency("row_max")
    def row_max(weight):
        return weight.detach().reshape(weight.shape[0], -1).abs().amax(dim=1)

    cleanup.append((SALIENCY_METHODS, "row_max"))
    w = torch.randn(6, 3, 3, 3)
    assert "row_max" in available_methods()
    assert torch.equal(compute_score(w, method="row_max"), row_max(w))
    assert select_prune_indices_by_method(w, 2, method="row_max").numel() == 2


def test_register_saliency_alias_refuses_silent_overwrite():
    with pytest.raises(ValueError, match="already registered"):
        register_saliency("l1", lambda w: w)


def test_register_metric_alias_for_self_distance_functions(cleanup):
    register_metric("chebyshev_compat", lambda v: torch.cdist(v, v, p=float("inf")))
    cleanup.append((DISTANCE_METRICS, "chebyshev_compat"))
    v = torch.randn(5, 7)
    expected = torch.cdist(v, v, p=float("inf"))
    assert torch.allclose(pairwise_distance_matrix(v, metric="chebyshev_compat"), expected)
    # cross-distances (what kmeans assignment would ask for) come from the joint matrix
    w = torch.randn(3, 7)
    cross = DISTANCE_METRICS.get("chebyshev_compat").pairwise(v, w)
    assert torch.allclose(cross, torch.cdist(v, w, p=float("inf")))


def test_csd_is_registered_and_shared_with_transformer_side():
    from prunelib.transformer import CSDScorer

    w = torch.randn(12, 7)
    assert torch.equal(compute_score(w, method="csd"), csd_saliency(w))
    assert torch.equal(CSDScorer().score(w), csd_saliency(w))
    conv = nn.Conv2d(3, 8, 3)
    assert compute_score(conv.weight, method="csd").shape == (8,)


def test_max_k_default_matches_paper_setting_without_kwargs():
    w = torch.randn(8, 4, 3, 3)
    assert torch.equal(compute_score(w, method="max_k"), compute_score(w, method="max_k", k=3))


def test_keep_indices_is_sorted_complement():
    scores = torch.tensor([5.0, 1.0, 4.0, 0.5, 3.0])
    assert keep_indices(scores, 2).tolist() == [0, 2, 4]
    assert keep_indices(scores, 0).tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("n,fraction,expected", [(64, 0.05, 3), (10, 0.01, 1), (32, 0.5, 16)])
def test_prune_count_rounds_and_is_at_least_one(n, fraction, expected):
    assert prune_count(n, fraction) == expected


def test_prune_conv_bn_rejects_non_depthwise_grouped_conv():
    conv = nn.Conv2d(8, 8, 3, groups=4)
    with pytest.raises((ValueError, NotImplementedError)):
        prune_conv_bn(conv, torch.arange(4))


def test_surgery_preserves_dtype():
    conv = nn.Conv2d(3, 8, 3).double()
    bn = nn.BatchNorm2d(8).double()
    nxt = nn.Conv2d(8, 4, 3).double()
    new_conv, new_bn, new_next = prune_conv_bn(conv, torch.arange(4), bn=bn, next_conv=nxt)
    assert new_conv.weight.dtype == new_bn.weight.dtype == new_next.weight.dtype == torch.float64


def test_transformer_subpackage_is_not_loaded_by_core_import():
    import subprocess
    import sys

    code = "import sys, prunelib; print('prunelib.transformer' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"
