import pytest
import torch
import torch.nn as nn

from prunelib.transformer import (
    LayerKind,
    Max3SaliencyScorer,
    NetworkSaliencyScanner,
    discover_layers,
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.intermediate = nn.Linear(4, 6)
        self.head = nn.Linear(6, 2)
        self.norm = nn.LayerNorm(6)


def test_discovers_conv_only_by_default():
    handles = discover_layers(TinyNet())
    assert [h.name for h in handles] == ["conv"]
    assert handles[0].kind == "conv2d"


def test_linear_name_filter_selects_ffn_layers():
    handles = discover_layers(TinyNet(), include_conv=False, include_linear=True,
                              linear_name_filter="intermediate")
    assert [h.name for h in handles] == ["intermediate"]


def test_extra_kinds_extend_discovery_without_editing_layers_py():
    handles = discover_layers(TinyNet(), include_conv=False,
                              extra_kinds=[LayerKind("layernorm", nn.LayerNorm)])
    assert [h.kind for h in handles] == ["layernorm"]


def test_scan_reports_one_row_per_layer():
    layers = discover_layers(TinyNet(), include_conv=True, include_linear=True)
    report = NetworkSaliencyScanner(layers, Max3SaliencyScorer()).scan()
    assert list(report["Layer"]) == ["conv", "intermediate", "head"]
    assert list(report.columns) == ["Layer", "Type", "Units", "Avg Score", "Min Score", "StdDev"]
    assert list(report["Units"]) == [4, 6, 2]


def test_weakest_units_returns_the_lowest_scorers():
    net = TinyNet()
    with torch.no_grad():
        net.intermediate.weight.fill_(1.0)
        net.intermediate.weight[3] = 1e-6
    layers = discover_layers(net, include_conv=False, include_linear=True,
                             linear_name_filter="intermediate")
    scanner = NetworkSaliencyScanner(layers, Max3SaliencyScorer())
    assert scanner.weakest_units("intermediate", n=1)[0][0] == 3


def test_scores_are_cached_and_invalidatable():
    net = TinyNet()
    layers = discover_layers(net, include_conv=False, include_linear=True,
                             linear_name_filter="intermediate")
    scanner = NetworkSaliencyScanner(layers, Max3SaliencyScorer())
    first = scanner.scan()["Avg Score"][0]
    with torch.no_grad():
        net.intermediate.weight *= 100.0
    assert scanner.scan()["Avg Score"][0] == first  # served from cache
    scanner.invalidate()
    assert scanner.scan()["Avg Score"][0] > first


def test_uncached_scanner_always_rescores():
    net = TinyNet()
    layers = discover_layers(net, include_conv=False, include_linear=True,
                             linear_name_filter="intermediate")
    scanner = NetworkSaliencyScanner(layers, Max3SaliencyScorer(), cache=False)
    first = scanner.scan()["Avg Score"][0]
    with torch.no_grad():
        net.intermediate.weight *= 100.0
    assert scanner.scan()["Avg Score"][0] > first


def test_unknown_layer_name_is_rejected():
    layers = discover_layers(TinyNet())
    scanner = NetworkSaliencyScanner(layers, Max3SaliencyScorer())
    with pytest.raises(ValueError, match="not found"):
        scanner.weakest_units("nope")
