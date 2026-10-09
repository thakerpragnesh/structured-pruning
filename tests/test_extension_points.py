"""
Open/closed: every pluggable part of `prunelib` -- saliency scorers,
selection rules, distance metrics, quantization methods, prunable layer
types, the ops a prune passes through -- is extended from the caller's own code, and the extension then
works through every entry point that dispatches on it, with no edit to the
library. Before `registry.py`, each of these was a private dict or an
if-chain inside the library (KT.md section 7 said "register it in
`_METHODS`"), so these tests had nothing public to exercise.
"""
import inspect

import pytest
import torch
import torch.nn as nn

from prunelib import (
    MODULE_RULES,
    OP_PROPAGATORS,
    ChannelRole,
    DependencyGraph,
    DistanceMetric,
    ModuleRule,
    complement_indices,
    compute_score,
    kmeans,
    l1_saliency,
    pairwise_distance_matrix,
    propagate_add,
    prune_model,
    quantize_model_,
    register_op_propagator,
    register_saliency_method,
    select_prune_indices_among,
    select_prune_indices_by_method,
)
from prunelib.distance import DISTANCE_METRICS
from prunelib.module_rules import Conv2dRule
from prunelib.quantization import QUANTIZATION_METHODS
from prunelib.registry import Registry
from prunelib.saliency import SALIENCY_METHODS
from prunelib.selection import SELECTION_METHODS, available_methods


@pytest.fixture
def temporarily_register():
    """Register into a global registry for one test only."""
    added = []

    def register(registry, name, entry):
        registry.register(name, entry)
        added.append((registry, name))
        return entry

    yield register
    for registry, name in reversed(added):
        registry.unregister(name)


def test_registry_basics():
    reg = Registry("widget")

    @reg.register("a")
    def a():
        return "a"

    assert reg.get("a") is a and "a" in reg and reg.names() == ["a"]
    assert reg.resolve("a") is a
    assert reg.resolve(len) is len  # a non-name is taken to already be an implementation
    with pytest.raises(ValueError, match="already registered"):
        reg.register("a", len)
    reg.register("a", len, overwrite=True)
    assert reg.get("a") is len
    reg.unregister("a")
    with pytest.raises(ValueError, match=r"unknown widget 'a', expected one of \[\]"):
        reg.get("a")


def test_built_in_names_cannot_be_silently_shadowed():
    with pytest.raises(ValueError, match="module rule Conv2d is already registered"):
        MODULE_RULES.register(nn.Conv2d, MODULE_RULES.get(nn.Conv2d))
    with pytest.raises(ValueError, match="already registered"):
        register_saliency_method("l1", l1_saliency)
    with pytest.raises(ValueError, match="already registered"):
        DISTANCE_METRICS.register("manhattan", DistanceMetric(pairwise=torch.cdist))


def test_registered_saliency_method_reaches_every_entry_point(temporarily_register):
    """A scorer registered once is accepted by `compute_score`, by the
    selection dispatcher, and by `prune_model`, none of which name it."""
    temporarily_register(SALIENCY_METHODS, "neg_l1", lambda w: -l1_saliency(w))  # prunes the *strongest* channels
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 3, padding=1))
    x = torch.randn(1, 3, 6, 6)
    weight = model[0].weight.detach().clone()
    strongest = torch.topk(l1_saliency(weight), 3).indices.sort().values

    assert torch.equal(compute_score(weight, method="neg_l1"), -l1_saliency(weight))
    assert torch.equal(select_prune_indices_by_method(weight, 3, method="neg_l1"), strongest)

    prune_model(model, x, model[0], prune_fraction=3 / 8, method="neg_l1")
    assert torch.equal(model[0].weight, weight[complement_indices(8, strongest)])


def test_selector_callable_can_be_passed_as_method():
    """Dependency inversion: `prune_model` depends on "a selector", not on
    the built-in rules -- a callable with the selector signature works
    directly, without registering anything."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
    weight = model[0].weight.detach().clone()

    def prune_first_n(w, n):
        return torch.arange(n)

    prune_model(model, torch.randn(2, 4), model[0], prune_fraction=0.5, method=prune_first_n)
    assert torch.equal(model[0].weight, weight[3:])
    assert model[2].in_features == 3


def test_unknown_selection_method_lists_every_available_rule():
    with pytest.raises(ValueError, match="unknown method 'nope'") as excinfo:
        select_prune_indices_by_method(torch.randn(4, 3), 1, method="nope")
    for name in ("kmeans", "max_k", "l1", "l2", "random"):
        assert repr(name) in str(excinfo.value)


def test_registered_distance_metric_works_for_scanning_and_kmeans(temporarily_register):
    """One `DistanceMetric` registration serves both consumers that used to
    carry their own if-chain of metrics (`scanners`, `clustering`)."""
    chebyshev = DistanceMetric(
        pairwise=lambda x, y: torch.cdist(x, y, p=float("inf")),
        centroid=lambda m: (m.max(dim=0).values + m.min(dim=0).values) / 2,  # midrange minimizes L-inf
    )
    temporarily_register(DISTANCE_METRICS, "chebyshev", chebyshev)
    torch.manual_seed(0)
    vectors = torch.randn(6, 3)

    brute = torch.tensor([[(a - b).abs().max().item() for b in vectors] for a in vectors])
    assert torch.allclose(pairwise_distance_matrix(vectors, metric="chebyshev"), brute)

    two_blobs = torch.cat([torch.randn(5, 2) * 0.1, torch.randn(5, 2) * 0.1 + 10])
    labels, _ = kmeans(two_blobs, n_clusters=2, metric="chebyshev")
    assert len(set(labels[:5].tolist())) == 1 and len(set(labels[5:].tolist())) == 1
    assert labels[0] != labels[5]

    # An unregistered DistanceMetric instance works too.
    assert torch.allclose(pairwise_distance_matrix(vectors, metric=chebyshev), brute)


def test_kmeans_rejects_a_metric_with_no_centroid_rule(temporarily_register):
    temporarily_register(DISTANCE_METRICS, "scan_only", DistanceMetric(pairwise=lambda x, y: torch.cdist(x, y)))
    vectors = torch.randn(5, 2)
    assert pairwise_distance_matrix(vectors, metric="scan_only").shape == (5, 5)  # fine for scanning
    with pytest.raises(ValueError, match="no centroid rule"):
        kmeans(vectors, n_clusters=2, metric="scan_only")


def test_registered_selection_rule_is_preferred_over_a_scorer_of_the_same_name(temporarily_register):
    temporarily_register(SALIENCY_METHODS, "dual", lambda w: torch.arange(w.shape[0], dtype=torch.float))
    temporarily_register(SELECTION_METHODS, "dual", lambda w, n: torch.arange(w.shape[0] - n, w.shape[0]))
    assert torch.equal(select_prune_indices_by_method(torch.randn(6, 2), 2, method="dual"), torch.tensor([4, 5]))


def test_registered_quantization_method_applies_through_quantize_model_(temporarily_register):
    temporarily_register(QUANTIZATION_METHODS, "tenths", lambda t: torch.round(t * 10) / 10)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(2, 3, 3), nn.Flatten(), nn.Linear(3, 2))
    conv_before = model[0].weight.detach().clone()

    quantize_model_(model, method="tenths", module_types=(nn.Linear,))

    assert torch.allclose(model[2].weight * 10, torch.round(model[2].weight * 10))
    assert torch.equal(model[0].weight, conv_before)  # Conv2d excluded by module_types


class _Conv1dRule(ModuleRule):
    """What a caller writes to make DependencyGraph prune a layer type
    `prunelib` has no built-in rule for -- here, ungrouped Conv1d."""

    def role(self, conv):
        return ChannelRole.MIXING

    def rebuild(self, conv, prune_out, prune_in):
        keep_out = complement_indices(conv.out_channels, prune_out)
        keep_in = complement_indices(conv.in_channels, prune_in)
        new = nn.Conv1d(keep_in.numel(), keep_out.numel(), conv.kernel_size, padding=conv.padding)
        with torch.no_grad():
            new.weight.copy_(conv.weight[keep_out][:, keep_in])
            new.bias.copy_(conv.bias[keep_out])
        return new


def _conv1d_net():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv1d(3, 8, 3, padding=1), nn.BatchNorm1d(8), nn.ReLU(), nn.Conv1d(8, 4, 3, padding=1),
    ).eval()


def test_module_rule_teaches_dependency_graph_a_new_layer_type():
    """The graph walk, the BatchNorm pass-through, and the rebuild all work
    for Conv1d once a rule is supplied -- without touching graph.py."""
    x = torch.randn(2, 3, 10)
    keep = torch.tensor([0, 2, 3, 6, 7])

    with pytest.raises(TypeError, match="Conv1d"):
        DependencyGraph(_conv1d_net(), x).get_pruning_group("0", keep)  # no rule -> refused, not guessed

    model = _conv1d_net()
    weight = model[0].weight.detach().clone()
    dep = DependencyGraph(model, x, module_rules={nn.Conv1d: _Conv1dRule()})
    group = dep.get_pruning_group("0", keep)
    assert set(group.output_targets) == {"0", "1"} and set(group.input_targets) == {"3"}
    group.prune()

    assert model[0].out_channels == 5 and model[1].num_features == 5 and model[3].in_channels == 5
    assert torch.equal(model[0].weight, weight[keep])
    assert model(x).shape == (2, 4, 10)
    assert nn.Conv1d not in MODULE_RULES  # scoped to that one graph, not leaked globally


def test_per_graph_overrides_are_layered_over_the_live_global_registry(temporarily_register):
    """A graph built with *any* `module_rules=` override used to snapshot the
    global table, so a rule registered globally afterwards reached graphs
    built without overrides but not this one. Both now see it."""
    x = torch.randn(2, 3, 10)
    dep = DependencyGraph(_conv1d_net(), x, module_rules={nn.Conv2d: MODULE_RULES.get(nn.Conv2d)})
    temporarily_register(MODULE_RULES, nn.Conv1d, _Conv1dRule())

    group = dep.get_pruning_group("0", torch.tensor([0, 2, 3, 6, 7]))
    assert set(group.output_targets) == {"0", "1"} and set(group.input_targets) == {"3"}


def test_module_rule_methods_are_positional_only():
    """Liskov: the built-in rules name their parameters for their own type
    (`role(self, conv)`), so `rule.role(module=m)` -- valid against the
    base's `role(self, module)` -- raised TypeError on every one of them.
    The base methods are positional-only now, so the contract is a
    positional call, which every rule (built-in or `_Conv1dRule`) accepts."""
    for name in ("role", "rebuild", "mask"):
        params = list(inspect.signature(getattr(ModuleRule, name)).parameters.values())[1:]
        assert all(p.kind is inspect.Parameter.POSITIONAL_ONLY for p in params), name
        for rule in [*MODULE_RULES.entries().values(), _Conv1dRule()]:
            inspect.signature(getattr(rule, name)).bind(*[None] * len(params))  # raises if the call doesn't fit


class _RecordingConv2dRule(Conv2dRule):
    """The built-in Conv2d rule, plus a record of every `mask` call."""

    def __init__(self):
        self.masked = []

    def mask(self, conv, prune_out):
        self.masked.append(prune_out.tolist())
        super().mask(conv, prune_out)


def test_pruning_group_masks_through_the_module_rule():
    """Dependency inversion: `PruningGroup.mask()` called
    `masking.mask_channels` on every layer itself, so a rule could say how to
    rebuild its type but not how to mask it -- a layer whose output channels
    aren't dim 0 of its weight would have been zeroed on the wrong axis. It
    now asks the rule."""
    rule = _RecordingConv2dRule()
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 6, 3, padding=1), nn.ReLU(), nn.Conv2d(6, 2, 1))
    dep = DependencyGraph(model, torch.randn(1, 3, 6, 6), module_rules={nn.Conv2d: rule})
    dep.get_pruning_group("0", torch.tensor([0, 2, 5])).mask()

    assert rule.masked == [[1, 3, 4]]  # the output target only; "2" shrinks on input, at compress time
    assert (model[0].weight[[1, 3, 4]] == 0).all() and (model[0].weight[[0, 2, 5]] != 0).any()


def test_select_prune_indices_among_never_picks_outside_the_candidates():
    """Iterative masking selects among the not-yet-masked channels only, or a
    masked channel (zero weight, the lowest score under every rule) wins
    again every iteration. That step lived inside `vgg.mask_vgg_layer`; it
    is a selection function now, usable by any masking loop."""
    torch.manual_seed(0)
    w = torch.randn(6, 3, 3, 3)
    w[[1, 4]] = 0  # already masked
    candidates = torch.tensor([0, 2, 3, 5])
    for method in available_methods():
        picked = select_prune_indices_among(w, candidates, 2, method=method)
        assert picked.numel() == 2 and set(picked.tolist()) <= set(candidates.tolist()), method
        assert torch.equal(picked, picked.sort().values)


class _SubResidual(nn.Module):
    """A residual block merged with subtraction, which DependencyGraph has no
    built-in propagator for. `method=True` writes it as `.sub()` (an fx
    `call_method` node) instead of `torch.sub` (a `call_function` node)."""

    def __init__(self, method: bool = False):
        super().__init__()
        self.method = method
        self.a = nn.Conv2d(3, 6, 3, padding=1)
        self.b = nn.Conv2d(6, 6, 3, padding=1)
        self.head = nn.Conv2d(6, 2, 1)

    def forward(self, x):
        y = self.a(x)
        z = self.b(y).sub(y) if self.method else torch.sub(self.b(y), y)
        return self.head(z)


def test_op_propagator_teaches_dependency_graph_a_new_op(temporarily_register):
    """`torch.sub` couples its two branches exactly like `+` does, so the
    built-in `propagate_add` handles it once registered -- per graph, or
    globally -- without touching graph.py."""
    x = torch.randn(1, 3, 8, 8)
    keep = torch.tensor([0, 1, 3, 5])

    torch.manual_seed(0)
    with pytest.raises(NotImplementedError, match="sub"):
        DependencyGraph(_SubResidual(), x).get_pruning_group("b", keep)  # unknown op -> refused, not guessed

    torch.manual_seed(0)
    model = _SubResidual()
    a_weight, b_weight = model.a.weight.detach().clone(), model.b.weight.detach().clone()
    group = DependencyGraph(model, x, op_propagators={torch.sub: propagate_add}).get_pruning_group("b", keep)
    assert set(group.output_targets) == {"a", "b"} and set(group.input_targets) == {"b", "head"}
    group.prune()
    assert torch.equal(model.a.weight, a_weight[keep]) and torch.equal(model.b.weight, b_weight[keep][:, keep])
    assert model(x).shape == (1, 2, 8, 8)
    assert torch.sub not in OP_PROPAGATORS  # scoped to that one graph, not leaked globally

    temporarily_register(OP_PROPAGATORS, "sub", propagate_add)
    model = _SubResidual(method=True)
    DependencyGraph(model, x).get_pruning_group("b", keep).prune()
    assert model(x).shape == (1, 2, 8, 8)
    with pytest.raises(ValueError, match="op propagator 'sub' is already registered"):
        register_op_propagator("sub", propagate_add)


def test_op_propagator_sees_only_the_walk_interface():
    """Interface segregation: a propagator was handed the `PruningGroup`
    being filled and, through it, the whole `DependencyGraph`, so it could
    call `group.prune()` halfway through a walk. It now sees five methods."""
    seen = []

    def checking_sub(walk, source, node, idx):
        seen.append({n for n in dir(walk) if not n.startswith("_")})
        propagate_add(walk, source, node, idx)

    torch.manual_seed(0)
    model = _SubResidual()
    x = torch.randn(1, 3, 8, 8)
    DependencyGraph(model, x, op_propagators={torch.sub: checking_sub}).get_pruning_group("b", torch.tensor([0, 1, 3, 5])).prune()

    assert seen and all(s == {"shape", "add_output_target", "add_input_target", "forward", "find_producer"} for s in seen)
    assert model(x).shape == (1, 2, 8, 8)


def test_every_method_argument_accepts_an_implementation(temporarily_register):
    """`Registry.resolve`'s contract: wherever `prunelib` takes `method=` by
    name, it takes the implementation itself too -- scoring and
    quantization included, not just selection."""
    w = torch.randn(5, 3, 3, 3)
    assert torch.equal(compute_score(w, method=l1_saliency), compute_score(w, method="l1"))

    def tenths(t):
        return torch.round(t * 10) / 10

    torch.manual_seed(0)
    by_name, by_fn = nn.Linear(4, 3), nn.Linear(4, 3)
    by_fn.load_state_dict(by_name.state_dict())
    temporarily_register(QUANTIZATION_METHODS, "tenths", tenths)
    quantize_model_(by_name, method="tenths")
    quantize_model_(by_fn, method=tenths)
    assert torch.equal(by_name.weight, by_fn.weight) and torch.equal(by_name.bias, by_fn.bias)
