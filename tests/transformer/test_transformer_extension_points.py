"""The package's extension points, exercised the way an extender would.

Each test here is a SOLID claim made concrete: a new model family, a new
allocation policy, a new score normalisation or head-similarity signal
plugs in *without editing package code*, and a minimal implementation of
an interface is enough for every consumer of that interface.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from conftest import StubBert
from prunelib.transformer import (
    ActivationAwareSelector,
    AllocationStrategy,
    AttentionHeadAnalyzer,
    AttentionLayerAdapter,
    BertLayerAdapter,
    FFNCalibrator,
    FFNLayerAdapter,
    GlobalAllocation,
    GradientHeadSelector,
    HEAD_SIMILARITIES,
    HeadCalibrator,
    HeadImportanceSelector,
    ImportanceSelector,
    InOutNormSelector,
    OVNormHeadSelector,
    RedundantHeadSelector,
    SCORE_NORMALIZERS,
    WeightClusterRedundancySelector,
    prune_attention_layer,
    prune_model_attention,
    prune_model_ffn,
    register_adapter,
    resolve_adapter,
)
from prunelib.transformer.surgery.linear_ops import rebuild_linear


# --- adapter resolution (Dependency Inversion / Open-Closed) -------------------

class StubClassifier(nn.Module):
    """A HuggingFace-style task model: no `encoder` of its own, the encoder
    reachable as `base_model`, and a loss when `labels` are given."""

    def __init__(self):
        super().__init__()
        self.bert = StubBert(hidden=16, intermediate=16, num_layers=2, heads=4)
        self.head = nn.Linear(16, 2)

    @property
    def base_model(self):
        return self.bert

    @property
    def device(self):
        return self.bert.device

    def forward(self, input_ids, attention_mask=None, labels=None, **_):
        logits = self.head(self.bert(input_ids)[:, 0])
        loss = None if labels is None else nn.functional.cross_entropy(logits, labels)
        return SimpleNamespace(loss=loss, logits=logits)


def labelled_batches():
    return [
        {"input_ids": torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]]),
         "attention_mask": torch.ones(2, 4, dtype=torch.long), "labels": torch.tensor([0, 1])},
        {"input_ids": torch.tensor([[11, 12, 13, 2]]),
         "attention_mask": torch.ones(1, 4, dtype=torch.long), "labels": torch.tensor([1])},
    ]


def test_resolve_adapter_prefers_an_explicit_adapter(model):
    explicit = BertLayerAdapter(model)
    assert resolve_adapter(model, explicit) is explicit


def test_resolve_adapter_recognises_the_bert_layout(model):
    adapter = resolve_adapter(model)
    assert isinstance(adapter, BertLayerAdapter) and adapter.model is model


def test_a_task_model_resolves_to_its_encoder():
    clf = StubClassifier()
    adapter = resolve_adapter(clf)
    assert isinstance(adapter, BertLayerAdapter) and adapter.model is clf.bert


def test_gradient_importance_runs_straight_off_a_task_model():
    """The old gotcha -- `HeadCalibrator(model, adapter=BertLayerAdapter(model.bert))`
    -- is gone: the task model is passed as-is, its loss drives the gates."""
    clf = StubClassifier()
    direct = HeadCalibrator(clf).collect_gradient_importance(labelled_batches())
    explicit = HeadCalibrator(clf, adapter=BertLayerAdapter(clf.bert)).collect_gradient_importance(
        labelled_batches()
    )
    for layer in (0, 1):
        assert torch.allclose(direct[layer], explicit[layer])
        assert (direct[layer] > 0).any()


def test_an_unknown_layout_explains_how_to_teach_it():
    with pytest.raises(TypeError, match="register_adapter"):
        resolve_adapter(nn.Linear(2, 2))


class Block(nn.Module):
    def __init__(self, hidden=8, inner=12):
        super().__init__()
        self.up = nn.Linear(hidden, inner)
        self.act = nn.GELU()
        self.down = nn.Linear(inner, hidden)

    def forward(self, x):
        return x + self.down(self.act(self.up(x)))


class TinyMLPStack(nn.Module):
    """A model family the package has never heard of: `blocks[i].up/down`."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(40, 8)
        self.blocks = nn.ModuleList(Block() for _ in range(2))

    def forward(self, input_ids, attention_mask=None, **_):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return x


class TinyMLPAdapter(FFNLayerAdapter):
    def __init__(self, model):
        self.model = model

    def get_ffn(self, layer_idx):
        block = self.model.blocks[layer_idx]
        return block.up, block.down

    def set_ffn(self, layer_idx, intermediate, output):
        block = self.model.blocks[layer_idx]
        block.up, block.down = intermediate, output

    def get_activation_module(self, layer_idx):
        return self.model.blocks[layer_idx].act

    def num_layers(self):
        return len(self.model.blocks)


def test_registering_a_family_teaches_every_consumer_at_once(batches):
    """One `register_adapter` call, then calibration, whole-model pruning and
    the forward pass all work without anyone passing `adapter=`."""
    unregister = register_adapter(lambda m: isinstance(m, TinyMLPStack), TinyMLPAdapter)
    try:
        model = TinyMLPStack()
        stats = FFNCalibrator(model).collect_many(batches)
        kept = prune_model_ffn(model, ActivationAwareSelector(prune_percent=50),
                               allocation="global", stats_by_layer=stats, compensate_bias=True)
        assert sum(kept.values()) == 12
        with torch.no_grad():
            model(**batches[0])
    finally:
        unregister()
    with pytest.raises(TypeError):
        resolve_adapter(TinyMLPStack())


# --- interface segregation -----------------------------------------------------

class MinimalAttentionAdapter(AttentionLayerAdapter):
    """Only the three required members -- no `hidden_size`, no
    `get_query_weight`, no `num_layers`."""

    def __init__(self, model):
        self.inner = BertLayerAdapter(model)

    def num_attention_heads(self, layer_idx):
        return self.inner.num_attention_heads(layer_idx)

    def get_attention_heads(self, layer_idx):
        return self.inner.get_attention_heads(layer_idx)

    def set_attention_heads(self, layer_idx, query, key, value, output, num_heads):
        self.inner.set_attention_heads(layer_idx, query, key, value, output, num_heads)


def test_a_minimal_attention_adapter_serves_every_head_consumer(batches):
    model = StubBert(hidden=16, intermediate=16, num_layers=2, heads=4)
    adapter = MinimalAttentionAdapter(model)
    sim = AttentionHeadAnalyzer(adapter=adapter).compute_similarity(0)  # via default get_query_weight
    assert sim.shape == (4, 4)
    assert prune_attention_layer(model, 0, OVNormHeadSelector(50), adapter=adapter) == 2
    with torch.no_grad():
        model(**batches[0])
    with pytest.raises(NotImplementedError, match="num_layers"):
        prune_model_attention(model, OVNormHeadSelector(50), adapter=adapter)
    assert prune_model_attention(model, OVNormHeadSelector(50), adapter=adapter,
                                 layer_indices=[1]) == {1: 2}


def test_resolve_layers_is_shared_by_both_adapter_sides(model):
    adapter = BertLayerAdapter(model)
    assert adapter.resolve_layers() == [0, 1, 2]
    assert adapter.resolve_layers((2, 0)) == [2, 0]


# --- allocation strategies (Open/Closed) --------------------------------------

class DeepestLayerOnly(AllocationStrategy):
    """A policy the package doesn't ship: spend the whole cut on the last
    layer, keeping its top-scored half."""

    def apply(self, structure, selector, layer_indices):
        last = max(layer_indices)
        scores = selector.importance(structure.context(last))
        keep = torch.topk(scores, k=scores.numel() // 2).indices.tolist()
        return {last: structure.prune_to(last, keep)}


def test_a_custom_strategy_drives_ffn_and_head_pruning_alike(batches):
    model = StubBert(hidden=16, intermediate=16, num_layers=3, heads=4)
    assert prune_model_ffn(model, InOutNormSelector(10), allocation=DeepestLayerOnly()) == {2: 8}
    assert prune_model_attention(model, OVNormHeadSelector(10), allocation=DeepestLayerOnly()) == {2: 2}
    assert BertLayerAdapter(model).get_ffn(0)[0].out_features == 16
    with torch.no_grad():
        model(**batches[0])


def test_global_allocation_object_equals_the_string_shorthand():
    a, b = StubBert(), StubBert()
    b.load_state_dict(a.state_dict())
    by_string = prune_model_ffn(a, InOutNormSelector(40), allocation="global", normalize="median")
    by_object = prune_model_ffn(b, InOutNormSelector(40), allocation=GlobalAllocation("median"))
    assert by_string == by_object


def test_a_new_score_normalizer_is_one_registry_entry(monkeypatch, model):
    monkeypatch.setitem(SCORE_NORMALIZERS, "max", lambda scores: scores.max())
    kept = prune_model_ffn(model, InOutNormSelector(50), allocation="global", normalize="max")
    assert sum(kept.values()) == 24


def test_unknown_normalizer_lists_the_registered_ones():
    with pytest.raises(ValueError, match="'l2'"):
        GlobalAllocation(normalize="softmax")


def test_a_new_head_similarity_is_one_registry_entry(monkeypatch):
    def value_only(ctx):
        heads = ctx.value_weight.reshape(ctx.num_heads, -1)
        unit = nn.functional.normalize(heads, dim=1)
        return unit @ unit.t(), heads.norm(dim=1)

    monkeypatch.setitem(HEAD_SIMILARITIES, "value", value_only)
    model = StubBert(hidden=16, intermediate=16, num_layers=1, heads=4)
    assert prune_attention_layer(model, 0, RedundantHeadSelector(25, similarity="value")) == 3


# --- shared budget and surgery mechanics ---------------------------------------

@pytest.mark.parametrize("make", [
    lambda: InOutNormSelector(prune_percent=100),
    lambda: WeightClusterRedundancySelector(prune_percent=100),
    lambda: OVNormHeadSelector(prune_percent=100),
    lambda: RedundantHeadSelector(prune_percent=100),
])
def test_every_budgeted_selector_validates_the_same_way(make):
    with pytest.raises(ValueError, match=r"prune_percent must be in \[0, 100\)"):
        make()


def test_budgeted_selectors_share_one_keep_rule():
    assert ImportanceSelector.n_keep is HeadImportanceSelector.n_keep
    assert InOutNormSelector(30).n_keep(10) == GradientHeadSelector(30).n_keep(10) == 7


def test_rebuild_linear_keeps_dtype_and_frozen_flags():
    source = nn.Linear(4, 3, dtype=torch.float16)
    source.weight.requires_grad_(False)
    new = rebuild_linear(source, source.weight.data[:2], source.bias.data[:2])
    assert new.weight.dtype == torch.float16 and new.out_features == 2
    assert new.weight.requires_grad is False and new.bias.requires_grad is True
    assert torch.equal(new.weight, source.weight[:2])


def test_rebuild_linear_never_adds_or_drops_a_bias():
    with pytest.raises(ValueError, match="exactly when"):
        rebuild_linear(nn.Linear(4, 3, bias=False), torch.zeros(3, 4), torch.zeros(3))
    with pytest.raises(ValueError, match="exactly when"):
        rebuild_linear(nn.Linear(4, 3), torch.zeros(3, 4))
