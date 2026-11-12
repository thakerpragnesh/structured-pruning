"""Abstraction over "how to reach into a transformer model", so pruning
orchestration, attention-head analysis, and activation recording never
need to know a specific model family's attribute layout.

Before this, `prune_ffn_layer`, `AttentionHeadAnalyzer`, and
`ActivationRecorder` each hardcoded `model.encoder.layer[i]...` -- fine
while only BERT existed, but it meant those three modules (high-level
policy) depended directly on one concrete model shape (a low-level
detail) instead of on an abstraction (Dependency Inversion). Supporting
a differently-shaped architecture meant editing all three.

Now they depend on adapter interfaces instead. Those interfaces are
split in two -- `FFNLayerAdapter` (get/set the FFN pair) and
`AttentionLayerAdapter` (get/set the attention quartet, head count) --
because no consumer needs both: `prune_ffn_layer` and `ActivationRecorder`
only touch the FFN side, `AttentionHeadAnalyzer` only touches the
attention side. Keeping them separate means a future adapter for a model
that only needs FFN pruning support implements `FFNLayerAdapter` alone,
instead of being forced to also stub out attention methods it has no use
for (Interface Segregation). Both share `LayerStackAdapter`, the one
place "how many layers, and which ones" is answered.

**Which adapter a consumer gets** is decided in exactly one place,
`resolve_adapter`. Every consumer used to default to
`adapter or BertLayerAdapter(model)` -- nine call sites that each named
the one concrete adapter, so supporting a second model family meant
passing `adapter=` at every call or editing every default. Now consumers
ask `resolve_adapter(model, adapter)`, which consults a registry; a new
family is one `register_adapter(...)` call (Open/Closed) and no consumer
names a concrete adapter (Dependency Inversion).
"""
from abc import ABC, abstractmethod

import torch.nn as nn


class LayerStackAdapter(ABC):
    """What every adapter shares: the model is a stack of numbered layers.

    `num_layers()` is optional -- only whole-model entry points need it --
    so its default explains what to override rather than being abstract.
    It used to be defined twice, once on each sub-interface, with two
    different error messages for the same missing method.
    """

    def num_layers(self) -> int:
        """How many transformer layers the model has."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement num_layers(), which whole-model "
            "pruning and calibration need to enumerate layers. Either override it or pass "
            "an explicit `layer_indices=` list."
        )

    def resolve_layers(self, layer_indices=None) -> list:
        """`layer_indices` as a list, or every layer when it is `None`.

        The single home of the "None means all layers" convention that
        `prune_model_ffn`, `prune_model_attention` and both calibrators
        share.
        """
        if layer_indices is None:
            return list(range(self.num_layers()))
        return list(layer_indices)


class FFNLayerAdapter(LayerStackAdapter):
    @abstractmethod
    def get_ffn(self, layer_idx: int):
        """Return (intermediate_linear, output_linear) for the layer's FFN block."""

    @abstractmethod
    def set_ffn(self, layer_idx: int, intermediate: nn.Linear, output: nn.Linear) -> None:
        """Install a resized (intermediate, output) pair back onto the model."""

    def get_activation_module(self, layer_idx: int) -> nn.Module:
        """Return the module whose output is the FFN's *post-activation*
        hidden tensor -- i.e. exactly what the output projection consumes.

        Deliberately not abstract. Only `ffn_calibration.FFNCalibrator` and
        `activation_recording.ActivationRecorder` need it, so an adapter
        written purely to enable weight-based pruning shouldn't be forced
        to implement it (same Interface Segregation reasoning that split
        `FFNLayerAdapter` from `AttentionLayerAdapter` in the first
        place). The default explains what to override rather than
        failing obscurely.

        Note this is *not* `get_ffn(...)[0]`: that returns the
        intermediate `nn.Linear`, whose output is pre-activation. For
        sign-based firing detection the two agree (GELU and ReLU are both
        positive exactly where their input is), but for any magnitude
        statistic they differ, which is why calibration hooks here.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_activation_module(), so it cannot "
            "be used for activation calibration or recording. Override it to return the "
            "module emitting the post-activation FFN hidden state (for BERT: "
            "`model.encoder.layer[i].intermediate`)."
        )


class AttentionLayerAdapter(LayerStackAdapter):
    """Three required members: the head count, and getting / setting the
    `(query, key, value, output)` quartet. Everything a head consumer needs
    derives from those -- which is why `hidden_size` (read by nothing) is no
    longer required, and `get_query_weight` is a default built on
    `get_attention_heads` rather than a fourth thing every adapter must
    write.
    """

    @abstractmethod
    def num_attention_heads(self, layer_idx: int) -> int:
        """How many attention heads layer `layer_idx` currently has.

        Deliberately per-layer rather than a model-wide property:
        `attention_surgery.prune_attention_heads` can remove heads from one
        layer while leaving others untouched, so a single shared number --
        such as `config.num_attention_heads` -- would be wrong for every
        layer but the one just pruned. HuggingFace's own head-count
        bookkeeping lives on the per-layer self-attention module for the
        same reason; see `BertLayerAdapter.set_attention_heads`.
        """

    @abstractmethod
    def get_attention_heads(self, layer_idx: int):
        """Return `(query, key, value, output)` Linear modules for the
        layer's multi-head attention block: `query`/`key`/`value` partition
        their output rows into per-head blocks, `output` is the attention
        block's output projection, whose input columns partition the same
        way.
        """

    @abstractmethod
    def set_attention_heads(self, layer_idx: int, query: nn.Linear, key: nn.Linear,
                             value: nn.Linear, output: nn.Linear, num_heads: int) -> None:
        """Install a resized `(query, key, value, output)` quartet and
        update the per-layer head-count bookkeeping the model's own forward
        pass relies on to reshape Q/K/V into heads.
        """

    def get_query_weight(self, layer_idx: int):
        """The layer's attention query projection weight."""
        return self.get_attention_heads(layer_idx)[0].weight.data


class TransformerLayerAdapter(FFNLayerAdapter, AttentionLayerAdapter):
    """Convenience union for an adapter that supports both sides, as
    `BertLayerAdapter` does. Consumers should still depend on whichever
    single sub-interface they actually need.
    """


class BertLayerAdapter(TransformerLayerAdapter):
    """Adapter for HuggingFace's `BertModel` layout (`encoder.layer[i]`),
    which RoBERTa's model class also uses.
    """

    def __init__(self, model):
        self.model = model

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def num_attention_heads(self, layer_idx: int) -> int:
        return self.model.encoder.layer[layer_idx].attention.self.num_attention_heads

    def get_ffn(self, layer_idx: int):
        layer = self.model.encoder.layer[layer_idx]
        return layer.intermediate.dense, layer.output.dense

    def set_ffn(self, layer_idx: int, intermediate: nn.Linear, output: nn.Linear) -> None:
        layer = self.model.encoder.layer[layer_idx]
        layer.intermediate.dense = intermediate
        layer.output.dense = output

    def get_activation_module(self, layer_idx: int) -> nn.Module:
        # BertIntermediate == dense + intermediate_act_fn, so its output
        # is post-activation; `layer.intermediate.dense` would not be.
        return self.model.encoder.layer[layer_idx].intermediate

    def num_layers(self) -> int:
        return len(self.model.encoder.layer)

    def get_attention_heads(self, layer_idx: int):
        self_attn = self.model.encoder.layer[layer_idx].attention.self
        output = self.model.encoder.layer[layer_idx].attention.output.dense
        return self_attn.query, self_attn.key, self_attn.value, output

    def set_attention_heads(self, layer_idx: int, query: nn.Linear, key: nn.Linear,
                             value: nn.Linear, output: nn.Linear, num_heads: int) -> None:
        layer = self.model.encoder.layer[layer_idx]
        self_attn = layer.attention.self
        self_attn.query, self_attn.key, self_attn.value = query, key, value
        layer.attention.output.dense = output
        # BertSelfAttention.forward reshapes Q/K/V using these three
        # attributes, not config.num_attention_heads -- HuggingFace's own
        # `prune_heads()` updates them the same way and deliberately leaves
        # config alone (see docs/CHANGELOG.md, "Deferred"). Recomputing
        # attention_head_size from the resized weight rather than trusting
        # the old value catches a caller who passed heads of mismatched size.
        self_attn.num_attention_heads = num_heads
        self_attn.attention_head_size = query.weight.shape[0] // num_heads
        self_attn.all_head_size = num_heads * self_attn.attention_head_size


def _has_bert_layout(model) -> bool:
    encoder = getattr(model, "encoder", None)
    return encoder is not None and hasattr(encoder, "layer")


# (matches(model) -> bool, factory(model) -> adapter), consulted in order.
_ADAPTER_REGISTRY = [(_has_bert_layout, BertLayerAdapter)]


def register_adapter(matches, factory):
    """Teach `resolve_adapter` a new model family. Returns a function that
    undoes the registration.

    `matches(model) -> bool` recognises the family's layout; `factory(model)`
    builds its adapter. Registrations are consulted newest first, so a
    caller can also override how an already-supported family is adapted.
    """
    entry = (matches, factory)
    _ADAPTER_REGISTRY.insert(0, entry)

    def unregister():
        if entry in _ADAPTER_REGISTRY:
            _ADAPTER_REGISTRY.remove(entry)

    return unregister


def resolve_adapter(model, adapter=None):
    """The adapter a consumer should use for `model`: `adapter` itself if
    given, otherwise the first registered family that recognises `model` --
    or, failing that, its HuggingFace `base_model`.

    The `base_model` step is what lets a task model be passed directly.
    `BertForSequenceClassification` has no `encoder` of its own, but its
    `base_model` (the `BertModel` inside) does. So the adapter targets the
    encoder's layers while any batches still run through the whole task
    model, loss and all. A bare encoder's `base_model` is itself.
    """
    if adapter is not None:
        return adapter
    candidates = [model]
    base = getattr(model, "base_model", None)
    if base is not None and base is not model:
        candidates.append(base)
    for candidate in candidates:
        for matches, factory in _ADAPTER_REGISTRY:
            if matches(candidate):
                return factory(candidate)
    raise TypeError(
        f"No layer adapter is registered for {type(model).__name__}. Pass `adapter=` "
        "explicitly, or call `adapters.register_adapter(matches, factory)` once to "
        "teach every consumer this model family."
    )
