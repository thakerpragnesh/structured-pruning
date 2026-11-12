"""High-level orchestration: apply any `NeuronSelector` to a
transformer's FFN blocks via `FFNSurgeon`, and any `HeadSelector` (or a
plain list of head indices) to its attention blocks via `AttentionSurgeon`.

This is the one place that wires a selection strategy to surgery
mechanics. Everything else in this package depends only on the abstract
`NeuronSelector` / `SaliencyScorer` / `FFNSurgeon` / `FFNLayerAdapter`
contracts, never on each other's concrete classes -- adding a new pruning
criterion, or targeting a new model family, never requires touching this
module.

`prune_ffn_layer` / `prune_attention_layer` handle one layer.
`prune_model_ffn` / `prune_model_attention` handle the whole stack, and
offer the choice that matters at model scale: whether the budget is spent
*uniformly* (every layer loses the same fraction) or *globally* (every
neuron in the model competes against every other, so layers that turn out
to be redundant give up more than layers that are carrying the model).
Uniform is the simpler baseline and the one the thesis's single-layer
experiments generalise to; global usually buys more accuracy at the same
compression, because redundancy -- of FFN neurons and of heads alike -- is
not evenly distributed across depth.

How the budget is spent is not decided here. It is an
`allocation.AllocationStrategy`, which sees FFN neurons and attention heads
through one interface, `allocation.PrunableStructure`. This module's only
job beyond the public functions is to implement that interface twice
(`_FFNStructure`, `_HeadStructure`): what a layer's context is, and how a
selection becomes surgery. That is why one `GlobalAllocation` ranks
neurons and heads by exactly the same rule, and why a new allocation
policy needs no edit here.
"""
from collections.abc import Mapping

from .allocation import PrunableStructure, resolve_allocation
from ..surgery.attention_surgery import AttentionSurgeon
from ..selection.context import FFNContext, HeadContext, Selection
from ..surgery.ffn_surgery import FFNSurgeon
from ..selection.head_selectors import HeadImportanceSelector
from ..models.adapters import resolve_adapter
from ..selection.ffn_selectors import ImportanceSelector


class _FFNStructure(PrunableStructure):
    """FFN neurons as a `PrunableStructure`: `FFNContext` in, `Selection`
    to `FFNSurgeon` out."""

    importance_type = ImportanceSelector
    unit = "neuron"
    rankable_examples = "SaliencySelector, InOutNormSelector, ActivationAwareSelector"

    def __init__(self, model, adapter=None, surgeon=None, stats_by_layer=None,
                 cross_moments_by_layer=None, compensate_bias=False):
        self.adapter = resolve_adapter(model, adapter)
        self.surgeon = surgeon or FFNSurgeon()
        self.stats_by_layer = stats_by_layer or {}
        self.cross_moments_by_layer = cross_moments_by_layer or {}
        self.compensate_bias = compensate_bias

    def context(self, layer_idx):
        intermediate, output = self.adapter.get_ffn(layer_idx)
        return FFNContext.from_layers(
            intermediate, output, stats=self.stats_by_layer.get(layer_idx), layer_index=layer_idx,
            cross_moments=self.cross_moments_by_layer.get(layer_idx),
        )

    def prune_with(self, layer_idx, selector):
        return self._apply(layer_idx, selector.select(self.context(layer_idx)))

    def prune_to(self, layer_idx, keep):
        num_neurons = self.adapter.get_ffn(layer_idx)[0].weight.shape[0]
        return self._apply(layer_idx, Selection.of(keep, num_neurons=num_neurons))

    def _apply(self, layer_idx, selection):
        intermediate, output = self.adapter.get_ffn(layer_idx)
        new_intermediate, new_output = self.surgeon.resize(
            intermediate, output, selection, stats=self.stats_by_layer.get(layer_idx),
            compensate_bias=self.compensate_bias,
        )
        self.adapter.set_ffn(layer_idx, new_intermediate, new_output)
        return len(selection.keep_indices)


class _HeadStructure(PrunableStructure):
    """Attention heads as a `PrunableStructure`: `HeadContext` in, a drop
    list to `AttentionSurgeon` out."""

    importance_type = HeadImportanceSelector
    unit = "head"
    rankable_examples = "OVNormHeadSelector, ActivationAwareHeadSelector, GradientHeadSelector"

    def __init__(self, model, adapter=None, surgeon=None, stats_by_layer=None,
                 gradient_importance_by_layer=None, compensate_bias=False):
        self.adapter = resolve_adapter(model, adapter)
        self.surgeon = surgeon or AttentionSurgeon()
        self.stats_by_layer = stats_by_layer or {}
        self.gradient_importance_by_layer = gradient_importance_by_layer or {}
        self.compensate_bias = compensate_bias

    def context(self, layer_idx):
        query, key, value, output = self.adapter.get_attention_heads(layer_idx)
        return HeadContext.from_layers(
            query, key, value, output, self.adapter.num_attention_heads(layer_idx),
            stats=self.stats_by_layer.get(layer_idx),
            gradient_importance=self.gradient_importance_by_layer.get(layer_idx),
            layer_index=layer_idx,
        )

    def prune_with(self, layer_idx, selector):
        return self.prune_to(layer_idx, selector.select(self.context(layer_idx)))

    def prune_to(self, layer_idx, keep):
        keep = set(keep)
        num_heads = self.adapter.num_attention_heads(layer_idx)
        return self.drop(layer_idx, [h for h in range(num_heads) if h not in keep])

    def drop(self, layer_idx, head_indices):
        query, key, value, output = self.adapter.get_attention_heads(layer_idx)
        new_query, new_key, new_value, new_output, new_num_heads = self.surgeon.resize(
            query, key, value, output, self.adapter.num_attention_heads(layer_idx), head_indices,
            stats=self.stats_by_layer.get(layer_idx), compensate_bias=self.compensate_bias,
        )
        self.adapter.set_attention_heads(
            layer_idx, new_query, new_key, new_value, new_output, new_num_heads
        )
        return new_num_heads


def _prune_model(structure, selector, layer_indices, allocation, normalize, min_keep_ratio):
    """The whole-stack prune both `prune_model_*` functions are."""
    if layer_indices is None and isinstance(selector, Mapping):
        layer_indices = sorted(selector)
    layer_indices = structure.adapter.resolve_layers(layer_indices)
    strategy = resolve_allocation(allocation, normalize, min_keep_ratio)
    return strategy.apply(structure, selector, layer_indices)


def prune_ffn_layer(model, layer_index, selector, surgeon=None, adapter=None, stats=None,
                    compensate_bias=False, cross_moments=None):
    """Prune one layer's FFN. Returns the number of neurons kept.

    `cross_moments` (from `ffn_calibration.FFNCalibrator.collect_cross_moments`)
    is only consulted by a merge-capable selector's least-squares scale
    (`ffn_selectors._merge_scale`); every other selector ignores it.
    """
    structure = _FFNStructure(
        model, adapter, surgeon, stats_by_layer={layer_index: stats},
        cross_moments_by_layer={layer_index: cross_moments}, compensate_bias=compensate_bias,
    )
    return structure.prune_with(layer_index, selector)


def prune_attention_heads(model, layer_index, head_indices, surgeon=None, adapter=None,
                          stats=None, compensate_bias=False):
    """Remove the given attention heads (indices to DROP) from one layer.
    Returns the number of heads kept.

    The low-level entry point: the caller has already decided which heads
    go. To let a criterion decide, use `prune_attention_layer` with a
    `head_selectors.HeadSelector`. `stats` (`head_calibration.HeadStats`)
    is only needed for `compensate_bias=True`.
    """
    structure = _HeadStructure(
        model, adapter, surgeon, stats_by_layer={layer_index: stats},
        compensate_bias=compensate_bias,
    )
    return structure.drop(layer_index, head_indices)


def prune_attention_layer(model, layer_index, selector, surgeon=None, adapter=None, stats=None,
                          gradient_importance=None, compensate_bias=False):
    """Prune one layer's attention heads with a `HeadSelector`. Returns the
    number of heads kept.

    `stats` (`head_calibration.HeadStats`) feeds both
    `ActivationAwareHeadSelector` and bias compensation;
    `gradient_importance` (this layer's `(num_heads,)` entry from
    `HeadCalibrator.collect_gradient_importance`) feeds
    `GradientHeadSelector`. Selectors that need neither ignore them.
    """
    structure = _HeadStructure(
        model, adapter, surgeon, stats_by_layer={layer_index: stats},
        gradient_importance_by_layer={layer_index: gradient_importance},
        compensate_bias=compensate_bias,
    )
    return structure.prune_with(layer_index, selector)


def prune_model_ffn(model, selector, layer_indices=None, allocation="uniform", surgeon=None,
                    adapter=None, stats_by_layer=None, compensate_bias=False,
                    normalize="mean", min_keep_ratio=0.1, cross_moments_by_layer=None):
    """Prune every FFN block in the model. Returns `{layer_index: n_kept}`.

    `allocation` is an `allocation.AllocationStrategy`, or one of the
    shorthands `"uniform"` / `"global"`:

    `"uniform"` (`UniformAllocation`) applies `selector` to each layer
    independently -- each layer loses the selector's own `prune_percent`.
    `selector` may also be a `{layer_index: NeuronSelector}` mapping, for
    criteria whose state belongs to one layer: a `TwinRedundancySelector`
    holds one layer's twin pairs, and handing the same instance to every
    layer would apply layer 0's twins to all of them. With a mapping,
    `layer_indices` defaults to its keys.

    `"global"` (`GlobalAllocation(normalize, min_keep_ratio)`) instead
    treats the selector's `prune_percent` as a *model-wide* budget: it
    scores every neuron in every layer, ranks them all together, and
    removes the globally weakest. This requires an `ImportanceSelector` (a
    criterion that produces a per-neuron score); a purely structural
    criterion like `TwinRedundancySelector` has no scalar to rank across
    layers and is rejected with an explanation.

    `normalize` controls how per-layer scores are made comparable before
    that global ranking. Raw scores are not: different layers have
    different weight scales, so ranking them directly tends to delete one
    or two low-magnitude layers wholesale rather than finding the genuinely
    redundant neurons. `"mean"` (default) divides each layer's scores by
    that layer's mean, `"median"` is its outlier-resistant counterpart,
    `"l2"` divides by the layer's score vector norm, and `"none"` ranks raw
    scores for callers whose criterion is already scale-free
    (`allocation.SCORE_NORMALIZERS`).

    `min_keep_ratio` floors how much of any single layer the global mode
    may take, so an unlucky ranking cannot collapse a layer entirely.

    `cross_moments_by_layer` (`{layer_index: {(i, j): E[a_i * a_j]}}`, from
    `ffn_calibration.FFNCalibrator.collect_cross_moments`) only matters for
    uniform allocation with a merge-capable selector -- global allocation
    requires an `ImportanceSelector`, which never merges.
    """
    structure = _FFNStructure(
        model, adapter, surgeon, stats_by_layer=stats_by_layer,
        cross_moments_by_layer=cross_moments_by_layer, compensate_bias=compensate_bias,
    )
    return _prune_model(structure, selector, layer_indices, allocation, normalize, min_keep_ratio)


def prune_model_attention(model, selector, layer_indices=None, allocation="uniform", surgeon=None,
                          adapter=None, stats_by_layer=None, gradient_importance_by_layer=None,
                          compensate_bias=False, normalize="mean", min_keep_ratio=0.1):
    """Prune the attention heads of every layer. Returns `{layer_index: heads_kept}`.

    The head counterpart of `prune_model_ffn`, with the same allocations,
    strategy objects included. `"uniform"` applies `selector` (or a
    `{layer: HeadSelector}` mapping) to each layer on its own. `"global"`
    treats `prune_percent` as a model-wide head budget and ranks every head
    against every other -- the setting in which Michel et al. (2019) found
    most heads removable -- which needs a `HeadImportanceSelector`.
    `normalize` (`"mean"`, `"median"`, `"l2"` -- the per-layer L2
    normalization that paper uses -- or `"none"`) and `min_keep_ratio`
    behave as in `prune_model_ffn`; the floor is never below
    `selector.min_keep` heads per layer.

    `stats_by_layer` (`{layer: HeadStats}`) feeds activation-aware scoring
    and bias compensation; `gradient_importance_by_layer` (the dict
    `HeadCalibrator.collect_gradient_importance` returns) feeds
    `GradientHeadSelector`. All scores are computed before any layer is
    cut, so every layer is judged against the original model.
    """
    structure = _HeadStructure(
        model, adapter, surgeon, stats_by_layer=stats_by_layer,
        gradient_importance_by_layer=gradient_importance_by_layer,
        compensate_bias=compensate_bias,
    )
    return _prune_model(structure, selector, layer_indices, allocation, normalize, min_keep_ratio)
