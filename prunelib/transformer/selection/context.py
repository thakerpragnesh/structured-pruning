"""What a `NeuronSelector` gets to look at when it decides which FFN
neurons survive (`FFNContext`), and what a `head_selectors.HeadSelector`
gets for attention heads (`HeadContext`).

Selectors used to receive a bare `intermediate.weight` tensor. That was
enough for the thesis's Max-3 rule but structurally too narrow: a
neuron's actual contribution to the layer output is
`W_out[:, i] * act(W_in[i] @ x + b_i)`, so any criterion that sees only
`W_in` is reasoning about half the neuron. Widening the argument to a
context object means new criteria can reach for the output projection or
for calibration statistics without another signature change rippling
through every existing selector (Open/Closed) -- which is exactly what
happened the first time, and the reason this type exists.

The context is a read-only *view*: it hands out the live parameter
tensors, and nothing that consumes it is permitted to mutate them.
Resizing is `ffn_surgery.FFNSurgeon`'s job alone.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:  # annotations only: contexts carry stats, they never compute them
    from ..measurement.ffn_calibration import CalibrationStats
    from ..measurement.head_calibration import HeadStats


@dataclass(frozen=True)
class FFNContext:
    """Everything known about one FFN block at selection time.

    `stats` is `None` whenever the caller did not calibrate. Criteria
    that cannot work without it should call `require_stats()` rather than
    dereferencing it, so the failure names the missing step instead of
    surfacing as `AttributeError: 'NoneType'`.
    """

    intermediate_weight: torch.Tensor
    output_weight: torch.Tensor
    intermediate_bias: Optional[torch.Tensor] = None
    output_bias: Optional[torch.Tensor] = None
    stats: Optional["CalibrationStats"] = None
    layer_index: Optional[int] = None
    # {(i, j): E[a_i * a_j]} for i < j, from ffn_calibration.FFNCalibrator.collect_cross_moments,
    # restricted to whichever pairs a merge-capable selector actually needs (never a full
    # (neurons, neurons) Gram matrix -- see TwinRedundancySelector / _merge_scale in
    # ffn_selectors.py for why only these specific off-diagonal terms are worth computing).
    cross_moments: Optional[dict] = None

    @property
    def num_neurons(self) -> int:
        return self.intermediate_weight.shape[0]

    def require_stats(self, criterion_name: str):
        """Return `stats`, or explain which step the caller skipped."""
        if self.stats is None:
            raise ValueError(
                f"{criterion_name} needs calibration statistics, but this FFNContext has "
                "stats=None. Collect them first with "
                "`ffn_calibration.FFNCalibrator(model).collect(batches, layer_idx)` and pass the "
                "result through (e.g. `prune_ffn_layer(..., stats=stats)`)."
            )
        if self.stats.num_neurons != self.num_neurons:
            raise ValueError(
                f"{criterion_name} got calibration stats for {self.stats.num_neurons} neurons "
                f"but this layer has {self.num_neurons}. They were most likely collected "
                "against a different layer, or against this one before an earlier pruning pass."
            )
        return self.stats

    @classmethod
    def from_layers(cls, intermediate, output, stats=None, layer_index=None,
                    cross_moments=None) -> "FFNContext":
        """Build a context from a live `(intermediate, output)` Linear pair."""
        return cls(
            intermediate_weight=intermediate.weight.data,
            output_weight=output.weight.data,
            intermediate_bias=None if intermediate.bias is None else intermediate.bias.data,
            output_bias=None if output.bias is None else output.bias.data,
            stats=stats,
            layer_index=layer_index,
            cross_moments=cross_moments,
        )


@dataclass(frozen=True)
class Selection:
    """What a `NeuronSelector` hands to `ffn_surgery.FFNSurgeon`.

    `keep_indices` alone was the original contract, and it forces every
    criterion into the same crude move: delete the loser outright.
    `merge_map` adds the alternative -- fold a doomed neuron's output
    column into a surviving one instead of discarding it. For a
    redundancy criterion that is the whole point: `JaccardTwinFinder`
    flags neuron `j` precisely *because* it behaves like neuron `i`, so
    `W_out[:, j] * a_j` is not noise to be thrown away, it is signal that
    `W_out[:, i] * a_i` can carry.

    `merge_map` maps a dropped neuron index to `(survivor_index, scale)`;
    surgery adds `scale * W_out[:, dropped]` onto `W_out[:, survivor]`.
    """

    keep_indices: list
    merge_map: Optional[dict] = None

    @classmethod
    def of(cls, keep_indices, merge_map=None, num_neurons=None) -> "Selection":
        """Normalise and validate a raw selection.

        Sorts and de-duplicates `keep_indices` -- surgery slices weights
        with them, so their order silently becomes the new neuron order,
        and an unsorted list would permute the layer for no reason.
        """
        keep = sorted({int(i) for i in keep_indices})
        if num_neurons is not None:
            out_of_range = [i for i in keep if i < 0 or i >= num_neurons]
            if out_of_range:
                raise ValueError(
                    f"keep_indices contains out-of-range neuron indices {out_of_range[:5]} "
                    f"for a layer with {num_neurons} neurons."
                )
        if not keep:
            raise ValueError(
                "Selection would keep zero neurons, which collapses the FFN block. "
                "Lower `prune_percent`, or relax the redundancy threshold."
            )
        if merge_map:
            kept = set(keep)
            normalised = {}
            for dropped, target in merge_map.items():
                survivor, scale = target if isinstance(target, tuple) else (target, 1.0)
                dropped, survivor = int(dropped), int(survivor)
                if dropped in kept:
                    raise ValueError(
                        f"merge_map wants to merge neuron {dropped} away, but it is also in "
                        "keep_indices -- a neuron cannot both survive and be folded into another."
                    )
                if survivor not in kept:
                    raise ValueError(
                        f"merge_map sends neuron {dropped} into neuron {survivor}, which is not "
                        "being kept. The survivor of a merge must itself survive."
                    )
                normalised[dropped] = (survivor, float(scale))
            merge_map = normalised
        return cls(keep_indices=keep, merge_map=merge_map or None)


@dataclass(frozen=True)
class HeadContext:
    """Everything known about one multi-head attention block at
    head-selection time -- the attention counterpart of `FFNContext`,
    consumed by `head_selectors.HeadSelector`s.

    A head's contribution to the block output is
    `W_O[:, h] @ softmax(Q_h K_h^T) V_h`, so like an FFN neuron it is not
    described by any single weight matrix: the attention pattern comes
    from `query`/`key`, what gets written comes from `value`/`output` (the
    "OV circuit"). The context therefore carries all four, plus the two
    optional data-driven signals -- `stats` (activation moments, from
    `head_calibration.HeadCalibrator.collect`) and `gradient_importance`
    (`HeadCalibrator.collect_gradient_importance`) -- that a criterion may
    need. Like `FFNContext`, it is a read-only view.
    """

    query_weight: torch.Tensor
    key_weight: torch.Tensor
    value_weight: torch.Tensor
    output_weight: torch.Tensor
    num_heads: int
    stats: Optional["HeadStats"] = None
    gradient_importance: Optional[torch.Tensor] = None  # (num_heads,)
    layer_index: Optional[int] = None

    def __post_init__(self):
        rows = self.query_weight.shape[0]
        if self.num_heads < 1 or rows % self.num_heads:
            raise ValueError(
                f"query output size {rows} is not divisible by num_heads {self.num_heads}; "
                "the adapter is reporting a head layout this layer does not have."
            )
        if self.output_weight.shape[1] != rows:
            raise ValueError(
                f"Attention block is inconsistent: query/key/value produce {rows} features "
                f"but the output projection expects {self.output_weight.shape[1]} inputs."
            )

    @property
    def head_dim(self) -> int:
        return self.query_weight.shape[0] // self.num_heads

    def require_stats(self, criterion_name: str):
        """Return `stats`, or explain which step the caller skipped."""
        if self.stats is None:
            raise ValueError(
                f"{criterion_name} needs head calibration statistics, but this HeadContext has "
                "stats=None. Collect them first with "
                "`head_calibration.HeadCalibrator(model).collect(batches, layer_idx)` and pass "
                "the result through (e.g. `prune_attention_layer(..., stats=stats)`)."
            )
        if self.stats.num_heads != self.num_heads:
            raise ValueError(
                f"{criterion_name} got head stats for {self.stats.num_heads} heads but this layer "
                f"has {self.num_heads}. They were most likely collected against a different "
                "layer, or against this one before an earlier pruning pass."
            )
        return self.stats

    def require_gradient_importance(self, criterion_name: str) -> torch.Tensor:
        """Return `gradient_importance`, or explain which step the caller skipped."""
        if self.gradient_importance is None:
            raise ValueError(
                f"{criterion_name} needs per-head gradient importance, but this HeadContext has "
                "gradient_importance=None. Collect it first with "
                "`head_calibration.HeadCalibrator(model).collect_gradient_importance(batches)` "
                "and pass the layer's entry through (e.g. `prune_attention_layer(..., "
                "gradient_importance=importance[layer_idx])`)."
            )
        if self.gradient_importance.numel() != self.num_heads:
            raise ValueError(
                f"{criterion_name} got gradient importance for "
                f"{self.gradient_importance.numel()} heads but this layer has {self.num_heads}."
            )
        return self.gradient_importance

    @classmethod
    def from_layers(cls, query, key, value, output, num_heads, stats=None,
                    gradient_importance=None, layer_index=None) -> "HeadContext":
        """Build a context from a live `(query, key, value, output)` Linear quartet."""
        return cls(
            query_weight=query.weight.data,
            key_weight=key.weight.data,
            value_weight=value.weight.data,
            output_weight=output.weight.data,
            num_heads=num_heads,
            stats=stats,
            gradient_importance=gradient_importance,
            layer_index=layer_index,
        )
