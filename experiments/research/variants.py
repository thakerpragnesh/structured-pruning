"""Every pruning condition the study compares, by name.

A variant is `apply(model, percent, ctx)`: prune `model` in place at
`percent`, calibrating on `ctx` if it needs data. `run_study.py` owns
everything else (baseline, evaluation, healing, logging), so adding a
condition is one entry in `VARIANTS` and nothing more.

Grouped by research question:

- RQ1-FFN   -- do the thesis's CNN criteria (Max-3, L1/L2, CSD, K-means)
               transfer to BERT FFN neurons, and how do they compare with
               output-aware and activation-aware criteria? Uniform
               allocation: every layer loses `percent`.
- RQ1-heads -- the same question for attention heads (query-weight
               redundancy is the thesis-era signal; OV-circuit, activation
               and gradient criteria are the alternatives).
- RQ2-order -- does pruning order matter in transformers, as it did for
               CNNs (Thaker & Mohan, ICCCNT 2023)? Heads then FFN vs. FFN
               then heads, and whether re-calibrating between the two
               steps matters. `percent` applies to both structures.

Percent semantics: with `allocation="uniform"` every layer loses `percent`
of its units; with `"global"` `percent` is a model-wide budget, so layers
lose different amounts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from prunelib.transformer import (
    ActivationAwareHeadSelector,
    ActivationAwareSelector,
    CSDScorer,
    FFNCalibrator,
    GradientHeadSelector,
    HeadCalibrator,
    HeadImportanceSelector,
    InOutNormSelector,
    OVNormHeadSelector,
    RedundantHeadSelector,
    RegisteredScorer,
    SaliencySelector,
    TopKMagnitudeScorer,
    WeightClusterRedundancySelector,
    prune_model_attention,
    prune_model_ffn,
)


@dataclass
class Context:
    """What a variant may use besides the model: calibration data."""

    harness: object
    calib_examples: int = 512

    def batches(self):
        return self.harness.calibration_batches(limit=self.calib_examples)

    def labelled_batches(self):
        return self.harness.calibration_batches(limit=self.calib_examples, with_labels=True)


@dataclass(frozen=True)
class Variant:
    group: str
    description: str
    apply: Callable


class RandomHeadSelector(HeadImportanceSelector):
    """Random head scores -- the floor every head criterion must beat.
    Seeded by the global RNG, which `run_study` seeds per run."""

    def importance(self, ctx):
        return torch.rand(ctx.num_heads)


# ----------------------------------------------------------------- FFN steps
def _ffn(selector_factory, *, calibrate=False, compensate=False, allocation="uniform"):
    def apply(model, percent, ctx):
        stats = FFNCalibrator(model).collect_many(ctx.batches()) if (calibrate or compensate) else None
        return prune_model_ffn(model, selector_factory(percent), allocation=allocation,
                               stats_by_layer=stats, compensate_bias=compensate)
    return apply


def _ffn_cluster(merge: bool):
    def apply(model, percent, ctx):
        stats = FFNCalibrator(model).collect_many(ctx.batches())
        return prune_model_ffn(model, WeightClusterRedundancySelector(percent, merge=merge),
                               stats_by_layer=stats, compensate_bias=True)
    return apply


# ---------------------------------------------------------------- head steps
def _heads(selector_factory, *, stats=False, gradient=False, compensate=False, allocation="uniform"):
    def apply(model, percent, ctx):
        calibrator = HeadCalibrator(model)
        head_stats = calibrator.collect_many(ctx.batches()) if (stats or compensate) else None
        importance = calibrator.collect_gradient_importance(ctx.labelled_batches()) if gradient else None
        return prune_model_attention(
            model, selector_factory(percent), allocation=allocation, normalize="l2",
            stats_by_layer=head_stats, gradient_importance_by_layer=importance,
            compensate_bias=compensate,
        )
    return apply


# The strongest calibrated criterion for each structure, used by RQ2.
_BEST_HEADS = _heads(GradientHeadSelector, gradient=True, compensate=True, allocation="global")
_BEST_FFN = _ffn(ActivationAwareSelector, calibrate=True, compensate=True, allocation="global")


def _sequence(*steps):
    def apply(model, percent, ctx):
        return [step(model, percent, ctx) for step in steps]
    return apply


def _heads_then_ffn_stale(model, percent, ctx):
    """FFN statistics measured on the *unpruned* model, then heads cut, then
    the FFN cut with those stale statistics -- the ablation for the
    "re-calibrate between steps" rule."""
    stale = FFNCalibrator(model).collect_many(ctx.batches())
    _BEST_HEADS(model, percent, ctx)
    return prune_model_ffn(model, ActivationAwareSelector(percent), allocation="global",
                           stats_by_layer=stale, compensate_bias=True)


VARIANTS: dict[str, Variant] = {
    # RQ1 -- FFN neurons
    "ffn_random": Variant("rq1-ffn", "random scores (floor)",
                          _ffn(lambda p: SaliencySelector(RegisteredScorer("random"), p))),
    "ffn_l1": Variant("rq1-ffn", "L1 norm of W_in row", _ffn(lambda p: SaliencySelector(RegisteredScorer("l1"), p))),
    "ffn_l2": Variant("rq1-ffn", "L2 norm of W_in row", _ffn(lambda p: SaliencySelector(RegisteredScorer("l2"), p))),
    "ffn_max3": Variant("rq1-ffn", "Max-3: sum of 3 strongest incoming weights (thesis)",
                        _ffn(lambda p: SaliencySelector(TopKMagnitudeScorer(3), p))),
    "ffn_csd": Variant("rq1-ffn", "CSD dispersion of W_in row (thesis)", _ffn(lambda p: SaliencySelector(CSDScorer(), p))),
    "ffn_cluster": Variant("rq1-ffn", "K-means on W_in direction, keep max-L1 per cluster (thesis)", _ffn_cluster(False)),
    "ffn_cluster_merge": Variant("rq1-ffn", "K-means + fold removed neurons into survivor", _ffn_cluster(True)),
    "ffn_inout": Variant("rq1-ffn", "||W_in[i]|| * ||W_out[:, i]||", _ffn(InOutNormSelector)),
    "ffn_act": Variant("rq1-ffn", "||W_out[:, i]|| * E|a_i| (calibrated)", _ffn(ActivationAwareSelector, calibrate=True)),
    "ffn_max3_comp": Variant("rq1-ffn", "Max-3 + bias compensation",
                             _ffn(lambda p: SaliencySelector(TopKMagnitudeScorer(3), p), compensate=True)),
    "ffn_act_comp": Variant("rq1-ffn", "activation-aware + bias compensation",
                            _ffn(ActivationAwareSelector, calibrate=True, compensate=True)),
    "ffn_act_global_comp": Variant("rq1-ffn", "activation-aware + comp., global allocation", _BEST_FFN),
    # RQ1 -- attention heads
    "heads_random": Variant("rq1-heads", "random scores (floor)", _heads(RandomHeadSelector)),
    "heads_query_redundancy": Variant("rq1-heads", "query-weight cosine redundancy (thesis-era)",
                                      _heads(lambda p: RedundantHeadSelector(p, similarity="query"))),
    "heads_ov_redundancy": Variant("rq1-heads", "OV-circuit cosine redundancy",
                                   _heads(lambda p: RedundantHeadSelector(p, similarity="ov"))),
    "heads_ov_norm": Variant("rq1-heads", "OV-circuit norm (weights only)", _heads(OVNormHeadSelector)),
    "heads_act_rms": Variant("rq1-heads", "measured contribution RMS", _heads(ActivationAwareHeadSelector, stats=True)),
    "heads_act_std_comp": Variant("rq1-heads", "contribution std + mean-ablation",
                                  _heads(lambda p: ActivationAwareHeadSelector(p, statistic="std"), stats=True,
                                         compensate=True)),
    "heads_grad": Variant("rq1-heads", "gradient importance (Michel et al. 2019)",
                          _heads(GradientHeadSelector, gradient=True)),
    "heads_grad_global_comp": Variant("rq1-heads", "gradient + comp., global allocation", _BEST_HEADS),
    # RQ2 -- order
    "order_heads_then_ffn": Variant("rq2-order", "heads, re-calibrate, then FFN", _sequence(_BEST_HEADS, _BEST_FFN)),
    "order_ffn_then_heads": Variant("rq2-order", "FFN, re-calibrate, then heads", _sequence(_BEST_FFN, _BEST_HEADS)),
    "order_heads_then_ffn_stale": Variant("rq2-order", "heads then FFN, FFN stats from unpruned model",
                                          _heads_then_ffn_stale),
}

PRESETS = {
    "quick": ["ffn_random", "ffn_max3", "ffn_act_comp", "heads_grad", "order_heads_then_ffn"],
    "rq1-ffn": [n for n, v in VARIANTS.items() if v.group == "rq1-ffn"],
    "rq1-heads": [n for n, v in VARIANTS.items() if v.group == "rq1-heads"],
    "rq2-order": [n for n, v in VARIANTS.items() if v.group == "rq2-order"]
                 + ["ffn_act_global_comp", "heads_grad_global_comp"],
    "all": list(VARIANTS),
}


def resolve(names_or_preset: list[str]) -> list[str]:
    """Expand presets, keep order, drop duplicates, reject unknown names."""
    out = []
    for item in names_or_preset:
        for name in PRESETS.get(item, [item]):
            if name not in VARIANTS:
                raise ValueError(f"unknown variant or preset {name!r}; "
                                 f"presets: {sorted(PRESETS)}, variants: {sorted(VARIANTS)}")
            if name not in out:
                out.append(name)
    return out
