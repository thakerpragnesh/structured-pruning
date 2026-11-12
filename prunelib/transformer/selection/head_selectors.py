"""Attention-head selection strategies: which heads survive.

The head counterpart of `ffn_selectors.py`, and the piece that was missing
between `head_analysis` (diagnosis) and `attention_surgery` (compression):
`prune_attention_heads` takes raw head indices, so until now every caller
had to invent its own rule for choosing them. Each strategy here receives
a `HeadContext` and returns the sorted list of head indices to KEEP;
`workflow.prune_attention_layer` / `prune_model_attention` turn
that into surgery.

The criteria, in rough order of how much they know:

- `RedundantHeadSelector` -- structural redundancy: repeatedly drop the
  weaker head of the most similar remaining pair. `similarity="ov"`
  (default) compares OV circuits, `"query"` reproduces the thesis-era
  query-weight cosine of experiment 06, kept as the baseline it should be
  measured against.
- `OVNormHeadSelector` -- weights only: `||W_O[:, h] W_V[h]||_F`, how much
  the head *can* write into the residual stream.
- `ActivationAwareHeadSelector` -- calibrated: how much the head *does*
  write on real inputs (`head_calibration.HeadStats`).
- `GradientHeadSelector` -- loss-aware: how much the task loss depends on
  the head (Michel et al., 2019; `HeadCalibrator.collect_gradient_importance`).

All but `RedundantHeadSelector` are `HeadImportanceSelector`s -- they
produce a per-head score -- which is what `prune_model_attention`'s
global allocation needs to rank heads across layers.
"""
from abc import ABC, abstractmethod

import torch

from .budget import PruneBudgetMixin
from .context import HeadContext
from ..analysis.head_analysis import flat_query_heads, ov_norms, ov_similarity


def _validated_keep(keep, num_heads: int) -> list:
    keep = sorted({int(h) for h in keep})
    out_of_range = [h for h in keep if h < 0 or h >= num_heads]
    if out_of_range:
        raise ValueError(
            f"Head selection contains out-of-range head indices {out_of_range[:5]} "
            f"for a layer with {num_heads} heads."
        )
    if not keep:
        raise ValueError(
            "Head selection would keep zero heads, which collapses the attention block. "
            "Lower `prune_percent`."
        )
    return keep


class HeadSelector(ABC):
    """Base contract: given a `HeadContext`, return the heads to KEEP."""

    @abstractmethod
    def select(self, ctx: HeadContext) -> list:
        """Return the sorted head indices to keep."""


class HeadImportanceSelector(PruneBudgetMixin, HeadSelector):
    """Shared machinery for every "score each head, keep the best" rule --
    the head counterpart of `ffn_selectors.ImportanceSelector`, sharing its
    budget arithmetic through `budget.PruneBudgetMixin`."""

    @abstractmethod
    def importance(self, ctx: HeadContext) -> torch.Tensor:
        """Return one importance score per head; higher survives."""

    def select(self, ctx: HeadContext) -> list:
        scores = self.importance(ctx)
        keep = torch.topk(scores, k=self.n_keep(ctx.num_heads)).indices
        return _validated_keep(keep.tolist(), ctx.num_heads)


class OVNormHeadSelector(HeadImportanceSelector):
    """Scores a head by the Frobenius norm of its OV circuit,
    `||W_O[:, h] @ W_V[h]||_F`.

    Needs no data. It bounds the head's possible contribution, since the
    attention weights in `ctx_h = softmax(...) @ V_h` are a convex
    combination and cannot amplify what `W_V`/`W_O` pass through. A small
    OV norm therefore means a head that cannot matter much, whatever it
    attends to. A large one says the head *could* matter, not that it does.
    """

    def importance(self, ctx: HeadContext) -> torch.Tensor:
        return ov_norms(ctx.value_weight, ctx.output_weight, ctx.num_heads)


class ActivationAwareHeadSelector(HeadImportanceSelector):
    """Scores a head by the measured magnitude of what it writes on real
    inputs: `contribution_rms` (default) or `contribution_std` from
    `head_calibration.HeadStats`.

    Pick the statistic to match the cut: `"rms"` is the expected damage of
    deleting the head outright; `"std"` is the damage left after
    `compensate_bias=True` has restored the head's mean contribution. A
    head that writes a large but nearly constant vector scores high on the
    first and low on the second -- correctly, since compensation makes it
    almost free to remove.
    """

    def __init__(self, prune_percent: float, statistic: str = "rms", min_keep: int = 1):
        super().__init__(prune_percent, min_keep)
        if statistic not in ("rms", "std"):
            raise ValueError(f"statistic must be 'rms' or 'std', got {statistic!r}")
        self.statistic = statistic

    def importance(self, ctx: HeadContext) -> torch.Tensor:
        stats = ctx.require_stats(type(self).__name__)
        return getattr(stats, f"contribution_{self.statistic}").float()


class GradientHeadSelector(HeadImportanceSelector):
    """Scores a head by Michel et al.'s gradient importance
    `E_x |dL(x)/d xi_h|` (see `head_calibration.HeadCalibrator.collect_gradient_importance`).

    The only criterion here that sees the task loss: a head can write a
    large, varied signal that the classifier simply ignores, which every
    magnitude criterion would protect and this one will not. It is a
    first-order estimate of the loss change from masking the head, so it is
    most trustworthy for small cuts; for large ones, prune iteratively and
    re-measure.
    """

    def importance(self, ctx: HeadContext) -> torch.Tensor:
        return ctx.require_gradient_importance(type(self).__name__).float()


def _ov_similarity_and_strength(ctx: HeadContext):
    """OV-circuit cosine, ranked by OV norm: what heads *write*."""
    sim = ov_similarity(ctx.value_weight, ctx.output_weight, ctx.num_heads)
    return sim, ov_norms(ctx.value_weight, ctx.output_weight, ctx.num_heads)


def _query_similarity_and_strength(ctx: HeadContext):
    """Query-weight cosine, ranked by query norm: where heads *look*."""
    heads = flat_query_heads(ctx.query_weight, ctx.num_heads)
    unit = torch.nn.functional.normalize(heads, p=2, dim=1)
    return unit @ unit.t(), torch.linalg.vector_norm(heads, dim=1)


# `similarity=` name -> ctx -> (pairwise similarity, per-head strength). A
# new redundancy signal is a new entry here, not a new branch in
# RedundantHeadSelector (Open/Closed) -- the same convention as
# head_analysis.LP_METRICS.
HEAD_SIMILARITIES = {
    "ov": _ov_similarity_and_strength,
    "query": _query_similarity_and_strength,
}


class RedundantHeadSelector(PruneBudgetMixin, HeadSelector):
    """Drops heads that duplicate another head, until `prune_percent` of
    the layer is gone (or, with `threshold`, until no remaining pair is at
    least that similar).

    Greedy: take the most similar pair of surviving heads, drop the weaker
    of the two (lower OV norm for `similarity="ov"`, lower query norm for
    `"query"`; ties drop the higher index), repeat. Re-picking from the
    survivors after every drop is what keeps a cluster of three
    near-identical heads from losing all three -- the third pair
    involving an already-dropped head is simply never considered.

    `similarity="ov"` compares what heads *write* (OV-circuit cosine,
    `head_analysis.ov_similarity`); `"query"` compares only where they look
    (the query-weight cosine experiment 06 plots), a weaker signal kept as
    the baseline to beat. Any key of `HEAD_SIMILARITIES` is accepted.
    """

    def __init__(self, prune_percent: float, similarity: str = "ov", threshold: float = None,
                 min_keep: int = 1):
        super().__init__(prune_percent, min_keep)
        if similarity not in HEAD_SIMILARITIES:
            raise ValueError(
                f"similarity must be one of {sorted(HEAD_SIMILARITIES)}, got {similarity!r}"
            )
        if threshold is not None and not -1 <= threshold <= 1:
            raise ValueError(f"threshold is a cosine similarity, so must be in [-1, 1], got {threshold}")
        self.similarity = similarity
        self.threshold = threshold

    def select(self, ctx: HeadContext) -> list:
        num_heads = ctx.num_heads
        n_keep = self.n_keep(num_heads)
        sim, strength = HEAD_SIMILARITIES[self.similarity](ctx)
        sim = sim.detach().float().cpu().clone()
        strength = strength.detach().float().cpu().tolist()
        sim.fill_diagonal_(float("-inf"))

        alive = list(range(num_heads))
        while len(alive) > n_keep:
            sub = sim[alive][:, alive]
            flat = int(torch.argmax(sub))
            i, j = divmod(flat, len(alive))
            if self.threshold is not None and sub[i, j] < self.threshold:
                break
            a, b = sorted((alive[i], alive[j]))
            alive.remove(a if strength[a] < strength[b] else b)
        return _validated_keep(alive, num_heads)
