"""Neuron-selection strategies.

Each selector decides which neuron indices in an FFN block survive,
behind one shared contract. All of them are fully substitutable for each
other (Liskov) -- `prune_ffn_layer` never branches on which criterion is
in play, and a new criterion is a new class here, not a new `if` in the
workflow (Open/Closed).

Selectors receive an `FFNContext`, not a bare weight tensor, so a
criterion can consult the output projection and calibration statistics as
well as the incoming weights. See `context.py` for why that widening was
necessary.

The criteria, in rough order of how much they know:

- `SaliencySelector` -- incoming weights only (the thesis's Max-3 rule,
  or any other `SaliencyScorer`).
- `InOutNormSelector` -- incoming *and* outgoing weights. A neuron whose
  output column is near zero contributes almost nothing no matter how
  strongly it is driven, which a `W_in`-only rule cannot see.
- `ActivationAwareSelector` -- outgoing weights scaled by how hard the
  neuron actually fires on real data. The closest of the three to the
  quantity that actually matters, and the only one needing calibration.
- `TwinRedundancySelector` -- behavioural redundancy, ignoring weight
  magnitude entirely.
"""
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from .budget import PruneBudgetMixin
from ..analysis.clustering import kmeans_assign
from .context import FFNContext, Selection


class NeuronSelector(ABC):
    """Base contract: given an `FFNContext`, produce a `Selection`.

    Subclasses normally implement `select_keep_indices`; override
    `select` directly only when the criterion also needs to express
    merges (as `TwinRedundancySelector` does).
    """

    def select(self, ctx: FFNContext) -> Selection:
        return Selection.of(self.select_keep_indices(ctx), num_neurons=ctx.num_neurons)

    def select_keep_indices(self, ctx: FFNContext) -> list:
        """Return the indices to KEEP."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement either select_keep_indices() or select()."
        )


class ImportanceSelector(PruneBudgetMixin, NeuronSelector):
    """Shared machinery for every "score each neuron, keep the best" rule.

    Subclasses supply `importance(ctx)`; the budget arithmetic and the
    guard against pruning a layer out of existence come from
    `budget.PruneBudgetMixin`, and the top-k is written once here rather
    than re-derived per criterion.
    """

    @abstractmethod
    def importance(self, ctx: FFNContext) -> torch.Tensor:
        """Return one importance score per neuron; higher survives."""

    def select_keep_indices(self, ctx: FFNContext) -> list:
        scores = self.importance(ctx)
        keep = torch.topk(scores, k=self.n_keep(ctx.num_neurons)).indices
        return keep.tolist()


class SaliencySelector(ImportanceSelector):
    """Keeps the highest-scoring neurons under a weight-only
    `SaliencyScorer`, dropping `prune_percent`% of the lowest scorers.
    """

    def __init__(self, scorer, prune_percent: float, min_keep: int = 1):
        super().__init__(prune_percent, min_keep)
        self.scorer = scorer

    def importance(self, ctx: FFNContext) -> torch.Tensor:
        return self.scorer.score(ctx.intermediate_weight)


class InOutNormSelector(ImportanceSelector):
    """Scores a neuron by `||W_in[i]||_p * ||W_out[:, i]||_p`.

    The product form matters: a neuron is only useful if it is both
    driven by its input *and* read by the output projection, so a near-zero
    factor on either side should sink it. A sum would let a large input
    norm mask a dead output column.
    """

    def __init__(self, prune_percent: float, p: float = 2.0, min_keep: int = 1):
        super().__init__(prune_percent, min_keep)
        self.p = p

    def importance(self, ctx: FFNContext) -> torch.Tensor:
        w_in = ctx.intermediate_weight.detach()
        w_out = ctx.output_weight.detach()
        in_norm = torch.linalg.vector_norm(w_in.reshape(w_in.shape[0], -1).float(), ord=self.p, dim=1)
        # W_out is (hidden, neurons): a neuron's outgoing weights are a
        # *column*, so this norm runs down dim 0, not across dim 1.
        out_norm = torch.linalg.vector_norm(w_out.float(), ord=self.p, dim=0)
        return in_norm * out_norm


class ActivationAwareSelector(ImportanceSelector):
    """Scores a neuron by the expected magnitude of what it actually
    contributes to the layer output: `||W_out[:, i]||_p * E[|a_i|]`.

    Weight-only criteria measure capacity; this measures use. A neuron
    can be strongly wired and still near-useless because it almost never
    fires on the target distribution -- common in a fine-tuned model,
    where the pretrained FFN carries capacity the downstream task never
    exercises. Requires calibration (`ffn_calibration.FFNCalibrator`).

    `statistic` picks which activation moment scales the output norm:
    `"mean_abs"` (default) for average contribution magnitude, or
    `"rms"` to weight occasional large excursions more heavily.
    """

    def __init__(self, prune_percent: float, p: float = 2.0, statistic: str = "mean_abs", min_keep: int = 1):
        super().__init__(prune_percent, min_keep)
        if statistic not in ("mean_abs", "rms"):
            raise ValueError(f"statistic must be 'mean_abs' or 'rms', got {statistic!r}")
        self.p = p
        self.statistic = statistic

    def importance(self, ctx: FFNContext) -> torch.Tensor:
        stats = ctx.require_stats(type(self).__name__)
        w_out = ctx.output_weight.detach().float()
        out_norm = torch.linalg.vector_norm(w_out, ord=self.p, dim=0)
        scale = getattr(stats, self.statistic).to(out_norm.device, out_norm.dtype)
        return out_norm * scale


class TwinRedundancySelector(NeuronSelector):
    """Drops the higher-indexed neuron of every co-activation "twin" pair
    (see `redundancy.JaccardTwinFinder`).

    With `merge=True` the dropped twin's output column is folded into its
    survivor rather than discarded, which is the point of calling them
    twins: if `a_j ~ a_i`, then `W_out[:, j] * a_j ~ W_out[:, j] * a_i`,
    so adding `W_out[:, j]` onto `W_out[:, i]` reproduces the removed
    contribution instead of silently deleting it. The scale is the
    least-squares-optimal `E[a_j * a_i] / E[a_i^2]` when `ctx.cross_moments`
    has the pair (see `_merge_scale`), the mean-matching `E[a_j] / E[a_i]`
    when only ordinary calibration is available, and 1.0 when neither is --
    1.0 is exactly right when the twins fire at equal strength, which is
    the regime a high Jaccard threshold selects for.

    Chains resolve transitively: given twins (i, j) and (j, k), `k` is
    merged into `i`, not into the already-doomed `j`.
    """

    def __init__(self, twin_pairs, merge: bool = False):
        self.twin_pairs = list(twin_pairs)
        self.merge = merge
        # Union-find over the twin graph: every connected group of twins
        # collapses onto its lowest index. Resolving each pair on its own
        # made the result depend on pair order -- `JaccardTwinFinder` sorts
        # by overlap, not index, so (4, 6) arriving before (1, 4) mapped 6
        # into 4 and then dropped 4, a merge into a removed neuron.
        parent = {}

        def find(node):
            root = node
            while parent.get(root, root) != root:
                root = parent[root]
            while parent.get(node, node) != root:
                parent[node], node = root, parent[node]
            return root

        for pair in self.twin_pairs:
            root_a, root_b = find(int(pair[0])), find(int(pair[1]))
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)

        # survivor[j] = the neuron j is represented by, always a root.
        self._survivor = {node: find(node) for node in parent}
        self.drop_indices = set(self._survivor)

    @property
    def merge_pairs(self) -> list:
        """`[(dropped, survivor), ...]`, sorted -- the pairs a merge folds,
        and so the pairs `FFNCalibrator.collect_cross_moments` must measure
        for the least-squares scale. Not the same as `twin_pairs`: a chain
        (i, j), (j, k) merges `k` into `i`, a pair no twin list names.
        """
        return sorted(self._survivor.items())

    def select(self, ctx: FFNContext) -> Selection:
        keep = [i for i in range(ctx.num_neurons) if i not in self.drop_indices]
        merge_map = None
        if self.merge:
            merge_map = {}
            for dropped, survivor in self._survivor.items():
                merge_map[dropped] = (
                    survivor, _merge_scale(ctx.stats, dropped, survivor, ctx.cross_moments)
                )
        return Selection.of(keep, merge_map=merge_map, num_neurons=ctx.num_neurons)


class WeightClusterRedundancySelector(PruneBudgetMixin, NeuronSelector):
    """Groups neurons by incoming-weight *direction* and keeps one
    representative per cluster -- redundancy detected from what a neuron
    *is* (its weight vector), which complements `TwinRedundancySelector`
    seeing redundancy from how a neuron *behaves* on real data. Needs no
    calibration corpus to run (though calibration still improves the
    merge, if requested).

    Adapted from the K-Means channel-selection method in Thaker & Mohan,
    "Channel Pruning of Transfer Learning Models Using Novel Techniques"
    (IEEE Access, 2024): cluster channels by weight similarity, then keep
    the highest-L1-norm channel per cluster and prune the rest. Ported
    from conv channels to FFN neurons -- a BERT neuron's `W_in` row is
    already the 1D feature vector that paper had to condense a 2D kernel
    down to (their Equation 1), so no condensing step is needed here.

    Rows are L2-normalized before clustering, unlike the source paper, so
    clusters form on weight *direction* rather than magnitude: two
    neurons pointing the same way are redundant candidates even if one
    fires much louder than the other, and merging (with calibration)
    compensates for that magnitude gap via the same scale
    `TwinRedundancySelector` uses.
    """

    def __init__(
        self,
        prune_percent: float,
        min_keep: int = 1,
        merge: bool = False,
        kmeans_iters: int = 50,
        seed: int = 0,
    ):
        super().__init__(prune_percent, min_keep)
        self.merge = merge
        self.kmeans_iters = kmeans_iters
        self.seed = seed

    def select(self, ctx: FFNContext) -> Selection:
        num_neurons = ctx.num_neurons
        n_clusters = self.n_keep(num_neurons)
        if n_clusters >= num_neurons:
            return Selection.of(list(range(num_neurons)), num_neurons=num_neurons)

        w_in = ctx.intermediate_weight.detach().float()
        directions = F.normalize(w_in, p=2, dim=1, eps=1e-12)
        assignment = kmeans_assign(directions, n_clusters, iters=self.kmeans_iters, seed=self.seed)

        l1_norms = w_in.abs().sum(dim=1)
        keep = []
        survivor_of = {}
        for cluster in torch.unique(assignment).tolist():
            members = (assignment == cluster).nonzero(as_tuple=True)[0].tolist()
            representative = max(members, key=lambda i: l1_norms[i].item())
            keep.append(representative)
            for member in members:
                if member != representative:
                    survivor_of[member] = representative

        # K-Means can leave clusters empty -- always when fewer distinct
        # directions exist than clusters asked for (duplicate or all-zero
        # rows) -- and one survivor per non-empty cluster would then prune
        # more than `prune_percent`. Refill the budget with the strongest
        # remaining neurons, which stop being merged away.
        shortfall = n_clusters - len(keep)
        if shortfall > 0:
            extra = sorted(survivor_of, key=lambda i: l1_norms[i].item(), reverse=True)[:shortfall]
            for member in extra:
                keep.append(member)
                del survivor_of[member]

        merge_map = None
        if self.merge:
            merge_map = {
                dropped: (survivor, _merge_scale(ctx.stats, dropped, survivor, ctx.cross_moments))
                for dropped, survivor in survivor_of.items()
            }
        return Selection.of(sorted(keep), merge_map=merge_map, num_neurons=num_neurons)


def _merge_scale(stats, dropped: int, survivor: int, cross_moments=None, eps: float = 1e-6) -> float:
    """The merge scale folding `dropped`'s output column into `survivor`'s.

    Prefers the variance-minimizing scale `s* = E[a_dropped * a_survivor] /
    E[a_survivor^2]` -- the `s` that minimizes `E[(a_dropped - s *
    a_survivor)^2]`, i.e. the least-squares fit of `a_dropped` as a multiple
    of `a_survivor` -- when `cross_moments` has the needed pair (from
    `ffn_calibration.FFNCalibrator.collect_cross_moments`). `E[a_survivor^2]` is
    `stats.rms[survivor] ** 2`, already computed by ordinary calibration;
    only the cross term `E[a_dropped * a_survivor]` needs the extra pass.

    Falls back to the mean-matching scale `E[a_dropped] / E[a_survivor]`
    when no cross moment was supplied for this pair -- exactly what this
    function computed before least-squares scales existed, so a caller
    that never collects cross moments sees no change in behavior. Falls
    back further to 1.0 when even that is unavailable or numerically
    unsafe: a GELU neuron's *signed* mean can sit near zero even when it
    is highly active, and an unguarded ratio would then blow the merged
    column up by orders of magnitude -- turning a conservative merge into
    a worse perturbation than the plain drop it replaced. `E[a^2] >= 0`,
    so `rms[survivor] ** 2` is only ever unsafe by being too small, never
    by cancellation the way the signed mean can be.
    """
    if stats is None:
        return 1.0
    if cross_moments is not None:
        pair = (min(dropped, survivor), max(dropped, survivor))
        cross = cross_moments.get(pair)
        if cross is not None:
            denom = float(stats.rms[survivor]) ** 2
            if denom >= eps:
                return cross / denom
    denom = float(stats.mean[survivor])
    if abs(denom) < eps:
        return 1.0
    return float(stats.mean[dropped]) / denom
