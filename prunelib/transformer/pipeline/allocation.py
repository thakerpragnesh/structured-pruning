"""Where a model-wide pruning budget is spent -- one strategy object per
answer.

`prune_model_ffn` and `prune_model_attention` used to each carry the same
`if allocation == "uniform": ... elif "global": ...` switch, the same
score normalisation chain, and the same validation, differing only in
whether they cut neurons or heads. A third allocation policy (say,
sensitivity-proportional budgets) would have meant a new branch in both.
Now:

- `PrunableStructure` is the small interface an allocation policy needs
  from "a kind of thing that can be pruned, layer by layer": build a
  layer's selector context, prune a layer with a selector, prune a layer
  to an explicit keep-list. `workflow` implements it once for FFN
  neurons and once for attention heads.
- `AllocationStrategy` is the policy: `UniformAllocation` and
  `GlobalAllocation` here, and anything a caller writes. Both workflow
  functions accept a strategy object as `allocation=`, so a new policy is
  a new class and no workflow edit (Open/Closed). The policy depends only
  on the structure *interface*, never on neurons or heads specifically
  (Dependency Inversion), which is why one `GlobalAllocation` serves both.

The strings `"uniform"` / `"global"` remain as shorthands (see
`resolve_allocation`), so existing calls are unchanged.
"""
from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch


def _mean_scale(scores):
    return scores.mean()


def _median_scale(scores):
    return scores.median()


def _l2_scale(scores):
    return torch.linalg.vector_norm(scores)


# `normalize=` name -> function giving the divisor that makes one layer's
# scores comparable to another's (`None` = leave raw). A new normalisation
# is a new entry here, not a new branch (Open/Closed).
SCORE_NORMALIZERS = {
    "mean": _mean_scale,      # default: divide by the layer's mean score
    "median": _median_scale,  # outlier-resistant counterpart
    "l2": _l2_scale,          # per-layer L2 normalisation (Michel et al., 2019, for heads)
    "none": None,             # the criterion is already scale-free
}


def normalize_scores(scores: torch.Tensor, normalize: str) -> torch.Tensor:
    """Divide one layer's scores by `SCORE_NORMALIZERS[normalize](scores)`.

    Raw scores are not comparable across layers -- different layers have
    different weight scales, so ranking them directly tends to delete one
    or two low-magnitude layers wholesale rather than find the genuinely
    redundant units.
    """
    if normalize not in SCORE_NORMALIZERS:
        raise ValueError(f"normalize must be one of {sorted(SCORE_NORMALIZERS)}, got {normalize!r}")
    scale_of = SCORE_NORMALIZERS[normalize]
    if scale_of is None:
        return scores
    scale = scale_of(scores)
    if not torch.isfinite(scale) or scale.abs() < 1e-12:
        # An all-zero layer would turn into NaNs and poison the global
        # ranking; leaving it unnormalised keeps it (correctly) at the
        # bottom instead.
        return scores
    return scores / scale


def global_keep_from_scores(per_layer_scores, layer_indices, prune_percent, min_keep,
                            min_keep_ratio) -> dict:
    """`{layer: keep_indices}` from already-normalised per-layer scores:
    the global top-k, then each layer floored at `min_keep_ratio` of its
    size (and never below `min_keep`)."""
    flat = torch.cat([per_layer_scores[idx] for idx in layer_indices])
    total = flat.numel()
    n_keep_total = max(1, round(total * (1 - prune_percent / 100)))
    if n_keep_total >= total:
        return {idx: list(range(per_layer_scores[idx].numel())) for idx in layer_indices}

    # Select the global top-k by index rather than by thresholding on the
    # k-th score. With a `>= threshold` test, any tie *at* the threshold
    # is kept in full, so a model with many equal scores (a uniformly
    # weighted layer, or a block of exact zeros) silently keeps more
    # units than the budget asked for.
    survivors = torch.zeros(total, dtype=torch.bool)
    survivors[torch.topk(flat, k=n_keep_total).indices] = True

    keep_by_layer = {}
    offset = 0
    for idx in layer_indices:
        scores = per_layer_scores[idx]
        n = scores.numel()
        layer_mask = survivors[offset:offset + n]
        offset += n
        floor = min(n, max(min_keep, int(round(n * min_keep_ratio))))
        if int(layer_mask.sum()) < floor:
            # This layer was hit harder than the floor allows: keep its own
            # strongest `floor` units instead of the global verdict. This
            # deliberately spends more than the budget rather than let a
            # ranking artifact collapse a layer.
            keep = torch.topk(scores, k=floor).indices
        else:
            keep = layer_mask.nonzero(as_tuple=True)[0]
        keep_by_layer[idx] = keep.tolist()
    return keep_by_layer


class PrunableStructure(ABC):
    """One kind of prunable unit (FFN neurons, attention heads) across a
    model's layers, as an `AllocationStrategy` sees it.

    `importance_type` is the selector type that can score individual units
    -- what global ranking needs. `unit` and `rankable_examples` only feed
    error messages.
    """

    importance_type: type
    unit: str = "unit"
    rankable_examples: str = ""

    @abstractmethod
    def context(self, layer_idx: int):
        """The selector context (`FFNContext` / `HeadContext`) for one layer."""

    @abstractmethod
    def prune_with(self, layer_idx: int, selector) -> int:
        """Let `selector` decide this layer's cut and apply it. Returns units kept."""

    @abstractmethod
    def prune_to(self, layer_idx: int, keep) -> int:
        """Cut this layer down to exactly the units `keep`. Returns units kept."""


class AllocationStrategy(ABC):
    """How a model-wide prune distributes its budget over layers."""

    @abstractmethod
    def apply(self, structure: PrunableStructure, selector, layer_indices) -> dict:
        """Prune `layer_indices` of `structure`. Returns `{layer: units_kept}`.

        `selector` is one selector, or a `{layer: selector}` mapping for
        strategies that support per-layer criteria.
        """


class UniformAllocation(AllocationStrategy):
    """Every layer is cut independently by its own selector: each loses the
    selector's own `prune_percent`.

    The only strategy that accepts a `{layer: selector}` mapping. That is
    for criteria whose state belongs to one layer: a
    `TwinRedundancySelector` holds one layer's twin pairs, and handing the
    same instance to every layer would apply layer 0's twins to all of them.
    """

    def apply(self, structure, selector, layer_indices) -> dict:
        if isinstance(selector, Mapping):
            missing = [idx for idx in layer_indices if idx not in selector]
            if missing:
                raise ValueError(f"No selector given for layer(s) {missing}.")
            return {idx: structure.prune_with(idx, selector[idx]) for idx in layer_indices}
        return {idx: structure.prune_with(idx, selector) for idx in layer_indices}


class GlobalAllocation(AllocationStrategy):
    """`prune_percent` is a *model-wide* budget: every unit in every layer is
    scored, ranked together, and the globally weakest are removed.

    Needs a selector of `structure.importance_type`, since ranking across
    layers needs a per-unit scalar. Every layer is scored before any is cut,
    so all are judged against the original model. `normalize` picks a
    `SCORE_NORMALIZERS` entry; `min_keep_ratio` floors how much of any one
    layer the global verdict may take.
    """

    def __init__(self, normalize: str = "mean", min_keep_ratio: float = 0.1):
        if normalize not in SCORE_NORMALIZERS:
            raise ValueError(
                f"normalize must be one of {sorted(SCORE_NORMALIZERS)}, got {normalize!r}"
            )
        self.normalize = normalize
        self.min_keep_ratio = min_keep_ratio

    def apply(self, structure, selector, layer_indices) -> dict:
        if isinstance(selector, Mapping):
            raise ValueError(
                "allocation='global' ranks every layer by one criterion, so it takes a single "
                "selector, not a per-layer mapping."
            )
        if not isinstance(selector, structure.importance_type):
            name = structure.importance_type.__name__
            article = "an" if name[0] in "AEIOU" else "a"
            raise ValueError(
                f"allocation='global' needs a per-{structure.unit} score to rank layers against "
                f"each other, but {type(selector).__name__} is not {article} {name}. Use a "
                f"scoring criterion ({structure.rankable_examples}), or allocation='uniform'."
            )
        per_layer_scores = {
            idx: normalize_scores(
                selector.importance(structure.context(idx)).detach().float().cpu(),
                self.normalize,
            )
            for idx in layer_indices
        }
        keep_by_layer = global_keep_from_scores(
            per_layer_scores, layer_indices, selector.prune_percent, selector.min_keep,
            self.min_keep_ratio,
        )
        return {idx: structure.prune_to(idx, keep_by_layer[idx]) for idx in layer_indices}


# `allocation=` string shorthand -> strategy factory taking the two global knobs.
ALLOCATIONS = {
    "uniform": lambda normalize, min_keep_ratio: UniformAllocation(),
    "global": GlobalAllocation,
}


def resolve_allocation(allocation, normalize: str = "mean",
                       min_keep_ratio: float = 0.1) -> AllocationStrategy:
    """An `AllocationStrategy` from either a strategy object (returned as-is)
    or one of the `ALLOCATIONS` shorthands, which take `normalize` and
    `min_keep_ratio` from the workflow function's own arguments."""
    if isinstance(allocation, AllocationStrategy):
        return allocation
    if allocation not in ALLOCATIONS:
        raise ValueError(
            f"allocation must be 'uniform' or 'global' (or an AllocationStrategy), "
            f"got {allocation!r}"
        )
    return ALLOCATIONS[allocation](normalize, min_keep_ratio)
