"""The pruning budget: how many units (FFN neurons, attention heads) a
`prune_percent` leaves standing.

Four selectors take a fractional budget -- `ImportanceSelector`,
`WeightClusterRedundancySelector`, `HeadImportanceSelector`,
`RedundantHeadSelector` -- and each used to validate and store it itself,
with the head selectors importing `ffn_selectors.py`'s private helpers to do
so. The rule is easy to get subtly wrong (CHANGELOG 0.2.0, bug 6: `int()`
truncation silently over-pruned), so it is written once, here, and the
selectors inherit it from `PruneBudgetMixin` instead of re-deriving it.
"""


def validate_budget(prune_percent: float, min_keep: int) -> None:
    if not 0 <= prune_percent < 100:
        raise ValueError(
            f"prune_percent must be in [0, 100), got {prune_percent}. "
            "100 would delete every unit in the layer."
        )
    if min_keep < 1:
        raise ValueError(f"min_keep must be >= 1, got {min_keep}")


def keep_count(num_units: int, prune_percent: float, min_keep: int) -> int:
    """How many of `num_units` survive a `prune_percent` cut, floored at
    `min_keep` and capped at `num_units`."""
    # round, not truncate. int() always truncates toward zero, so it
    # prunes one unit more than asked whenever the product isn't whole
    # -- e.g. 768 neurons at 40% keeps 460 rather than 461. Small per
    # layer, but it is a one-directional bias that compounds across a
    # whole-model sweep.
    keep = round(num_units * (1 - prune_percent / 100))
    return max(min_keep, min(num_units, keep))


class PruneBudgetMixin:
    """`prune_percent` / `min_keep` / `n_keep()` for any selector with a
    fractional budget. Validates at construction, so a bad budget fails
    where it was written, not at the first `select()`.

    `allocation.GlobalAllocation` reads `prune_percent` and `min_keep` off
    the selector, which is why they stay plain attributes.
    """

    def __init__(self, prune_percent: float, min_keep: int = 1):
        validate_budget(prune_percent, min_keep)
        self.prune_percent = prune_percent
        self.min_keep = min_keep

    def n_keep(self, num_units: int) -> int:
        return keep_count(num_units, self.prune_percent, self.min_keep)
