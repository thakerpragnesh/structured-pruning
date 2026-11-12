"""Pruning criteria: given what is known about one layer, which units
survive.

- `context` -- `FFNContext` / `HeadContext` (what a criterion may look at)
  and `Selection` (what an FFN criterion hands to surgery).
- `budget` -- `PruneBudgetMixin` / `keep_count`: the `prune_percent` rule
  every budgeted selector shares.
- `ffn_selectors` -- `NeuronSelector` and the FFN criteria (saliency, in/out
  norm, activation-aware, twin and weight-cluster redundancy).
- `head_selectors` -- `HeadSelector` and the head criteria (OV norm,
  activation-aware, gradient, redundancy).

Depends on: `analysis` (scores, clustering, OV math) and `measurement` (for
the stats types the contexts carry, as annotations only).
"""
