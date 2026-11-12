"""Orchestration: wiring criteria to surgery on a live model, one layer or
the whole stack.

- `allocation` -- `AllocationStrategy` (`UniformAllocation`,
  `GlobalAllocation`), the `PrunableStructure` interface they work through,
  and the `SCORE_NORMALIZERS` registry.
- `workflow` -- `prune_ffn_layer`, `prune_model_ffn`, `prune_attention_heads`,
  `prune_attention_layer`, `prune_model_attention`.

Depends on: `models`, `selection`, `surgery`. Nothing else in the package
depends on this subpackage, so it is the only place that knows about all of
them.
"""
