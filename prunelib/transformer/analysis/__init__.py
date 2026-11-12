"""Signals computed from weights or recorded firings, independent of any
decision about what to prune.

- `scoring` -- `SaliencyScorer`s (Max-3 / top-k magnitude, Lp norm, CSD).
- `network_scanner` -- `NetworkSaliencyScanner`: per-layer saliency reports.
- `head_analysis` -- `AttentionHeadAnalyzer` and the OV-circuit math
  (`ov_gram`, `ov_norms`, `ov_similarity`, `flat_query_heads`).
- `redundancy` -- `JaccardTwinFinder`: co-activation twins and dead neurons.
- `clustering` -- `kmeans_assign`.

Depends on: `models` (only `head_analysis`, to resolve an adapter).
"""
