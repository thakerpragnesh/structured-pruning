"""Reports per-layer saliency statistics and lowest-scoring units.

Takes its layer list and its scoring rule as dependencies (Dependency
Inversion) instead of hardcoding either, so the same class covers both a
pure-CNN pass and a mixed Conv2d+Linear pass -- what used to be two
near-duplicate classes (`SaliencyPruner`, `UniversalPruner`) differing
only in which layer types and scorer they hardcoded.
"""
import torch


class NetworkSaliencyScanner:
    def __init__(self, layers, scorer, cache=True):
        self.layers = layers
        self.scorer = scorer
        # Scoring a layer is a full pass over its weights; `scan()` then
        # `weakest_units()` on the same layer would otherwise pay for it
        # twice. Disable if you intend to mutate weights between calls.
        self._cache = {} if cache else None

    def _scores(self, layer) -> torch.Tensor:
        if self._cache is not None and layer.name in self._cache:
            return self._cache[layer.name]
        scores = self.scorer.score(layer.module.weight.data)
        if self._cache is not None:
            self._cache[layer.name] = scores
        return scores

    def invalidate(self):
        """Drop cached scores, e.g. after pruning or fine-tuning."""
        if self._cache is not None:
            self._cache.clear()

    def scan(self):
        """Per-layer summary as a `pandas.DataFrame` (pandas is imported
        here, not at module load, so it stays an optional dependency)."""
        import pandas as pd

        rows = []
        for layer in self.layers:
            scores = self._scores(layer).float()
            # One device->host transfer per layer instead of three: each
            # .item() is a separate synchronising copy, which on GPU costs
            # far more than the reduction it returns.
            summary = torch.stack([scores.mean(), scores.min(), scores.std()]).tolist()
            rows.append(
                {
                    "Layer": layer.name,
                    "Type": layer.kind,
                    "Units": layer.module.weight.shape[0],
                    "Avg Score": summary[0],
                    "Min Score": summary[1],
                    "StdDev": summary[2],
                }
            )
        return pd.DataFrame(rows)

    def weakest_units(self, layer_name, n=5):
        layer = next((l for l in self.layers if l.name == layer_name), None)
        if layer is None:
            known = ", ".join(l.name for l in self.layers[:5])
            raise ValueError(
                f"Layer '{layer_name}' not found among scanned layers (have: {known}...)."
            )
        scores = self._scores(layer)
        n = max(0, min(n, scores.shape[0]))
        values, indices = torch.topk(scores, k=n, largest=False)
        return list(zip(indices.tolist(), values.tolist()))
