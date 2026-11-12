"""Per-neuron activation statistics for one FFN layer, measured on real
inputs.

Weight-only criteria (`scoring.py`) can only see how *strongly* a neuron
is wired; they cannot see how *often* or how *hard* it actually fires.
Two neurons with identical incoming weights contribute very differently
to the layer output if one saturates on real text and the other is
almost always dead. `CalibrationStats` supplies that missing half, and
is what `ffn_selectors.ActivationAwareSelector` and the bias-compensation
path in `ffn_surgery.FFNSurgeon` consume.

Statistics are accumulated in a streaming fashion -- one small `(neurons,)`
reduction per batch -- so calibrating on a large corpus never materialises
the full `(tokens, neurons)` activation tensor the way
`activation_recording.ActivationRecorder` deliberately does. The two are
complementary, not redundant: the recorder keeps per-token detail because
`redundancy.JaccardTwinFinder` needs to compare firing *patterns*; the
calibrator throws per-token detail away because nothing downstream of it
needs more than a mean.

`FFNCalibrator.collect_many` gathers several layers' stats from one pass
over the corpus -- every forward pass computes every layer anyway, so
calibrating a 12-layer model with 12 `collect()` calls paid for 11 passes
it didn't need.

`FFNCalibrator.collect_cross_moments` is the one exception to "per-neuron
only": a handful of specific *pairwise* moments `E[a_i * a_j]`, computed on
request for whichever pairs `ffn_selectors.TwinRedundancySelector`'s
least-squares merge scale needs (see `ffn_selectors._merge_scale`). It stays a
separate pass rather than a field on `CalibrationStats` because the full
`(neurons, neurons)` cross-moment matrix nothing else needs would be
`O(neurons^2)` memory for every calibration run, not just the ones asking
for it.
"""
from dataclasses import dataclass

import torch

from .hooks import flatten_tokens, run_hooked
from ..models.adapters import resolve_adapter


@dataclass(frozen=True)
class CalibrationStats:
    """Per-neuron activation moments over the calibration corpus.

    `mean` is the signed expectation `E[a_i]` -- the quantity bias
    compensation needs, because what pruning removes from the layer
    output is `W_out[:, i] * a_i` and its expectation is
    `W_out[:, i] * E[a_i]`. `mean_abs` and `rms` are unsigned magnitude
    measures for importance scoring, where a neuron that swings hard in
    both directions matters even though its signed mean is near zero.
    """

    mean: torch.Tensor
    mean_abs: torch.Tensor
    rms: torch.Tensor
    tokens: int

    @property
    def num_neurons(self) -> int:
        return self.mean.shape[0]

    def to(self, *args, **kwargs) -> "CalibrationStats":
        """Move/cast the three stat vectors, as `torch.Tensor.to` would."""
        return CalibrationStats(
            mean=self.mean.to(*args, **kwargs),
            mean_abs=self.mean_abs.to(*args, **kwargs),
            rms=self.rms.to(*args, **kwargs),
            tokens=self.tokens,
        )

    def select(self, indices) -> "CalibrationStats":
        """Restrict the stats to a subset of neurons, keeping `tokens`.

        Needed when stats collected before surgery are reused against the
        resized layer afterwards.
        """
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.mean.device)
        return CalibrationStats(
            mean=self.mean[idx],
            mean_abs=self.mean_abs[idx],
            rms=self.rms[idx],
            tokens=self.tokens,
        )


class FFNCalibrator:
    """Runs batches through a model and accumulates `CalibrationStats`
    for one FFN layer, or for several in a single pass.

    Takes an `FFNLayerAdapter` rather than reaching into a model layout
    directly, for the same reason every other consumer in this package
    does (see `adapters`): a new architecture is a new adapter, not
    an edit here. Without one, `adapters.resolve_adapter` picks it.
    """

    def __init__(self, model, adapter=None):
        self.model = model
        self.adapter = resolve_adapter(model, adapter)

    def collect(self, batches, layer_idx: int, max_batches: int = None) -> CalibrationStats:
        """Accumulate stats over `batches`.

        `batches` is any iterable of kwarg dicts ready to splat into the
        model (e.g. what a HuggingFace `DataCollatorWithPadding` yields).
        If a batch carries an `attention_mask`, padding positions are
        excluded -- counting them would drag every mean toward whatever
        the model happens to emit on `[PAD]`, which is an artifact of
        batching rather than a property of the data.
        """
        return self.collect_many(batches, [layer_idx], max_batches=max_batches)[layer_idx]

    def collect_many(self, batches, layer_indices=None, max_batches: int = None) -> dict:
        """`{layer_idx: CalibrationStats}` for several layers from *one*
        pass over `batches`.

        Calling `collect()` once per layer costs one full forward pass of
        the corpus per layer, although every pass computes every layer's
        activations anyway; hooking all of them at once gets the same
        numbers for the price of one. `layer_indices=None` means every
        layer (`adapter.num_layers()`).
        """
        layer_indices = self.adapter.resolve_layers(layer_indices)

        # float64 CPU accumulators: the per-batch reduction stays in the
        # model's dtype on-device (fast), only the running total is
        # promoted, so a long corpus can't drift the way a float32 (or,
        # worse, float16) running sum would.
        totals = {idx: {"sum": None, "abs_sum": None, "sq_sum": None, "tokens": 0}
                  for idx in layer_indices}
        state = {"mask": None}

        def make_hook(acc):
            def hook(_module, _inputs, output):
                acts = flatten_tokens(output, state["mask"]).float()
                if not acts.shape[0]:
                    return
                batch = (acts.sum(dim=0), acts.abs().sum(dim=0), acts.square().sum(dim=0))
                for key, value in zip(("sum", "abs_sum", "sq_sum"), batch):
                    value = value.double().cpu()
                    acc[key] = value if acc[key] is None else acc[key] + value
                acc["tokens"] += acts.shape[0]
            return hook

        registrations = [
            (self.adapter.get_activation_module(idx), make_hook(totals[idx]), False)
            for idx in layer_indices
        ]
        run_hooked(self.model, batches, registrations, state, max_batches=max_batches)

        stats = {}
        for idx in layer_indices:
            acc = totals[idx]
            n = acc["tokens"]
            if n == 0:
                raise ValueError("Calibration saw no tokens -- `batches` was empty or fully masked.")
            stats[idx] = CalibrationStats(
                mean=(acc["sum"] / n).float(),
                mean_abs=(acc["abs_sum"] / n).float(),
                rms=(acc["sq_sum"] / n).sqrt().float(),
                tokens=n,
            )
        return stats

    def collect_cross_moments(self, batches, layer_idx: int, pairs, max_batches: int = None) -> dict:
        """`{(i, j): E[a_i * a_j]}` for each `(i, j)` in `pairs`, `i < j`.

        `TwinRedundancySelector`'s least-squares merge scale needs this
        off-diagonal moment (see `ffn_selectors._merge_scale`); `collect()`
        alone only ever gives per-neuron moments (`mean`, `mean_abs`,
        `rms`), which is exactly the diagonal. A full `(neurons, neurons)`
        cross-moment matrix would answer this for free, but at
        `O(neurons^2)` memory for a stat almost nothing else needs -- this
        instead takes the specific pairs a merge-capable selector actually
        asked about (typically a handful of twin pairs, not every pair),
        which is why it is a separate pass from `collect()` rather than an
        unconditional extra field on every `CalibrationStats`.

        `pairs` is any iterable of `(i, j)` index pairs; order and
        duplicates in the input don't matter, the same pair is only ever
        computed once. Returns a plain `dict` (not a `CalibrationStats`)
        since there is no fixed-size per-neuron vector to hang it on.
        """
        unique_pairs = sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in pairs})
        if not unique_pairs:
            return {}
        left_idx = torch.tensor([p[0] for p in unique_pairs], dtype=torch.long)
        right_idx = torch.tensor([p[1] for p in unique_pairs], dtype=torch.long)

        totals = {"sum": None, "tokens": 0}
        state = {"mask": None}

        def hook(_module, _inputs, output):
            acts = flatten_tokens(output, state["mask"]).float()
            if not acts.shape[0]:
                return
            products = acts[:, left_idx.to(acts.device)] * acts[:, right_idx.to(acts.device)]
            batch_sum = products.sum(dim=0).double().cpu()
            totals["sum"] = batch_sum if totals["sum"] is None else totals["sum"] + batch_sum
            totals["tokens"] += acts.shape[0]

        module = self.adapter.get_activation_module(layer_idx)
        run_hooked(self.model, batches, [(module, hook, False)], state, max_batches=max_batches)

        if totals["tokens"] == 0:
            raise ValueError("Calibration saw no tokens -- `batches` was empty or fully masked.")

        cross = (totals["sum"] / totals["tokens"]).tolist()
        return {pair: value for pair, value in zip(unique_pairs, cross)}
