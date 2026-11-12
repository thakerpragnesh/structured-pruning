"""Physically resizes a paired (intermediate, output) FFN block.

This is the only place that knows how to reshape those weights. Every
pruning criterion -- Max-3 saliency, in/out norms, activation-aware
scoring, twin redundancy, or anything added later -- reduces to producing
a `Selection` and handing it to this one surgeon, so selection logic and
surgery mechanics never need to know about each other (Dependency
Inversion) and the mechanics themselves are written exactly once.

Beyond slicing, the surgeon can apply two corrections that make a cut
cost less accuracy than a naive delete:

- **Merging** -- fold a removed neuron's output column into a surviving
  one (`Selection.merge_map`), so a redundant neuron's contribution is
  transferred rather than dropped.
- **Bias compensation** -- add the removed neurons' *expected* output
  contribution back into the output bias. Deleting neuron `i` removes
  `W_out[:, i] * a_i` from the layer output; its expectation
  `W_out[:, i] * E[a_i]` is a constant, and a constant is exactly what a
  bias can represent exactly. So the layer's mean output is preserved for
  free, leaving only the zero-mean residual as real damage. This is
  ordinary practice in structured pruning and costs one calibration pass.
"""
import torch
import torch.nn as nn

from ..selection.context import Selection
from .linear_ops import rebuild_linear


class FFNSurgeon:
    def resize(self, intermediate: nn.Linear, output: nn.Linear, selection, stats=None,
               compensate_bias: bool = False):
        """Return a new `(intermediate, output)` pair with only the
        selected neurons.

        `selection` may be a `Selection` or a plain list of indices to
        keep. `stats` is a `ffn_calibration.CalibrationStats` for this layer,
        required only when `compensate_bias` is true or the selection
        carries merges whose scales were derived from calibration.
        """
        selection = self._checked_selection(intermediate, output, selection)
        num_neurons = intermediate.weight.shape[0]
        merge_map = selection.merge_map or {}

        out_weight = self._fold_merges(output.weight.data, merge_map)
        out_bias = None if output.bias is None else output.bias.data.clone()
        if compensate_bias:
            self._check_compensation(out_bias, stats, num_neurons)
            out_bias += self._bias_compensation(
                output.weight.data, stats, selection.keep_indices, merge_map, num_neurons
            ).to(out_bias.dtype)

        keep_in = torch.as_tensor(selection.keep_indices, dtype=torch.long,
                                  device=intermediate.weight.device)
        keep_out = keep_in.to(output.weight.device)
        new_intermediate = rebuild_linear(
            intermediate, intermediate.weight.data[keep_in],
            None if intermediate.bias is None else intermediate.bias.data[keep_in],
        )
        new_output = rebuild_linear(output, out_weight[:, keep_out], out_bias)
        return new_intermediate, new_output

    @staticmethod
    def _checked_selection(intermediate, output, selection) -> Selection:
        num_neurons = intermediate.weight.shape[0]
        if not isinstance(selection, Selection):
            selection = Selection.of(selection, num_neurons=num_neurons)
        if selection.keep_indices[-1] >= num_neurons:
            raise ValueError(
                f"Selection keeps neuron {selection.keep_indices[-1]} but the layer has only "
                f"{num_neurons} neurons."
            )
        if output.weight.shape[1] != num_neurons:
            raise ValueError(
                f"FFN pair is inconsistent: intermediate has {num_neurons} output neurons but "
                f"output projection expects {output.weight.shape[1]} inputs. Check the adapter's "
                "get_ffn() is returning a matching (intermediate, output) pair."
            )
        return selection

    @staticmethod
    def _fold_merges(source_out_weight, merge_map):
        """The output weight with every merged neuron's column folded into
        its survivor's, at full width (before slicing).

        Works from the original column values throughout, so two neurons
        merging into the same survivor, or a chain of merges, can't make the
        result depend on dict iteration order.
        """
        folded = source_out_weight.clone()
        for dropped, (survivor, scale) in merge_map.items():
            folded[:, survivor] += (
                source_out_weight[:, dropped].to(torch.float32) * scale
            ).to(folded.dtype)
        return folded

    @staticmethod
    def _check_compensation(out_bias, stats, num_neurons):
        if out_bias is None:
            raise ValueError(
                "compensate_bias=True but the output projection has no bias to fold the "
                "removed neurons' mean contribution into. Pass compensate_bias=False, or "
                "give the layer a bias."
            )
        if stats is None:
            raise ValueError(
                "compensate_bias=True requires calibration stats. Collect them with "
                "`ffn_calibration.FFNCalibrator(model).collect(batches, layer_idx)`."
            )
        if stats.num_neurons != num_neurons:
            raise ValueError(
                f"Calibration stats cover {stats.num_neurons} neurons but the layer has "
                f"{num_neurons}; they were collected against a different (or already "
                "pruned) layer."
            )

    @staticmethod
    def _bias_compensation(source_out_weight, stats, keep_indices, merge_map, num_neurons):
        """Mean output contribution lost by this cut.

        For a plainly-deleted neuron `j` that is `W_out[:, j] * E[a_j]`.
        For one merged into `i` with scale `s`, the merge already
        reintroduces `s * W_out[:, j] * a_i`, so only the *residual*
        `W_out[:, j] * (E[a_j] - s * E[a_i])` is still missing -- which
        is zero exactly when the scale was chosen as `E[a_j] / E[a_i]`.
        Handling both in one expression keeps merge and compensation from
        double-counting each other.
        """
        w = source_out_weight.to(torch.float32)
        mean = stats.mean.to(w.device, torch.float32)
        dropped = torch.ones(num_neurons, dtype=torch.bool, device=w.device)
        dropped[torch.as_tensor(keep_indices, dtype=torch.long, device=w.device)] = False

        residual = mean * dropped  # E[a_j] for dropped neurons, 0 for kept
        for j, (survivor, scale) in merge_map.items():
            residual[j] = mean[j] - scale * mean[survivor]
        return w @ residual
