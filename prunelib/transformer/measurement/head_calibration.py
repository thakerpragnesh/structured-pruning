"""Data-driven attention-head statistics: the head counterpart of
`ffn_calibration.py`.

Weight-only head criteria (`head_analysis.ov_norms`, query similarity) see
what a head *could* write; they cannot see what it actually writes on real
inputs, which depends on its attention pattern. Two complementary
data-driven signals are collected here, both by hooking the attention
block's output projection -- whose input is exactly the heads'
concatenated context vectors `[ctx_1, ..., ctx_H]` -- so they need nothing
model-specific beyond `AttentionLayerAdapter.get_attention_heads`:

- `HeadCalibrator.collect` -> `HeadStats`: per-head moments of the
  contribution `W_O[:, h] @ ctx_h`. Streaming: per batch it keeps only a
  `(heads, head_dim)` sum and a `(heads, head_dim, head_dim)` second
  moment, from which `E||W_O[:, h] ctx_h||^2 = tr(W_O[:, h]^T W_O[:, h]
  E[ctx_h ctx_h^T])` follows exactly -- no per-token tensor is ever kept.
  `context_mean` is also what `AttentionSurgeon`'s bias compensation folds
  into the output bias.
- `HeadCalibrator.collect_gradient_importance`: Michel, Levy & Neubig,
  "Are Sixteen Heads Really Better than One?" (NeurIPS 2019) --
  `I_h = E_x |dL(x)/d xi_h|` for a gate `xi_h = 1` multiplying head `h`'s
  context. A loss-aware signal: it measures how much the *task* cares
  about a head, not how loud the head is. HuggingFace's `head_mask`
  argument, which the paper's reference code used to place `xi`, is gone
  from transformers 5.x, so the gate is applied by a forward pre-hook on
  the output projection instead -- the same place and the same arithmetic,
  without depending on any model's forward signature.
"""
from dataclasses import dataclass

import torch

from .hooks import flatten_tokens, run_hooked
from ..models.adapters import resolve_adapter


@dataclass(frozen=True)
class HeadStats:
    """Per-head moments of what each head writes, over the calibration corpus.

    `context_mean[h]` is `E[ctx_h]`, the head's mean context vector; its
    image `W_O[:, h] @ E[ctx_h]` is the head's mean contribution to the
    block output -- the constant bias compensation restores when the head
    is removed. `contribution_rms[h] = sqrt(E||W_O[:, h] ctx_h||^2)` is the
    damage of deleting the head outright; `contribution_std[h]` is the same
    with the mean subtracted, i.e. exactly the damage that remains *after*
    bias compensation. Rank by whichever matches how the cut will be made.
    """

    context_mean: torch.Tensor
    contribution_rms: torch.Tensor
    contribution_std: torch.Tensor
    tokens: int

    @property
    def num_heads(self) -> int:
        return self.contribution_rms.shape[0]

    @property
    def head_dim(self) -> int:
        return self.context_mean.shape[1]

    def to(self, *args, **kwargs) -> "HeadStats":
        """Move/cast the stat tensors, as `torch.Tensor.to` would."""
        return HeadStats(
            context_mean=self.context_mean.to(*args, **kwargs),
            contribution_rms=self.contribution_rms.to(*args, **kwargs),
            contribution_std=self.contribution_std.to(*args, **kwargs),
            tokens=self.tokens,
        )

    def select(self, heads) -> "HeadStats":
        """Restrict the stats to a subset of heads, keeping `tokens` --
        for reusing pre-surgery stats against the resized layer."""
        idx = torch.as_tensor(heads, dtype=torch.long, device=self.contribution_rms.device)
        return HeadStats(
            context_mean=self.context_mean[idx],
            contribution_rms=self.contribution_rms[idx],
            contribution_std=self.contribution_std[idx],
            tokens=self.tokens,
        )


class HeadCalibrator:
    """Runs batches through a model and measures each attention head.

    Takes an `AttentionLayerAdapter`; without one,
    `adapters.resolve_adapter` picks it from the model, like every
    other consumer in this package. `model` may be a task model wrapping
    the encoder (e.g. `BertForSequenceClassification`): the hooks land on
    the encoder's layers while batches run through the whole model, which
    is exactly what `collect_gradient_importance` needs for the task loss.
    """

    def __init__(self, model, adapter=None):
        self.model = model
        self.adapter = resolve_adapter(model, adapter)

    def collect(self, batches, layer_idx: int, max_batches: int = None) -> HeadStats:
        """`HeadStats` for one layer. Padding positions (per the batch's
        `attention_mask`) are excluded, as in `FFNCalibrator.collect`."""
        return self.collect_many(batches, [layer_idx], max_batches=max_batches)[layer_idx]

    def collect_many(self, batches, layer_indices=None, max_batches: int = None) -> dict:
        """`{layer_idx: HeadStats}` for several layers from one pass over
        `batches`. `layer_indices=None` means every layer."""
        layer_indices = self.adapter.resolve_layers(layer_indices)
        state = {"mask": None}
        totals, registrations, outputs = {}, [], {}

        for idx in layer_indices:
            num_heads = self.adapter.num_attention_heads(idx)
            output = self.adapter.get_attention_heads(idx)[3]
            outputs[idx] = (output, num_heads)
            totals[idx] = {"sum": None, "second": None, "tokens": 0}

            def hook(_module, args, acc=totals[idx], num_heads=num_heads):
                rows = flatten_tokens(args[0], state["mask"]).float()
                if not rows.shape[0]:
                    return
                ctx = rows.reshape(rows.shape[0], num_heads, -1)
                batch_sum = ctx.sum(dim=0).double().cpu()
                batch_second = torch.einsum("nhd,nhe->hde", ctx, ctx).double().cpu()
                if acc["sum"] is None:
                    acc["sum"], acc["second"] = batch_sum, batch_second
                else:
                    acc["sum"] += batch_sum
                    acc["second"] += batch_second
                acc["tokens"] += rows.shape[0]

            registrations.append((output, hook, True))

        run_hooked(self.model, batches, registrations, state, max_batches=max_batches)

        stats = {}
        for idx in layer_indices:
            acc = totals[idx]
            n = acc["tokens"]
            if n == 0:
                raise ValueError("Calibration saw no tokens -- `batches` was empty or fully masked.")
            output, num_heads = outputs[idx]
            mean = acc["sum"] / n                        # (H, d)
            second = acc["second"] / n                   # (H, d, d)
            w = output.weight.detach().double().cpu()    # (hidden_out, H * d)
            w = w.reshape(w.shape[0], num_heads, -1).permute(1, 0, 2)   # (H, hidden_out, d)
            gram = torch.einsum("hod,hoe->hde", w, w)    # W_O[:, h]^T W_O[:, h]
            mean_square = (gram * second).sum(dim=(1, 2))
            mean_part = torch.einsum("hd,hde,he->h", mean, gram, mean)
            stats[idx] = HeadStats(
                context_mean=mean.float(),
                contribution_rms=mean_square.clamp_min(0).sqrt().float(),
                # Cancellation can push the difference a hair below zero
                # for a head that writes a near-constant vector.
                contribution_std=(mean_square - mean_part).clamp_min(0).sqrt().float(),
                tokens=n,
            )
        return stats

    def collect_gradient_importance(self, batches, layer_indices=None, loss_fn=None,
                                    max_batches: int = None) -> dict:
        """`{layer_idx: (num_heads,) tensor}` of Michel et al.'s head
        importance `E_x |dL(x)/d xi_h|`, for every requested layer from one
        forward/backward per batch.

        The loss is `outputs.loss` by default, so batches must carry
        `labels` (a HuggingFace `*ForSequenceClassification` model then
        computes it); pass `loss_fn(outputs, batch) -> scalar` for anything
        else. The loss is assumed mean-reduced over the batch, as every
        HuggingFace head's is: each example's gate gradient is rescaled by
        the batch size to recover the per-example `dL(x)/d xi`, so batches
        of different sizes are weighted per example, not per batch.

        Gradients are taken with `torch.autograd.grad` with respect to the
        gates only, so no parameter's `.grad` is touched, and the model
        runs in eval mode (no dropout), as in the paper.
        """
        layer_indices = self.adapter.resolve_layers(layer_indices)
        state = {"mask": None, "gates": {}}
        totals = {idx: None for idx in layer_indices}
        examples = {"n": 0}
        registrations = []

        for idx in layer_indices:
            num_heads = self.adapter.num_attention_heads(idx)
            output = self.adapter.get_attention_heads(idx)[3]

            def gate_hook(_module, args, idx=idx, num_heads=num_heads):
                x = args[0]
                gate = torch.ones(x.shape[0], num_heads, device=x.device, dtype=x.dtype,
                                  requires_grad=True)
                state["gates"][idx] = gate
                head_dim = x.shape[-1] // num_heads
                expanded = gate.repeat_interleave(head_dim, dim=1)
                expanded = expanded.reshape(x.shape[0], *([1] * (x.dim() - 2)), x.shape[-1])
                return (x * expanded,) + tuple(args[1:])

            registrations.append((output, gate_hook, True))

        def step(batch):
            state["gates"].clear()
            with torch.enable_grad():
                outputs = self.model(**batch)
                loss = loss_fn(outputs, batch) if loss_fn is not None else _loss_of(outputs)
                missing = [idx for idx in layer_indices if idx not in state["gates"]]
                if missing:
                    raise ValueError(
                        f"The attention output projection of layer(s) {missing} never ran, so "
                        "their heads could not be gated. Check the adapter's "
                        "get_attention_heads() returns modules on the model's forward path."
                    )
                gates = [state["gates"][idx] for idx in layer_indices]
                grads = torch.autograd.grad(loss, gates, allow_unused=True)
            batch_size = gates[0].shape[0]
            for idx, gate, grad in zip(layer_indices, gates, grads):
                if grad is None:  # gate not on the loss path at all: importance 0
                    grad = torch.zeros_like(gate)
                contribution = (grad.detach().abs() * batch_size).sum(dim=0).double().cpu()
                totals[idx] = contribution if totals[idx] is None else totals[idx] + contribution
            examples["n"] += batch_size

        run_hooked(self.model, batches, registrations, state, max_batches=max_batches, step=step)

        if examples["n"] == 0:
            raise ValueError("Gradient importance saw no examples -- `batches` was empty.")
        return {idx: (totals[idx] / examples["n"]).float() for idx in layer_indices}


def _loss_of(outputs):
    loss = getattr(outputs, "loss", None)
    if loss is None and isinstance(outputs, dict):
        loss = outputs.get("loss")
    if loss is None:
        raise ValueError(
            "The model returned no `loss`. Gradient importance needs one: include `labels` in "
            "each batch (HuggingFace task models then compute it), or pass "
            "`loss_fn(outputs, batch) -> scalar`."
        )
    return loss
