"""
Channel/neuron saliency scoring for structured pruning.

Reference implementation of the Max-k criterion from:
  Thaker & Mohan, "Channel Pruning of Transfer Learning Models Using Novel
  Techniques," IEEE Access, vol. 12, pp. 94914-94925, 2024.

Design note (fixes a real bug found in the original pruning_framwork_v4 repo,
tracked as D1 in GITHUB_AUDIT.md): that code computed the Max-3 score into one
tensor (`channel_norm`) but selected channels to prune by comparing a *different*
tensor (`channel_norm_temp`, which was never overwritten and still held the plain
L1 norm). The score was computed, printed, and then silently ignored.

Here there is exactly one function that produces a score, and exactly one
function that selects indices from a score. Nothing else touches the tensor in
between, so the two cannot drift apart the way they did before.

Every scorer accepts either a Conv2d weight `[out_channels, in_channels, kh,
kw]` or a Linear weight `[out_features, in_features]` -- a Linear layer's
weight *is* a Conv2d weight with a 1x1 kernel: each output feature's row is
its entire "kernel" over the input features, with no spatial extent to
flatten. `_channel_view` below reshapes either to a common `(out, in,
spatial)` so one code path covers both, instead of every caller reshaping a
Linear weight to a fake `[out, in, 1, 1]` conv tensor to reuse this module (as
`experiments/02_bert_sst2_sweep.py` used to -- see its `_score_ffn_neurons`
before this change).
"""
from __future__ import annotations

import torch


def _channel_view(weight: torch.Tensor) -> torch.Tensor:
    """Reshape a weight to `(out_channels, in_channels, spatial)`.

    A 2D Linear weight `(out, in)` becomes `(out, in, 1)`: one "spatial"
    position, matching a 1x1 kernel. A 4D Conv2d weight `(out, in, kh, kw)`
    becomes `(out, in, kh * kw)`. Every scorer below reduces over the last
    axis (and sums over the input-channel axis), so this is the only place
    that needs to know about the shape difference between the two layer
    types.
    """
    if weight.dim() == 2:
        return weight.unsqueeze(-1)
    if weight.dim() == 4:
        out_ch, in_ch, kh, kw = weight.shape
        return weight.reshape(out_ch, in_ch, kh * kw)
    raise ValueError(
        f"expected a 2D Linear weight [out_features, in_features] or a 4D Conv2d "
        f"weight [out_channels, in_channels, kh, kw], got shape {tuple(weight.shape)}"
    )


def max_k_saliency(weight: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Max-k saliency score per output channel/neuron.

    For each kernel (one per input channel/feature of a given output unit),
    take the sum of the ``k`` largest-magnitude entries. Sum those kernel
    scores across all input channels/features to get one score per output
    unit.

    weight: Conv2d weight `[out_channels, in_channels, kh, kw]`, or Linear
        weight `[out_features, in_features]`
    returns: [out_channels] (or [out_features]) tensor, higher = more
        important (keep it)

    This replaces a hand-rolled "track the top 3 values seen so far" loop
    (D2: the rotation logic only fired on a *new* maximum, so a value landing
    between the current 2nd and 1st place was silently dropped) with
    ``torch.topk``, which is correct by construction for any k.

    It also fixes D3: the original accumulated `max1+max2+max3` into the
    output-channel's running total *inside* the input-channel loop using
    variables that were only reset once per output channel, so the sum grew
    from monotonically-increasing partial maxima rather than being a clean
    sum-of-per-kernel-scores. Here each input channel's top-k sum is computed
    independently (`topk_vals.sum(dim=2)`) and summed across the input-channel
    axis exactly once (`.sum(dim=1)`).

    And D4: the original iterated kernel columns with `range(size[2])` for
    both height and width, silently dropping columns on non-square kernels.
    Here `kh` and `kw` are read from their own tensor dimensions and the
    kernel is flattened, so there is no way to conflate the two.
    """
    flat = _channel_view(weight.detach()).abs()
    k_eff = min(k, flat.shape[-1])
    topk_vals, _ = flat.topk(k_eff, dim=2)      # [out, in, k_eff]
    kernel_scores = topk_vals.sum(dim=2)        # [out, in]  top-k sum per kernel
    channel_scores = kernel_scores.sum(dim=1)   # [out]      summed across input channels once
    return channel_scores


def l1_saliency(weight: torch.Tensor) -> torch.Tensor:
    """L1-norm saliency: sum of absolute weights per output channel/neuron."""
    return _channel_view(weight.detach()).abs().sum(dim=(1, 2))


def l2_saliency(weight: torch.Tensor) -> torch.Tensor:
    """L2-norm saliency: Euclidean norm of weights per output channel/neuron."""
    return _channel_view(weight.detach()).pow(2).sum(dim=(1, 2)).sqrt()


def random_saliency(weight: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random baseline score, one draw per output channel/neuron. Seed via `generator`."""
    out_ch = _channel_view(weight).shape[0]
    if generator is not None:
        return torch.rand(out_ch, generator=generator)
    return torch.rand(out_ch)


_METHODS = {
    "max_k": max_k_saliency,
    "l1": l1_saliency,
    "l2": l2_saliency,
    "random": random_saliency,
}


def compute_score(weight: torch.Tensor, method: str = "max_k", **kwargs) -> torch.Tensor:
    """Single entry point used by every experiment and by `select_prune_indices`.

    Because scoring always goes through here, there is no code path where a
    caller can compute one score and select against another (which is exactly
    how D1 happened in the original codebase).
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}, expected one of {list(_METHODS)}")
    return _METHODS[method](weight, **kwargs)


def select_prune_indices(scores: torch.Tensor, prune_amount: int) -> torch.Tensor:
    """Indices of the `prune_amount` lowest-scoring channels, ascending order.

    Takes a *score tensor*, not a weight tensor — the caller must have already
    produced it via `compute_score`, so there is no separate "recompute the
    norm to decide what to prune" step for the two to disagree over.
    """
    prune_amount = max(0, min(prune_amount, scores.numel()))
    if prune_amount == 0:
        return torch.empty(0, dtype=torch.long)
    idx = torch.topk(scores, prune_amount, largest=False).indices
    return idx.sort().values


def keep_indices(scores: torch.Tensor, prune_amount: int) -> torch.Tensor:
    """Complement of `select_prune_indices`: the channels to keep, ascending order."""
    prune_idx = select_prune_indices(scores, prune_amount)
    keep_mask = torch.ones(scores.numel(), dtype=torch.bool)
    keep_mask[prune_idx] = False
    return keep_mask.nonzero(as_tuple=True)[0]
