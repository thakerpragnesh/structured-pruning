"""
Two-phase pruning: mask first, compress second.

This is the workflow the original codebase was trying to implement (score a
channel, mask it, eventually copy the "unmasked" survivors into a smaller
model) but never got working end to end -- see GITHUB_AUDIT.md section 11
for the full list of what actually went wrong (a `layer_number` tracking
variable that was supposed to tell each `compute_mask` callback which layer
it was operating on, but the line that updated it was commented out in all
six driver scripts; a mask-assignment line commented out in the one file
that got furthest; etc).

The root design flaw in the old approach was implementing masking via a
subclass of `torch.nn.utils.prune.BasePruningMethod` whose `compute_mask`
had to reconstruct, from scratch and from global state, a full mask for the
whole tensor on every call. That's exactly the kind of place a bookkeeping
bug hides. `torch.nn.utils.prune.custom_from_mask` sidesteps this entirely:
you compute the *whole* mask once, up front, using the already-tested
saliency/selection code in `saliency.py`, and hand it to PyTorch's own
pruning reparametrization. There's no per-call global state for a
`layer_number`-style bug to live in.

Phase 1 -- mask (`mask_channels`): zero out the weakest channels of a layer
via reparametrization. The original weights are preserved in
`<name>_orig`; nothing is resized, nothing is destroyed. Safe to fine-tune
or evaluate with masks applied, across as many layers and iterations as you
want, before committing to anything.

Phase 2 -- compress (`compress_masked_conv_bn`): once you're done pruning
and ready to commit, `commit_mask` bakes the zeros permanently into the real
weight tensor, and `compress_masked_conv_bn` reads back which channels are
now all-zero (masked out) versus everything else (the "unmasked" survivors),
and calls `surgery.prune_conv_bn` to build the actual smaller model,
copying only the unmasked channels across.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .surgery import prune_conv_bn


def build_channel_mask(weight: torch.Tensor, prune_idx: torch.Tensor) -> torch.Tensor:
    """Full-shape mask for `weight` (any shape whose dim 0 is the output/row
    axis -- Conv2d's [out,in,kh,kw] or Linear's [out,in]) that zeroes every
    channel/row listed in `prune_idx` and keeps everything else at 1."""
    mask = torch.ones_like(weight)
    if prune_idx.numel() > 0:
        mask[prune_idx.to(torch.long)] = 0
    return mask


def mask_channels(module: nn.Module, prune_idx: torch.Tensor, name: str = "weight") -> None:
    """Phase 1. Zero the given output channels of `module`'s `name` parameter
    via PyTorch's pruning reparametrization -- non-destructive, reversible,
    safe to call once per layer per pruning iteration. If `module` is
    already masked (e.g. a second pruning iteration on the same layer),
    this composes with the existing mask rather than replacing it, matching
    `torch.nn.utils.prune`'s normal iterative-pruning behavior.

    If the module has a bias, it is masked too (same channel indices) --
    otherwise a "pruned" channel would still emit a constant bias value
    downstream, which isn't what pruning that channel is supposed to mean,
    and wouldn't match what happens once the channel is physically removed
    during compression.
    """
    weight = getattr(module, name)
    mask = build_channel_mask(weight, prune_idx)
    prune.custom_from_mask(module, name=name, mask=mask)

    if name == "weight" and getattr(module, "bias", None) is not None:
        bias_mask = torch.ones_like(module.bias)
        if prune_idx.numel() > 0:
            bias_mask[prune_idx.to(torch.long)] = 0
        prune.custom_from_mask(module, name="bias", mask=bias_mask)


def commit_mask(module: nn.Module, name: str = "weight") -> None:
    """Bake a module's mask(s) permanently into its parameter tensors (the
    reparametrization is removed; the zeros become real zeros). Commits both
    `name` and `bias` if each is currently masked; no-op for either that
    isn't."""
    for pname in (name, "bias"):
        if hasattr(module, f"{pname}_orig"):
            prune.remove(module, pname)


def zeroed_channels(weight: torch.Tensor, atol: float = 0.0) -> torch.Tensor:
    """Indices of output channels that are entirely zero. Meaningful once a
    mask has been committed (see `commit_mask`) -- everything *not* in this
    list is an "unmasked" survivor."""
    flat = weight.reshape(weight.shape[0], -1)
    is_zero = flat.abs().sum(dim=1) <= atol
    return is_zero.nonzero(as_tuple=True)[0]


def surviving_channels(weight: torch.Tensor, atol: float = 0.0) -> torch.Tensor:
    """Complement of `zeroed_channels`: the "unmasked" channels to copy
    across during compression."""
    n = weight.shape[0]
    zero_idx = set(zeroed_channels(weight, atol=atol).tolist())
    return torch.tensor([i for i in range(n) if i not in zero_idx], dtype=torch.long)


def compress_masked_conv_bn(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d | None = None,
    next_conv: nn.Conv2d | None = None,
    atol: float = 0.0,
):
    """Phase 2 for one layer. `conv`'s mask must already be committed (see
    `commit_mask`) -- this reads back which output channels ended up
    all-zero and builds the smaller replacement from the survivors, via
    `surgery.prune_conv_bn` (the same tested, index-based copy used
    elsewhere in this library -- no new copy logic to get wrong here).

    Returns (new_conv, new_bn, new_next_conv, keep_idx).
    """
    if prune.is_pruned(conv):
        raise ValueError(
            "conv is still masked (reparametrized) -- call commit_mask(conv) "
            "before compressing, or the 'zero channel' check below would be "
            "reading conv.weight_orig's un-masked values instead of the "
            "actual masked-and-committed weights."
        )
    keep_idx = surviving_channels(conv.weight, atol=atol)
    new_conv, new_bn, new_next_conv = prune_conv_bn(conv, keep_idx, bn=bn, next_conv=next_conv)
    return new_conv, new_bn, new_next_conv, keep_idx
