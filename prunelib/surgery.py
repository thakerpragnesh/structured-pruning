"""
Structural surgery: given indices of channels/neurons to keep, produce
genuinely smaller modules (fewer real parameters, less real compute) rather
than zero-masking the pruned ones.

Design note: the original `deep_model_copy_channelwise` (D6 in GITHUB_AUDIT.md)
had two independent bugs — a "keep this channel?" check that computed
`torch.norm(tensor != 0)`, i.e. the norm of a *boolean* tensor (almost always
non-zero, so the check was nearly always true), and a destination-index
counter that was never incremented, so every surviving channel got written to
index 0 and the rest of the new tensor stayed at its uninitialised value.

Here there is no manual loop over "should I keep this one" at all. Keeping is
decided once (by `prunelib.saliency`) and surgery is pure indexing:
`weight[keep_idx]`. Advanced indexing can't leave a counter behind to forget
to increment.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def prune_conv_bn(
    conv: nn.Conv2d,
    keep_out_idx: torch.Tensor,
    bn: nn.BatchNorm2d | None = None,
    next_conv: nn.Conv2d | None = None,
) -> tuple[nn.Conv2d, nn.BatchNorm2d | None, nn.Conv2d | None]:
    """Shrink a Conv2d's output channels to `keep_out_idx`, and propagate the
    new channel count through an optional following BatchNorm2d and an
    optional next Conv2d whose input channels must shrink to match.

    Returns new modules; the originals are left untouched.
    """
    keep_out_idx = keep_out_idx.to(torch.long)
    new_out = keep_out_idx.numel()

    new_conv = nn.Conv2d(
        conv.in_channels,
        new_out,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.index_select(0, keep_out_idx))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias.index_select(0, keep_out_idx))

    new_bn = None
    if bn is not None:
        # Carrying running statistics across is a deliberate choice: the
        # original omitted this, so the forward pass still ran but produced
        # silently wrong activations after pruning (no error, just bad
        # numbers). Copying them keeps the pruned model numerically sane
        # before any fine-tuning happens.
        new_bn = nn.BatchNorm2d(new_out, eps=bn.eps, momentum=bn.momentum, affine=bn.affine,
                                 track_running_stats=bn.track_running_stats)
        with torch.no_grad():
            if bn.affine:
                new_bn.weight.copy_(bn.weight.index_select(0, keep_out_idx))
                new_bn.bias.copy_(bn.bias.index_select(0, keep_out_idx))
            if bn.track_running_stats:
                new_bn.running_mean.copy_(bn.running_mean.index_select(0, keep_out_idx))
                new_bn.running_var.copy_(bn.running_var.index_select(0, keep_out_idx))
                new_bn.num_batches_tracked.copy_(bn.num_batches_tracked)

    new_next_conv = None
    if next_conv is not None:
        if next_conv.in_channels != conv.out_channels:
            raise ValueError(
                f"seam mismatch: conv emits {conv.out_channels} channels, "
                f"next_conv expects {next_conv.in_channels}"
            )
        new_next_conv = nn.Conv2d(
            new_out,
            next_conv.out_channels,
            kernel_size=next_conv.kernel_size,
            stride=next_conv.stride,
            padding=next_conv.padding,
            dilation=next_conv.dilation,
            groups=1,
            bias=next_conv.bias is not None,
        )
        with torch.no_grad():
            new_next_conv.weight.copy_(next_conv.weight.index_select(1, keep_out_idx))
            if next_conv.bias is not None:
                new_next_conv.bias.copy_(next_conv.bias)

    return new_conv, new_bn, new_next_conv


def prune_ffn_block(fc1: nn.Linear, fc2: nn.Linear, keep_idx: torch.Tensor) -> tuple[nn.Linear, nn.Linear]:
    """Shrink a Transformer FFN block (Linear -> activation -> Linear) by
    physically removing intermediate neurons at the positions *not* in
    `keep_idx`.

    Validates the seam and raises rather than silently producing
    incorrectly-shaped output — the original codebase had no equivalent check
    anywhere (D5), so a shape mismatch surfaced as a confusing runtime error
    deep inside a later matmul instead of at the point of the actual mistake.
    """
    if fc1.out_features != fc2.in_features:
        raise ValueError(
            f"seam mismatch before surgery: fc1 emits {fc1.out_features} features, "
            f"fc2 expects {fc2.in_features}"
        )
    keep_idx = keep_idx.to(torch.long)
    new_hidden = keep_idx.numel()

    new_fc1 = nn.Linear(fc1.in_features, new_hidden, bias=fc1.bias is not None)
    new_fc2 = nn.Linear(new_hidden, fc2.out_features, bias=fc2.bias is not None)
    with torch.no_grad():
        new_fc1.weight.copy_(fc1.weight.index_select(0, keep_idx))
        if fc1.bias is not None:
            new_fc1.bias.copy_(fc1.bias.index_select(0, keep_idx))
        new_fc2.weight.copy_(fc2.weight.index_select(1, keep_idx))
        if fc2.bias is not None:
            new_fc2.bias.copy_(fc2.bias)

    if new_fc1.out_features != new_fc2.in_features:  # pragma: no cover - should be unreachable
        raise RuntimeError("post-surgery seam mismatch — this indicates a bug in prune_ffn_block")
    return new_fc1, new_fc2
