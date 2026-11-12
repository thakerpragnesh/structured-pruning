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

`prune_attention_heads` (added 2026-09-21) extends this to multi-head
attention: `head_analysis`-style redundancy detection (`experiments/03`,
`pairwise_distance_matrix`) only ever *found* candidate heads, with nothing
in this module able to act on the result. See its docstring for why it
touches four projections (Q/K/V rows + output columns) instead of the two
`prune_ffn_block` needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _factory_kwargs(param: torch.Tensor) -> dict:
    """Build replacement modules on the same device/dtype as the original,
    so surgery on a GPU (or half-precision) model doesn't silently hand back
    CPU float32 layers."""
    return {"device": param.device, "dtype": param.dtype}


def _select(t: torch.Tensor, dim: int, idx: torch.Tensor | None) -> torch.Tensor:
    return t if idx is None else t.index_select(dim, idx.to(t.device))


def _slice_conv(
    conv: nn.Conv2d,
    keep_out: torch.Tensor | None = None,
    keep_in: torch.Tensor | None = None,
) -> nn.Conv2d:
    """New Conv2d keeping output channels `keep_out` and input channels
    `keep_in` (None = keep all along that axis). Bias follows the output
    axis only."""
    if conv.groups != 1:
        raise ValueError(
            f"grouped/depthwise convolutions (groups={conv.groups}) are not supported -- "
            "slicing channels independently would break the group partition"
        )
    out_ch = conv.out_channels if keep_out is None else keep_out.numel()
    in_ch = conv.in_channels if keep_in is None else keep_in.numel()
    new = nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        **_factory_kwargs(conv.weight),
    )
    with torch.no_grad():
        new.weight.copy_(_select(_select(conv.weight, 0, keep_out), 1, keep_in))
        if conv.bias is not None:
            new.bias.copy_(_select(conv.bias, 0, keep_out))
    return new


def _slice_bn(bn: nn.BatchNorm2d, keep: torch.Tensor) -> nn.BatchNorm2d:
    """New BatchNorm2d over channels `keep`, carrying affine params *and*
    running statistics.

    Carrying running statistics across is a deliberate choice: the original
    omitted this, so the forward pass still ran but produced silently wrong
    activations after pruning (no error, just bad numbers). Copying them
    keeps the pruned model numerically sane before any fine-tuning happens.
    """
    new = nn.BatchNorm2d(
        keep.numel(),
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
        **_factory_kwargs(bn.running_mean if bn.running_mean is not None else bn.weight),
    )
    with torch.no_grad():
        if bn.affine:
            new.weight.copy_(_select(bn.weight, 0, keep))
            new.bias.copy_(_select(bn.bias, 0, keep))
        if bn.track_running_stats:
            new.running_mean.copy_(_select(bn.running_mean, 0, keep))
            new.running_var.copy_(_select(bn.running_var, 0, keep))
            new.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new


def _slice_linear(
    linear: nn.Linear,
    keep_out: torch.Tensor | None = None,
    keep_in: torch.Tensor | None = None,
) -> nn.Linear:
    """New Linear keeping output rows `keep_out` and input columns `keep_in`
    (None = keep all along that axis). Bias follows the output axis only,
    so a layer shrunk on its input side keeps its bias untouched."""
    out_f = linear.out_features if keep_out is None else keep_out.numel()
    in_f = linear.in_features if keep_in is None else keep_in.numel()
    new = nn.Linear(in_f, out_f, bias=linear.bias is not None, **_factory_kwargs(linear.weight))
    with torch.no_grad():
        new.weight.copy_(_select(_select(linear.weight, 0, keep_out), 1, keep_in))
        if linear.bias is not None:
            new.bias.copy_(_select(linear.bias, 0, keep_out))
    return new


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
    if next_conv is not None and next_conv.in_channels != conv.out_channels:
        raise ValueError(
            f"seam mismatch: conv emits {conv.out_channels} channels, "
            f"next_conv expects {next_conv.in_channels}"
        )
    keep_out_idx = keep_out_idx.to(torch.long)

    new_conv = _slice_conv(conv, keep_out=keep_out_idx)
    new_bn = _slice_bn(bn, keep_out_idx) if bn is not None else None
    new_next_conv = _slice_conv(next_conv, keep_in=keep_out_idx) if next_conv is not None else None
    return new_conv, new_bn, new_next_conv


def prune_attention_heads(
    query: nn.Linear,
    key: nn.Linear,
    value: nn.Linear,
    output: nn.Linear,
    keep_heads: torch.Tensor,
    num_heads: int,
) -> tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]:
    """Shrink a multi-head attention block to just the heads in `keep_heads`.

    `query`/`key`/`value` are `[hidden, hidden]` Linear projections whose
    *output* rows partition into `num_heads` equal-size blocks -- head `h`
    owns rows `[h*head_dim, (h+1)*head_dim)`, HuggingFace's layout (e.g.
    `BertSelfAttention.query/key/value`). `output` is the block's output
    projection, whose *input* columns partition the same way, since its
    input is the heads' concatenated context vectors (HuggingFace:
    `BertSelfOutput.dense`). Removing a head therefore means dropping its
    rows from Q/K/V *and* its columns from `output` simultaneously -- never
    a partial row or column, or the surviving heads' math would be corrupted.

    Unlike `prune_ffn_block`, there is no merge or compensation option here:
    a dropped head's contribution is a function of the input (its attention
    pattern over the sequence), not a per-neuron constant a bias term can
    absorb. `output`'s bias is indexed by hidden size, not head, so -- like
    `prune_ffn_block`'s `fc2` bias -- it passes through untouched.
    """
    hidden = query.out_features
    if hidden % num_heads:
        raise ValueError(f"query has {hidden} output features, not divisible by num_heads {num_heads}")
    head_dim = hidden // num_heads

    if key.out_features != hidden or value.out_features != hidden:
        raise ValueError(
            f"seam mismatch before surgery: query emits {hidden} features, "
            f"key emits {key.out_features}, value emits {value.out_features}"
        )
    if output.in_features != hidden:
        raise ValueError(
            f"seam mismatch before surgery: query/key/value emit {hidden} features, "
            f"output projection expects {output.in_features}"
        )

    keep_heads = keep_heads.to(torch.long)
    out_of_range = keep_heads[(keep_heads < 0) | (keep_heads >= num_heads)]
    if out_of_range.numel():
        raise ValueError(
            f"keep_heads contains out-of-range head indices {out_of_range.tolist()} "
            f"for a block with {num_heads} heads"
        )
    if keep_heads.numel() == 0:
        raise ValueError("keep_heads is empty -- pruning every head would collapse the attention block")

    keep_rows = torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim) for h in keep_heads.tolist()])

    new_query = _slice_linear(query, keep_out=keep_rows)
    new_key = _slice_linear(key, keep_out=keep_rows)
    new_value = _slice_linear(value, keep_out=keep_rows)
    new_output = _slice_linear(output, keep_in=keep_rows)

    if new_output.in_features != new_query.out_features:  # pragma: no cover - should be unreachable
        raise RuntimeError("post-surgery seam mismatch — this indicates a bug in prune_attention_heads")
    return new_query, new_key, new_value, new_output


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

    new_fc1 = _slice_linear(fc1, keep_out=keep_idx)
    new_fc2 = _slice_linear(fc2, keep_in=keep_idx)

    if new_fc1.out_features != new_fc2.in_features:  # pragma: no cover - should be unreachable
        raise RuntimeError("post-surgery seam mismatch — this indicates a bug in prune_ffn_block")
    return new_fc1, new_fc2
