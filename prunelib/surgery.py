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

Everything here is built from four primitives -- `slice_conv2d`,
`slice_depthwise_conv2d`, `slice_linear`, `slice_batchnorm` -- each "keep
these output rows / input columns of one module, return a new module". They
are the only code in `prunelib` that constructs a pruned replacement module:
`graph.py`'s per-module-type rules (`module_rules.py`) and `vgg.py`'s
classifier resize call them too, where each used to carry its own copy of
the same index-and-copy logic (and of BatchNorm's running-statistics copy).

A replacement must be a drop-in substitute for the module it replaces, so
the primitives carry over everything about the original except its channel
counts: kernel/stride/padding/dilation, `padding_mode`, and the device and
dtype of its parameters. (Before they were consolidated, every rebuilt
module was created on the CPU in float32 with `padding_mode='zeros'`
regardless of the original -- pruning a CUDA model left CPU layers inside
it, and a `'reflect'`-padded conv silently became zero-padded.)
"""
from __future__ import annotations

import itertools

import torch
import torch.nn as nn

from .indices import expand_blocks


def _factory_kwargs(module: nn.Module) -> dict:
    """`device=`/`dtype=` matching `module`'s own floating-point tensors, so
    a replacement is constructed where the original lives."""
    tensors = itertools.chain(module.parameters(recurse=False), module.buffers(recurse=False))
    ref = next((t for t in tensors if t.is_floating_point()), None)
    return {} if ref is None else {"device": ref.device, "dtype": ref.dtype}


def _take(weight: torch.Tensor, bias: torch.Tensor | None, keep_out, keep_in):
    """Index `weight`'s output rows (dim 0, and `bias` with them) and input
    columns (dim 1). `None` keeps that whole dimension."""
    if keep_out is not None:
        keep_out = keep_out.to(torch.long)
        weight = weight.index_select(0, keep_out)
        if bias is not None:
            bias = bias.index_select(0, keep_out)
    if keep_in is not None:
        weight = weight.index_select(1, keep_in.to(torch.long))
    return weight, bias


def _copy_into(new: nn.Module, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    new.weight.copy_(weight)
    if bias is not None:
        new.bias.copy_(bias)


def _conv_like(conv: nn.Conv2d, in_channels: int, out_channels: int, groups: int) -> nn.Conv2d:
    """A Conv2d configured exactly like `conv` except for its channel counts."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=groups, bias=conv.bias is not None, padding_mode=conv.padding_mode,
        **_factory_kwargs(conv),
    )


def is_depthwise_conv(module: nn.Module) -> bool:
    """One filter per channel: `groups == in_channels == out_channels > 1`.
    Input and output channels are then the same dimension."""
    return (
        isinstance(module, nn.Conv2d)
        and module.groups > 1
        and module.groups == module.in_channels == module.out_channels
    )


@torch.no_grad()
def slice_conv2d(
    conv: nn.Conv2d, keep_out: torch.Tensor | None = None, keep_in: torch.Tensor | None = None
) -> nn.Conv2d:
    """New ungrouped Conv2d keeping output channels `keep_out` and input
    channels `keep_in` of `conv` (`None` keeps all of that dimension).

    Raises for a grouped conv: its weight's dim 1 is `in_channels / groups`,
    not `in_channels`, so neither index means what the caller asked for.
    (`prune_conv_bn` used to build `groups=1` regardless -- a depthwise
    conv's `[C, 1, kh, kw]` filters were then silently broadcast across
    every input channel.) Use `slice_depthwise_conv2d` for depthwise.
    """
    if conv.groups != 1:
        raise ValueError(
            f"slice_conv2d handles ungrouped convs (groups=1) only, got groups={conv.groups}"
            + ("; use slice_depthwise_conv2d for a depthwise conv" if is_depthwise_conv(conv) else "")
        )
    weight, bias = _take(conv.weight, conv.bias, keep_out, keep_in)
    new = _conv_like(conv, weight.shape[1], weight.shape[0], groups=1)
    _copy_into(new, weight, bias)
    return new


@torch.no_grad()
def slice_depthwise_conv2d(conv: nn.Conv2d, keep: torch.Tensor) -> nn.Conv2d:
    """New depthwise Conv2d keeping channels `keep` -- one index set for
    input and output alike (each filter owns exactly one channel), and
    `groups` shrinks to match."""
    if not is_depthwise_conv(conv):
        raise ValueError(f"expected a depthwise conv (groups == in == out > 1), got {conv}")
    weight, bias = _take(conv.weight, conv.bias, keep, None)
    n = weight.shape[0]
    new = _conv_like(conv, n, n, groups=n)
    _copy_into(new, weight, bias)
    return new


@torch.no_grad()
def slice_linear(
    linear: nn.Linear, keep_out: torch.Tensor | None = None, keep_in: torch.Tensor | None = None
) -> nn.Linear:
    """New Linear keeping output features `keep_out` (rows of weight and
    bias) and input features `keep_in` (columns of weight) of `linear`
    (`None` keeps all of that dimension)."""
    weight, bias = _take(linear.weight, linear.bias, keep_out, keep_in)
    new = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, **_factory_kwargs(linear))
    _copy_into(new, weight, bias)
    return new


@torch.no_grad()
def slice_batchnorm(bn: nn.modules.batchnorm._BatchNorm, keep: torch.Tensor) -> nn.modules.batchnorm._BatchNorm:
    """New BatchNorm of the same type keeping channels `keep`.

    Carrying running statistics across is a deliberate choice: the original
    omitted this, so the forward pass still ran but produced silently wrong
    activations after pruning (no error, just bad numbers). Copying them
    keeps the pruned model numerically sane before any fine-tuning happens.
    """
    keep = keep.to(torch.long)
    new_bn = type(bn)(
        keep.numel(), eps=bn.eps, momentum=bn.momentum, affine=bn.affine,
        track_running_stats=bn.track_running_stats, **_factory_kwargs(bn),
    )
    if bn.affine:
        new_bn.weight.copy_(bn.weight.index_select(0, keep))
        new_bn.bias.copy_(bn.bias.index_select(0, keep))
    if bn.track_running_stats:
        new_bn.running_mean.copy_(bn.running_mean.index_select(0, keep))
        new_bn.running_var.copy_(bn.running_var.index_select(0, keep))
        new_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new_bn


def prune_conv_bn(
    conv: nn.Conv2d,
    keep_out_idx: torch.Tensor,
    bn: nn.BatchNorm2d | None = None,
    next_conv: nn.Conv2d | None = None,
) -> tuple[nn.Conv2d, nn.BatchNorm2d | None, nn.Conv2d | None]:
    """Shrink a Conv2d's output channels to `keep_out_idx`, and propagate the
    new channel count through an optional following BatchNorm2d and an
    optional next Conv2d whose input channels must shrink to match.

    Returns new modules; the originals are left untouched. The seam is
    validated before anything is built.
    """
    if next_conv is not None and next_conv.in_channels != conv.out_channels:
        raise ValueError(
            f"seam mismatch: conv emits {conv.out_channels} channels, "
            f"next_conv expects {next_conv.in_channels}"
        )
    new_conv = slice_conv2d(conv, keep_out=keep_out_idx)
    new_bn = slice_batchnorm(bn, keep_out_idx) if bn is not None else None
    new_next_conv = slice_conv2d(next_conv, keep_in=keep_out_idx) if next_conv is not None else None
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

    keep_rows = expand_blocks(keep_heads, head_dim)

    new_query = slice_linear(query, keep_out=keep_rows)
    new_key = slice_linear(key, keep_out=keep_rows)
    new_value = slice_linear(value, keep_out=keep_rows)
    new_output = slice_linear(output, keep_in=keep_rows)
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
    new_fc1 = slice_linear(fc1, keep_out=keep_idx)
    new_fc2 = slice_linear(fc2, keep_in=keep_idx)
    return new_fc1, new_fc2
