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
each primitive builds it as a *copy* of the original with resized tensors
(`_resized_copy`): same class, kernel/stride/padding/dilation,
`padding_mode`, device, dtype, train/eval mode and `requires_grad`. Two
earlier versions each lost part of that. Before the primitives were
consolidated, every rebuilt module was created on the CPU in float32 with
`padding_mode='zeros'` -- pruning a CUDA model left CPU layers inside it,
and a `'reflect'`-padded conv silently became zero-padded. After that, they
were still constructed as a fresh `nn.Conv2d` / `nn.Linear`: a subclass
(a weight-standardized conv, say) silently became the plain base class and
computed something else, a frozen layer came back trainable, and a layer
of a model in eval mode came back in training mode.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune

from .indices import expand_blocks


def _resized_copy(module: nn.Module, tensors: dict[str, torch.Tensor | None], **attrs) -> nn.Module:
    """A copy of `module` -- same class, configuration, train/eval mode and
    `requires_grad` -- whose parameters/buffers named in `tensors` are the
    given (resized) tensors, and whose attributes in `attrs` (its channel
    counts) are updated. The original is left untouched.

    A masked module (`masking.mask_channels`, i.e. `torch.nn.utils.prune`'s
    reparametrization) comes back unmasked, its masked values baked in: the
    caller sliced `module.weight`, which already has the mask applied. A
    tensor that is computed from others -- weight norm, spectral norm, a
    `torch.nn.utils.parametrize` parametrization -- raises rather than be
    resized here and silently recomputed at its old size."""
    pruning_hooks = {k: h for k, h in module._forward_pre_hooks.items() if isinstance(h, torch_prune.BasePruningMethod)}
    masked = {h._tensor_name for h in pruning_hooks.values()}
    # A masked tensor the caller didn't resize keeps its (masked) value.
    tensors = {**tensors, **{name: getattr(module, name) for name in sorted(masked) if name not in tensors}}
    for name in tensors:
        if name not in module._parameters and name not in module._buffers and name not in masked:
            raise NotImplementedError(
                f"{type(module).__name__}.{name} is computed from other tensors (weight norm, spectral norm or a "
                f"parametrization), not stored, so it can't be resized here -- remove that reparametrization first"
            )

    # Copy everything but the tensors being replaced and the masking state
    # being dropped: mapping them to None in the memo skips copying them.
    skipped = [getattr(module, name) for name in tensors]
    skipped += [module._parameters[f"{name}_orig"] for name in masked] + [module._buffers[f"{name}_mask"] for name in masked]
    memo = {id(obj): None for obj in [*skipped, *pruning_hooks.values()] if obj is not None}
    new = copy.deepcopy(module, memo)
    for name in masked:
        del new._parameters[f"{name}_orig"], new._buffers[f"{name}_mask"]
        new.__dict__.pop(name, None)
    for key in pruning_hooks:
        del new._forward_pre_hooks[key]

    for name, value in tensors.items():
        value = None if value is None else value.detach().clone()
        if name in module._buffers:
            new._buffers[name] = value
        else:
            original = module._parameters.get(f"{name}_orig", module._parameters.get(name))
            setattr(new, name, None if value is None else nn.Parameter(value, requires_grad=original.requires_grad))
    for name, value in attrs.items():
        setattr(new, name, value)
    return new


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
    return _resized_copy(conv, {"weight": weight, "bias": bias}, in_channels=weight.shape[1], out_channels=weight.shape[0])


@torch.no_grad()
def slice_depthwise_conv2d(conv: nn.Conv2d, keep: torch.Tensor) -> nn.Conv2d:
    """New depthwise Conv2d keeping channels `keep` -- one index set for
    input and output alike (each filter owns exactly one channel), and
    `groups` shrinks to match."""
    if not is_depthwise_conv(conv):
        raise ValueError(f"expected a depthwise conv (groups == in == out > 1), got {conv}")
    weight, bias = _take(conv.weight, conv.bias, keep, None)
    n = weight.shape[0]
    return _resized_copy(conv, {"weight": weight, "bias": bias}, in_channels=n, out_channels=n, groups=n)


@torch.no_grad()
def slice_linear(
    linear: nn.Linear, keep_out: torch.Tensor | None = None, keep_in: torch.Tensor | None = None
) -> nn.Linear:
    """New Linear keeping output features `keep_out` (rows of weight and
    bias) and input features `keep_in` (columns of weight) of `linear`
    (`None` keeps all of that dimension)."""
    weight, bias = _take(linear.weight, linear.bias, keep_out, keep_in)
    return _resized_copy(linear, {"weight": weight, "bias": bias}, in_features=weight.shape[1], out_features=weight.shape[0])


@torch.no_grad()
def slice_batchnorm(bn: nn.modules.batchnorm._BatchNorm, keep: torch.Tensor) -> nn.modules.batchnorm._BatchNorm:
    """New BatchNorm of the same type keeping channels `keep`.

    Carrying running statistics across is a deliberate choice: the original
    omitted this, so the forward pass still ran but produced silently wrong
    activations after pruning (no error, just bad numbers). Copying them
    keeps the pruned model numerically sane before any fine-tuning happens.
    """
    keep = keep.to(torch.long)
    tensors = {}
    if bn.affine:
        tensors.update(weight=bn.weight.index_select(0, keep), bias=bn.bias.index_select(0, keep))
    if bn.track_running_stats:
        tensors.update(running_mean=bn.running_mean.index_select(0, keep), running_var=bn.running_var.index_select(0, keep))
    return _resized_copy(bn, tensors, num_features=keep.numel())  # num_batches_tracked is copied as-is


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
