"""
VGG-specific structural surgery, built on the generic `prune_conv_bn`.

Pulled out of experiments/01_vgg_cifar10_sweep.py so the corrected legacy
pipeline (archive/legacy_pipeline/pipeline.py) and the experiment script
share one implementation instead of two copies drifting apart.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision

from .clustering import select_prune_indices_by_method
from .masking import commit_mask, mask_channels, surviving_channels
from .surgery import prune_conv_bn


def vgg_conv_bn_positions(features: nn.Sequential) -> list[tuple[int, int | None]]:
    """List of (conv_idx, bn_idx_or_None) for every Conv2d in a torchvision
    VGG `.features` Sequential, in forward order."""
    pairs = []
    for i, layer in enumerate(features):
        if isinstance(layer, nn.Conv2d):
            bn_idx = i + 1 if i + 1 < len(features) and isinstance(features[i + 1], nn.BatchNorm2d) else None
            pairs.append((i, bn_idx))
    return pairs


def _shrink_classifier_input(linear: nn.Linear, keep_idx: torch.Tensor, n_channels: int) -> nn.Linear:
    """Resize VGG's `classifier[0]` to match a pruned last conv layer.

    torchvision's VGG flattens the `[N, C, 7, 7]` output of `avgpool` into
    `C * 49` features, channel-major, so channel `c` owns input columns
    `[c * spatial, (c + 1) * spatial)` of `classifier[0]`. Pruning channel
    `c` therefore means dropping that whole block of columns -- the same
    expansion `graph.DependencyGraph._handle_flatten` does for an arbitrary
    model, specialised here to VGG's fixed layout.
    """
    if linear.in_features % n_channels:
        raise ValueError(
            f"seam mismatch: classifier[0] expects {linear.in_features} features, not a "
            f"multiple of the last conv layer's {n_channels} channels"
        )
    spatial = linear.in_features // n_channels
    keep_idx = keep_idx.to(torch.long)
    cols = (keep_idx.unsqueeze(1) * spatial + torch.arange(spatial)).reshape(-1)

    new = nn.Linear(cols.numel(), linear.out_features, bias=linear.bias is not None)
    with torch.no_grad():
        new.weight.copy_(linear.weight.index_select(1, cols))
        if linear.bias is not None:
            new.bias.copy_(linear.bias)
    return new


def _apply_vgg_surgery(model: nn.Module, layer_position: int, keep_idx: torch.Tensor) -> None:
    """Shrink conv layer `layer_position` to `keep_idx` in place, along with
    its BatchNorm (if any) and whatever consumes its output: the next conv,
    or `classifier[0]` for the last conv layer."""
    pairs = vgg_conv_bn_positions(model.features)
    conv_idx, bn_idx = pairs[layer_position]
    conv = model.features[conv_idx]
    bn = model.features[bn_idx] if bn_idx is not None else None
    is_last = layer_position == len(pairs) - 1
    next_conv = None if is_last else model.features[pairs[layer_position + 1][0]]

    new_conv, new_bn, new_next_conv = prune_conv_bn(conv, keep_idx, bn=bn, next_conv=next_conv)
    model.features[conv_idx] = new_conv
    if bn_idx is not None:
        model.features[bn_idx] = new_bn
    if is_last:
        model.classifier[0] = _shrink_classifier_input(model.classifier[0], keep_idx, conv.out_channels)
    else:
        model.features[pairs[layer_position + 1][0]] = new_next_conv


def _check_layer_position(pairs: list, layer_position: int) -> None:
    if not 0 <= layer_position < len(pairs):
        raise ValueError(f"layer_position {layer_position} out of range: model has {len(pairs)} conv layers")


def build_vgg16(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Modern torchvision weights API (`weights=`), not the deprecated
    `pretrained=True/False` boolean removed in recent torchvision versions."""
    weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.vgg16(weights=weights)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def mask_vgg_layer(model: nn.Module, layer_position: int, prune_fraction: float, method: str = "max_k", **method_kwargs) -> int:
    """Phase 1 for one VGG conv layer, safe to call once per iteration across
    a multi-iteration schedule: extend this layer's channel mask by roughly
    `prune_fraction` of its *original* channel count, chosen from channels
    not already masked.

    Restricting the score-based selection to `surviving_channels` (not all
    channels) matters: without it, an already-masked channel's weight is
    zero, which is the lowest possible score under every criterion here, so
    it would keep winning re-selection on every later iteration. Two or
    three of the `prune_amount` slots each iteration would then be "spent"
    re-selecting channels that were already gone, and the pruning schedule
    would advance far slower than the requested `prune_fraction` per
    iteration actually implies. This is a correctness requirement of the
    mask-then-compress design, not an optimization.

    `method` is any rule `clustering.select_prune_indices_by_method` accepts
    (`"max_k"`, `"l1"`, `"l2"`, `"random"`, or `"kmeans"`); `method_kwargs`
    are forwarded to it (e.g. `metric="manhattan"` for `"kmeans"`). For
    `"kmeans"`, clustering runs over the surviving channels only, for the
    same reason scoring does.

    Every conv layer can be masked, including the last one (its consumer,
    `classifier[0]`, is resized by `compress_masked_vgg`).

    Returns the number of newly-masked channels (0 if this layer has no
    survivors left to prune).
    """
    pairs = vgg_conv_bn_positions(model.features)
    _check_layer_position(pairs, layer_position)
    conv_idx, _ = pairs[layer_position]
    conv = model.features[conv_idx]
    n_out_original = conv.weight.shape[0]

    survivors = surviving_channels(conv.weight)
    if survivors.numel() == 0:
        return 0  # nothing left to prune in this layer

    prune_amount = max(1, int(round(n_out_original * prune_fraction)))
    prune_amount = min(prune_amount, survivors.numel())

    # Select among survivors only. `conv.weight` is the masked weight here,
    # so indexing it by survivors gives exactly their (unmasked) values.
    survivor_weight = conv.weight.detach().index_select(0, survivors)
    newly_selected_positions = select_prune_indices_by_method(survivor_weight, prune_amount, method=method, **method_kwargs)
    new_prune_idx = survivors[newly_selected_positions]

    mask_channels(conv, new_prune_idx)
    return new_prune_idx.numel()


def compress_masked_vgg(model: nn.Module) -> int:
    """Phase 2, called once after however many masking iterations you want.
    Commits every conv layer's mask (bakes zeros in permanently) and rebuilds
    the whole network with only the surviving ("unmasked") channels,
    chaining each layer's output-channel resize into the next layer's input
    -- this is the "create a compressed model and copy the unmask weights
    from original model" step.

    The last conv layer's consumer is `classifier[0]` rather than another
    conv; its input columns are resized to match (see
    `_shrink_classifier_input`).

    Layers that were never masked at all pass through with `keep_idx` equal
    to every channel -- a no-op resize for that layer, but still needed so
    that a *previous* layer's shrunk output gets propagated into this
    layer's input count. Returns the total number of channels removed across
    the whole network.
    """
    pairs = vgg_conv_bn_positions(model.features)
    total_removed = 0
    for pos, (conv_idx, _) in enumerate(pairs):
        conv = model.features[conv_idx]

        commit_mask(conv)
        if prune.is_pruned(conv):  # pragma: no cover -- would indicate a bug in commit_mask
            raise RuntimeError(f"conv at features[{conv_idx}] is still masked after commit_mask")

        keep_idx = surviving_channels(conv.weight)
        total_removed += conv.out_channels - keep_idx.numel()
        _apply_vgg_surgery(model, pos, keep_idx)

    return total_removed


def prune_vgg_layer(model: nn.Module, layer_position: int, prune_fraction: float, method: str = "max_k", **method_kwargs) -> int:
    """Prune the `layer_position`-th conv layer of `model.features` (0-indexed
    among conv layers). Surgery happens in place: `prune_conv_bn` returns
    already-correctly-sized modules, substituted back into the Sequential.

    `method`/`method_kwargs`: as in `mask_vgg_layer` -- a saliency scorer or
    `"kmeans"` (cluster channels, prune the lowest-L1 ones within each
    cluster). Any conv layer can be pruned, including the last one, which
    also resizes `model.classifier[0]`'s input.

    Returns the number of channels kept.
    """
    pairs = vgg_conv_bn_positions(model.features)
    _check_layer_position(pairs, layer_position)
    conv = model.features[pairs[layer_position][0]]

    n_out = conv.out_channels
    prune_amount = min(max(1, int(round(n_out * prune_fraction))), n_out - 1)
    prune_idx = select_prune_indices_by_method(conv.weight, prune_amount, method=method, **method_kwargs)
    keep_mask = torch.ones(n_out, dtype=torch.bool)
    keep_mask[prune_idx] = False
    keep = keep_mask.nonzero(as_tuple=True)[0]

    _apply_vgg_surgery(model, layer_position, keep)
    return len(keep)
