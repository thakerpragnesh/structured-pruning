"""
VGG-specific structural surgery, built on the generic `prune_conv_bn`.

Pulled out of experiments/01_vgg_cifar10_sweep.py so the corrected legacy
pipeline (archive/legacy_pipeline/pipeline.py) and the experiment script
share one implementation instead of two copies drifting apart.

Only `build_vgg16` needs torchvision, and it imports it itself: the surgery
functions here are plain torch over any torchvision-*shaped* VGG (a
`.features` Sequential of Conv2d/BatchNorm2d, a `.classifier` whose first
layer is a Linear), so they work without the optional
`vision-experiments` dependency installed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .indices import complement_indices, expand_blocks
from .masking import commit_mask, mask_channels, surviving_channels
from .selection import select_prune_indices_by_method
from .surgery import prune_conv_bn, slice_linear


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
    return slice_linear(linear, keep_in=expand_blocks(keep_idx, spatial))


def _layer_indices(model: nn.Module, layer_position: int) -> tuple[int, int | None, int | None]:
    """`(conv_idx, bn_idx, next_conv_idx)` in `model.features` for the
    `layer_position`-th conv layer; `bn_idx` is None if it has no
    BatchNorm, `next_conv_idx` is None for the last conv layer (whose
    consumer is `classifier[0]`)."""
    pairs = vgg_conv_bn_positions(model.features)
    if not 0 <= layer_position < len(pairs):
        raise ValueError(f"layer_position {layer_position} out of range: model has {len(pairs)} conv layers")
    conv_idx, bn_idx = pairs[layer_position]
    next_conv_idx = pairs[layer_position + 1][0] if layer_position + 1 < len(pairs) else None
    return conv_idx, bn_idx, next_conv_idx


def _apply_vgg_surgery(model: nn.Module, layer_position: int, keep_idx: torch.Tensor) -> None:
    """Shrink conv layer `layer_position` to `keep_idx` in place, along with
    its BatchNorm (if any) and whatever consumes its output: the next conv,
    or `classifier[0]` for the last conv layer."""
    conv_idx, bn_idx, next_conv_idx = _layer_indices(model, layer_position)
    conv = model.features[conv_idx]
    bn = model.features[bn_idx] if bn_idx is not None else None
    next_conv = model.features[next_conv_idx] if next_conv_idx is not None else None

    new_conv, new_bn, new_next_conv = prune_conv_bn(conv, keep_idx, bn=bn, next_conv=next_conv)
    model.features[conv_idx] = new_conv
    if bn_idx is not None:
        model.features[bn_idx] = new_bn
    if next_conv_idx is None:
        model.classifier[0] = _shrink_classifier_input(model.classifier[0], keep_idx, conv.out_channels)
    else:
        model.features[next_conv_idx] = new_next_conv


def build_vgg16(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Modern torchvision weights API (`weights=`), not the deprecated
    `pretrained=True/False` boolean removed in recent torchvision versions.
    Needs torchvision (`pip install structured-pruning[vision-experiments]`)."""
    import torchvision

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

    `method` is any rule `selection.select_prune_indices_by_method` accepts
    (`"max_k"`, `"l1"`, `"l2"`, `"random"`, `"kmeans"`, anything registered
    later, or a selector callable); `method_kwargs`
    are forwarded to it (e.g. `metric="manhattan"` for `"kmeans"`). For
    `"kmeans"`, clustering runs over the surviving channels only, for the
    same reason scoring does.

    Every conv layer can be masked, including the last one (its consumer,
    `classifier[0]`, is resized by `compress_masked_vgg`).

    Returns the number of newly-masked channels (0 if this layer has no
    survivors left to prune).
    """
    conv = model.features[_layer_indices(model, layer_position)[0]]
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
    conv = model.features[_layer_indices(model, layer_position)[0]]

    n_out = conv.out_channels
    prune_amount = min(max(1, int(round(n_out * prune_fraction))), n_out - 1)
    prune_idx = select_prune_indices_by_method(conv.weight, prune_amount, method=method, **method_kwargs)
    keep = complement_indices(n_out, prune_idx)

    _apply_vgg_surgery(model, layer_position, keep)
    return len(keep)
