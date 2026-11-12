"""
VGG-specific structural surgery, built on the generic `prune_conv_bn`.

Pulled out of experiments/01_vgg_cifar10_sweep.py so the corrected legacy
pipeline (archive/legacy_pipeline/pipeline.py) and the experiment script
share one implementation instead of two copies drifting apart.

Only `build_vgg16` needs torchvision, and it imports it lazily: the
seam-level helpers below work on any `nn.Module` whose `.features` is a
Conv/BN/ReLU `nn.Sequential`, so they don't force an optional dependency
on callers who bring their own model.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.prune as prune

from .masking import commit_mask, mask_channels, surviving_channels
from .saliency import compute_score, keep_indices, prune_count, select_prune_indices
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


def build_vgg16(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Modern torchvision weights API (`weights=`), not the deprecated
    `pretrained=True/False` boolean removed in recent torchvision versions."""
    import torchvision  # optional dependency: `pip install structured-pruning[vision-experiments]`

    weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.vgg16(weights=weights)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def _require_prunable(pairs: list[tuple[int, int | None]], layer_position: int) -> None:
    """The last conv feeds `model.classifier[0]` rather than another Conv2d,
    so shrinking it would also require resizing the classifier -- which none
    of the helpers here do."""
    if not 0 <= layer_position < len(pairs) - 1:
        raise ValueError(
            f"layer_position {layer_position} is the last conv layer (or out of range); "
            "pruning it requires also resizing model.classifier[0].in_features, "
            "which these helpers don't do. Use 0 <= layer_position < len(pairs) - 1."
        )


def _shrink_seam(features: nn.Sequential, pairs: list[tuple[int, int | None]], position: int, keep_idx) -> None:
    """Replace conv `position` (plus its BN and the next conv's input side)
    in `features` with physically smaller copies keeping only `keep_idx`."""
    conv_idx, bn_idx = pairs[position]
    next_conv_idx, _ = pairs[position + 1]
    bn = features[bn_idx] if bn_idx is not None else None

    new_conv, new_bn, new_next_conv = prune_conv_bn(
        features[conv_idx], keep_idx, bn=bn, next_conv=features[next_conv_idx]
    )
    features[conv_idx] = new_conv
    if bn_idx is not None:
        features[bn_idx] = new_bn
    features[next_conv_idx] = new_next_conv


def mask_vgg_layer(model: nn.Module, layer_position: int, prune_fraction: float, method: str = "max_k") -> int:
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

    Returns the number of newly-masked channels (0 if this layer has no
    survivors left to prune).
    """
    pairs = vgg_conv_bn_positions(model.features)
    _require_prunable(pairs, layer_position)
    conv = model.features[pairs[layer_position][0]]

    survivors = surviving_channels(conv.weight)
    if survivors.numel() == 0:
        return 0  # nothing left to prune in this layer

    scores = compute_score(conv.weight, method=method)
    prune_amount = prune_count(conv.weight.shape[0], prune_fraction)  # clamped to survivors by select_prune_indices
    new_prune_idx = survivors[select_prune_indices(scores[survivors], prune_amount)]

    mask_channels(conv, new_prune_idx)
    return new_prune_idx.numel()


def compress_masked_vgg(model: nn.Module) -> int:
    """Phase 2, called once after however many masking iterations you want.
    Commits every conv layer's mask (bakes zeros in permanently) and rebuilds
    the whole network with only the surviving ("unmasked") channels,
    chaining each layer's output-channel resize into the next layer's input
    -- this is the "create a compressed model and copy the unmask weights
    from original model" step.

    Layers that were never masked at all pass through with `keep_idx` equal
    to every channel -- a no-op resize for that layer, but still needed so
    that a *previous* layer's shrunk output gets propagated into this
    layer's input count. Returns the total number of channels removed across
    the whole network.
    """
    pairs = vgg_conv_bn_positions(model.features)
    total_removed = 0
    for pos in range(len(pairs) - 1):
        conv_idx, _ = pairs[pos]
        conv = model.features[conv_idx]

        commit_mask(conv)
        if prune.is_pruned(conv):  # pragma: no cover -- would indicate a bug in commit_mask
            raise RuntimeError(f"conv at features[{conv_idx}] is still masked after commit_mask")

        keep_idx = surviving_channels(conv.weight)
        total_removed += conv.out_channels - keep_idx.numel()
        _shrink_seam(model.features, pairs, pos, keep_idx)

    return total_removed


def prune_vgg_layer(model: nn.Module, layer_position: int, prune_fraction: float, method: str = "max_k") -> int:
    """Prune the `layer_position`-th conv layer of `model.features` (0-indexed
    among conv layers). Surgery happens in place: `prune_conv_bn` returns
    already-correctly-sized modules, substituted back into the Sequential.

    Does not support pruning the last conv layer (feeds `model.classifier[0]`
    rather than another Conv2d). Returns the number of channels kept.
    """
    pairs = vgg_conv_bn_positions(model.features)
    _require_prunable(pairs, layer_position)
    conv = model.features[pairs[layer_position][0]]

    scores = compute_score(conv.weight, method=method)
    keep = keep_indices(scores, prune_count(conv.out_channels, prune_fraction))
    _shrink_seam(model.features, pairs, layer_position, keep)
    return keep.numel()
