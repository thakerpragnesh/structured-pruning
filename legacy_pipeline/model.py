"""
Model loading for the corrected legacy pipeline.

Corrects `load_model.py`:
- `torchvision.models.vgg16(pretrained=True/False)` is the deprecated
  boolean API, removed in current torchvision. Uses `weights=` throughout.
- The original hand-rolled a full `VGG`/`make_layers`/`create_vgg_from_feature_list`
  reimplementation of `torchvision.models.VGG` so that pruning could change
  layer widths -- reasonable when pruning meant "rebuild a same-shaped
  model, then manually copy surviving weights in" (which is where D6/D7
  lived). Not needed here: `prunelib.prune_vgg_layer` mutates a *real*
  `torchvision.models.vgg16` instance's `.features` Sequential in place, so
  there's no separate model class to keep in sync with torchvision's own.
- `freeze`/`freeze_feature` in the original hardcoded parameter *counts*
  ("if count == 30: ...") that silently go stale if VGG's classifier head
  changes shape (e.g. a different `num_classes` changes `classifier[6]`'s
  parameter count but not its position). Reimplemented here by freezing
  everything except `model.classifier` (or a named subset of it) directly,
  which can't drift out of sync with the architecture the way a magic
  parameter-index count can.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from prunelib import build_vgg16

from .config import PruningConfig


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(cfg: PruningConfig, device: torch.device) -> nn.Module:
    if cfg.saved_model_path is not None:
        model = torch.load(cfg.saved_model_path, map_location=device, weights_only=False)
    else:
        model = build_vgg16(num_classes=cfg.num_classes, pretrained=cfg.load_pretrained)
    return model.to(device)


def freeze_all_but_classifier(model: nn.Module) -> None:
    """Freeze every parameter except the classifier head. Replaces the
    original's `freeze()`, which hardcoded a specific parameter *index*
    ("if count == 30: param.requires_grad = True") per VGG variant --
    correct only for exactly one `num_classes` value, and silently wrong
    (freezes everything, including the head) for any other, with no error
    to indicate it happened."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
