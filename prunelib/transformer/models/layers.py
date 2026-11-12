"""Layer discovery, kept separate from scoring or pruning logic so that
changing what counts as prunable never touches how it's scored or
surgically resized.
"""
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class LayerHandle:
    name: str
    module: nn.Module
    kind: str  # "conv2d" or "linear"


@dataclass(frozen=True)
class LayerKind:
    """A prunable module type `discover_layers` should recognize.

    Passing new `LayerKind`s via `extra_kinds` is how a caller teaches
    `discover_layers` about a module type it doesn't already know (e.g.
    `LayerKind("layernorm", nn.LayerNorm)`) without editing this file --
    the built-in conv/linear handling stays untouched (Open/Closed).
    """

    name: str
    module_type: type
    name_filter: str = None


def discover_layers(model, include_conv=True, include_linear=False, linear_name_filter=None, extra_kinds=()):
    """linear_name_filter: only nn.Linear modules whose qualified name
    contains this substring are included (e.g. "intermediate" for BERT's
    FFN expansion layers). Pass None to include every nn.Linear.

    extra_kinds: additional `LayerKind`s to recognize, checked after the
    built-in conv/linear kinds.
    """
    kinds = []
    if include_conv:
        kinds.append(LayerKind("conv2d", nn.Conv2d))
    if include_linear:
        kinds.append(LayerKind("linear", nn.Linear, linear_name_filter))
    kinds.extend(extra_kinds)

    handles = []
    for name, module in model.named_modules():
        for kind in kinds:
            if isinstance(module, kind.module_type) and (kind.name_filter is None or kind.name_filter in name):
                handles.append(LayerHandle(name, module, kind.name))
                break
    return handles
