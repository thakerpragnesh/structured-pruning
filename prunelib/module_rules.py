"""
Per-module-type pruning rules: everything `graph.DependencyGraph` and
`group.PruningGroup` need to know about a layer type, behind one small
interface.

`graph.py` used to hard-code Conv2d/Linear/BatchNorm as three separate
`isinstance` chains -- one deciding how a prune propagates *through* a
module (`_visit`), one deciding where a skip connection's channels were
produced (`_find_producer`), one deciding how to rebuild it (`prune`). Supporting
another layer type meant editing all three consistently. Now each type
answers its questions once, in a `ModuleRule`:

- `role(module)`: how channel indices flow through it (`ChannelRole`).
- `rebuild(module, prune_out, prune_in)`: the smaller replacement module.
- `mask(module, prune_out)`: zero those output channels in place, for
  `PruningGroup.mask()`. Optional -- the default zeroes dim 0 of `weight`
  and `bias`, which fits every built-in type.

and the graph code depends only on those answers. The methods are
positional-only, so a rule may name its parameters for its own type
(`role(self, conv)`) and still accept every call the base does. Built-in rules cover
`nn.Conv2d` (ungrouped or depthwise), `nn.Linear`, `nn.BatchNorm1d` and
`nn.BatchNorm2d`; `register_module_rule(nn.Conv1d, MyConv1dRule())` adds a
type globally, or pass `module_rules={...}` to one `DependencyGraph`.
Lookup walks the MRO, so a subclass of a registered type uses its parent's
rule, exactly as the `isinstance` chains did.

A module with no rule is not necessarily unsupported: `DependencyGraph`
still lets indices pass through anything shape-preserving (activations,
pooling, dropout) and handles `nn.Flatten` itself -- a rule is only needed
for modules that *own per-channel weights* a prune has to resize.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Mapping

import torch
import torch.nn as nn

from .indices import complement_indices
from .masking import mask_channels
from .registry import Registry, resolve_by_type
from .surgery import is_depthwise_conv, slice_batchnorm, slice_conv2d, slice_depthwise_conv2d, slice_linear


class ChannelRole(Enum):
    MIXING = "mixing"
    """Each output channel combines every input channel (ungrouped Conv2d,
    Linear). A prune arriving from upstream shrinks only this module's
    *input* and stops here -- its outputs are a fresh set of channels. It
    is also where a set of channels is *produced*, so it can be the starting
    layer of a prune and the far end of a skip connection."""

    PER_CHANNEL = "per_channel"
    """Output channel `c` depends only on input channel `c`, through
    per-channel weights (BatchNorm, depthwise conv). A prune arriving from
    upstream shrinks this module and continues through it unchanged."""


class ModuleRule(ABC):
    """How one module type takes part in structured pruning. Index
    arguments are channels to *remove* (the convention `PruningGroup`
    stores), `None` meaning that dimension is untouched."""

    @abstractmethod
    def role(self, module: nn.Module, /) -> ChannelRole:
        """Raise `NotImplementedError` for a configuration of this type the
        rule can't handle (e.g. a grouped conv that isn't depthwise)."""

    @abstractmethod
    def rebuild(
        self, module: nn.Module, prune_out: torch.Tensor | None, prune_in: torch.Tensor | None, /
    ) -> nn.Module:
        """A new module with those output / input channels removed."""

    def mask(self, module: nn.Module, prune_out: torch.Tensor, /) -> None:
        """Zero output channels `prune_out` of `module` in place, shapes
        unchanged, so that a later `rebuild` gives the same model (phase 1
        of `PruningGroup.mask()`). The default masks dim 0 of `weight` and
        `bias` (`masking.mask_channels`). Override it for a type whose output
        channels live elsewhere, or raise `NotImplementedError` for one
        that can't be masked to zero."""
        mask_channels(module, prune_out)


def _keep(n: int, prune: torch.Tensor | None) -> torch.Tensor | None:
    return None if prune is None else complement_indices(n, prune)


def _per_channel_prune(prune_out: torch.Tensor | None, prune_in: torch.Tensor | None, module: nn.Module) -> torch.Tensor:
    """A PER_CHANNEL module has one channel dimension, so whichever side
    recorded a prune target is authoritative for both."""
    prune = prune_out if prune_out is not None else prune_in
    if prune is None:
        raise ValueError(f"{type(module).__name__} has neither an output nor an input prune target recorded")
    return prune


class Conv2dRule(ModuleRule):
    def role(self, conv: nn.Conv2d) -> ChannelRole:
        if conv.groups == 1:
            return ChannelRole.MIXING
        if is_depthwise_conv(conv):
            return ChannelRole.PER_CHANNEL
        raise NotImplementedError(f"grouped conv (groups={conv.groups}) that isn't depthwise is not supported")

    def rebuild(self, conv, prune_out, prune_in):
        if self.role(conv) is ChannelRole.PER_CHANNEL:
            return slice_depthwise_conv2d(conv, complement_indices(conv.out_channels, _per_channel_prune(prune_out, prune_in, conv)))
        return slice_conv2d(conv, keep_out=_keep(conv.out_channels, prune_out), keep_in=_keep(conv.in_channels, prune_in))


class LinearRule(ModuleRule):
    def role(self, linear: nn.Linear) -> ChannelRole:
        return ChannelRole.MIXING

    def rebuild(self, linear, prune_out, prune_in):
        return slice_linear(linear, keep_out=_keep(linear.out_features, prune_out), keep_in=_keep(linear.in_features, prune_in))


class BatchNormRule(ModuleRule):
    def role(self, bn) -> ChannelRole:
        return ChannelRole.PER_CHANNEL

    def rebuild(self, bn, prune_out, prune_in):
        return slice_batchnorm(bn, complement_indices(bn.num_features, _per_channel_prune(prune_out, prune_in, bn)))

    def mask(self, bn, prune_out):
        if not bn.affine:
            raise NotImplementedError(
                f"{type(bn).__name__}(affine=False) can't be masked: it has no weight or bias to zero, and its "
                f"running mean turns a zeroed input channel into a non-zero constant -- call .prune() instead"
            )
        super().mask(bn, prune_out)


# Keyed by module class; a rule also covers that class's subclasses.
# `register_module_rule(nn.Conv1d, MyConv1dRule())` teaches every
# `DependencyGraph` a new type (re-registering one raises unless
# `overwrite=True`).
MODULE_RULES: Registry[ModuleRule] = Registry("module rule")
register_module_rule = MODULE_RULES.register

register_module_rule(nn.Conv2d, Conv2dRule())
register_module_rule(nn.Linear, LinearRule())
register_module_rule(nn.BatchNorm1d, BatchNormRule())
register_module_rule(nn.BatchNorm2d, BatchNormRule())


def find_module_rule(module: nn.Module, rules: Mapping[type, ModuleRule] | None = None) -> ModuleRule | None:
    """The rule for `module`'s type or nearest registered base class, from
    `rules` (default: the global `MODULE_RULES`), or None."""
    return resolve_by_type(MODULE_RULES.entries() if rules is None else rules, type(module))
