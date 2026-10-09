"""
Applying a prune decision: `PruningGroup` is the set of modules one decision
touches -- which output and input channels each one loses -- plus the code
that rebuilds them. `graph.DependencyGraph` is what *finds* that set; this
module only acts on it and knows nothing about tracing or propagation. Each
targeted module is rebuilt through its `module_rules.ModuleRule`, or, for a
starting layer whose type is registered in `DependencyGraph`'s
`special_handlers`, through that handler.

One departure from the rest of `prunelib` worth flagging explicitly:
`prune_conv_bn` et al. always return *new* modules and never touch the ones
passed in. `PruningGroup.prune()` mutates the model in place instead. That's
not an oversight -- the entire point of dependency-graph pruning is that the
caller shouldn't need to know the graph shape well enough to wire new
modules back together themselves; something has to do that wiring, and once
it's been computed there's nothing left for the caller to reassemble.

`.mask()` / `.commit_and_compress()` are the two-phase mask-then-compress
workflow from `masking.py`, extended to a whole dependency group instead of
one layer at a time -- `.mask()` zeroes every targeted module's channels via
reparametrization (safe to fine-tune against, shapes unchanged),
`.commit_and_compress()` bakes the zeros in and runs the same rebuild
`.prune()` does. `.prune()` is the one-shot alternative.

This used to be the top half of `graph.py`. It is its own module because
changing how a decision is applied (a new rebuild path, masking) and
changing how one is found (tracing, propagation) are different reasons to
edit the code.
"""
from __future__ import annotations

from typing import Callable, Mapping

import torch
import torch.nn as nn

from .masking import commit_mask, mask_channels
from .module_rules import ModuleRule, find_module_rule

SpecialHandler = Callable[[nn.Module, torch.Tensor], nn.Module]


def _set_submodule(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    """`setattr` a module back onto its parent by dotted path. `nn.Module`
    has no `set_submodule` guaranteed at this package's `torch>=2.0` floor,
    so this is the hand-written equivalent of `get_submodule`."""
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _record_target(targets: dict[str, torch.Tensor], name: str, idx: torch.Tensor) -> bool:
    """Store `idx` under `name`; return True if this is the first time
    `name` has been targeted. A second, *different* index set for the same
    module is a real conflict (two branches disagreeing on what a shared
    layer's surviving channels are) and raises rather than silently keeping
    whichever arrived first."""
    existing = targets.get(name)
    if existing is None:
        targets[name] = idx
        return True
    if existing.numel() == idx.numel() and torch.equal(torch.sort(existing).values, torch.sort(idx).values):
        return False
    raise ValueError(
        f"{name!r} is already targeted for pruning with a different index set than this "
        f"path computed ({sorted(existing.tolist())} vs {sorted(idx.tolist())}) -- two "
        f"branches of the graph disagree on which channels of a shared layer survive"
    )


class PruningGroup:
    """The result of `DependencyGraph.get_pruning_group`: every module whose
    weights must be resized for one prune decision to be structurally valid,
    keyed by qualified name. `output_targets[name]` are indices to remove
    from that module's *output* dimension (dim 0 of its weight);
    `input_targets[name]` from its *input* dimension (dim 1). A module can
    appear in both (a plain conv-BN-conv chain has the middle conv shrink on
    input from the previous layer and needs no output change here; a
    depthwise conv appears only in `output_targets`, by convention, since
    its input and output channel counts are the same dimension).
    `special_targets[name]` is `(handler, keep_idx)` for a starting layer
    whose type was registered in `DependencyGraph`'s `special_handlers` --
    see `.prune()`'s handling of it below, and `DependencyGraph.
    get_pruning_group` for the handler's signature.

    Add targets through `add_output_target`/`add_input_target`, which
    detect two paths disagreeing about the same module, and
    `add_special_target`; `module_rules` (default: the global
    `module_rules.MODULE_RULES`) decides how each targeted module is
    rebuilt."""

    def __init__(self, model: nn.Module, module_rules: Mapping[type, ModuleRule] | None = None):
        self.model = model
        self.output_targets: dict[str, torch.Tensor] = {}
        self.input_targets: dict[str, torch.Tensor] = {}
        self.special_targets: dict[str, tuple[SpecialHandler, torch.Tensor]] = {}
        self.module_rules = module_rules

    def add_output_target(self, name: str, idx: torch.Tensor) -> bool:
        """Remove output channels `idx` of module `name`. Returns False if
        this exact target was already recorded; raises `ValueError` if a
        *different* one was."""
        return _record_target(self.output_targets, name, idx)

    def add_input_target(self, name: str, idx: torch.Tensor) -> bool:
        """Remove input channels `idx` of module `name` (see
        `add_output_target`)."""
        return _record_target(self.input_targets, name, idx)

    def add_special_target(self, name: str, handler: SpecialHandler, keep_idx: torch.Tensor) -> None:
        """Rebuild module `name` as `handler(module, keep_idx)` on `.prune()`,
        instead of through its module rule."""
        self.special_targets[name] = (handler, keep_idx)

    def prune(self) -> nn.Module:
        """Rebuild every targeted module with the recorded indices removed,
        and write each one back onto `self.model` in place. Returns the
        (mutated) model for convenience."""
        for name in dict.fromkeys([*self.output_targets, *self.input_targets]):
            module = self.model.get_submodule(name)
            rule = find_module_rule(module, self.module_rules)
            if rule is None:
                raise TypeError(f"{name!r}: don't know how to prune a {type(module).__name__}")
            new_module = rule.rebuild(module, self.output_targets.get(name), self.input_targets.get(name))
            _set_submodule(self.model, name, new_module)

        for name, (handler, keep_idx) in self.special_targets.items():
            module = self.model.get_submodule(name)
            new_module = handler(module, keep_idx)
            _set_submodule(self.model, name, new_module)

        return self.model

    def mask(self) -> None:
        """Phase 1 of the two-phase mask-then-compress workflow (see
        `masking.py`'s module docstring), extended here to a whole
        dependency group instead of one layer at a time: zero every
        *output*-target module's pruned channels via reparametrization
        (`masking.mask_channels`) rather than immediately rebuilding smaller
        modules. Shapes are unchanged, so the model keeps running -- gradients
        naturally don't reach the masked positions, so it's safe to fine-tune
        or evaluate with the mask active, across as many groups/iterations as
        you want, before calling `.commit_and_compress()` once ready to
        physically shrink everything this group touches.

        Only `output_targets` are masked. An `input_targets`-only module (a
        downstream conv/Linear whose *input* must shrink to match an
        upstream prune) needs no weight change here -- it already receives
        zero activations from the masked upstream output, so its own surgery
        can wait for `.commit_and_compress()`. Raises `NotImplementedError`
        for a group with `special_targets`: a handler rebuilds its module in
        one step, with nothing to reparametrize, so masking would leave that
        module unmasked while fine-tuning -- call `.prune()` instead.
        """
        if self.special_targets:
            raise NotImplementedError(
                f"{sorted(self.special_targets)} are rebuilt by a special handler, which can't be "
                f"masked first -- call .prune() on this group instead"
            )
        for name, idx in self.output_targets.items():
            if idx.numel() == 0:
                continue
            mask_channels(self.model.get_submodule(name), idx)

    def commit_and_compress(self) -> nn.Module:
        """Phase 2: bake every mask `.mask()` applied into real zeros
        (`masking.commit_mask`) on each `output_targets` module, then run
        the same rebuild `.prune()` does. Equivalent to calling `.prune()`
        directly on an unmasked group -- see
        `tests/test_masking.py::test_compress_masked_conv_bn_matches_direct_surgery`
        for the single-layer version of this same equivalence proof -- except
        the model can be fine-tuned with the mask active in between `.mask()`
        and this call. `commit_mask` is a no-op for a module that was never
        masked, so this is also safe to call on a group `.mask()` was never
        called on (identical to plain `.prune()` in that case)."""
        for name in self.output_targets:
            commit_mask(self.model.get_submodule(name))
        return self.prune()
