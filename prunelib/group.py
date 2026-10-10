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
one layer at a time -- `.mask()` zeroes every targeted module's channels
through its module rule (safe to fine-tune against, shapes unchanged),
`.commit_and_compress()` bakes the zeros in and runs the same rebuild
`.prune()` does. `.prune()` is the one-shot alternative.

This used to be the top half of `graph.py`. It is its own module because
changing how a decision is applied (a new rebuild path, masking) and
changing how one is found (tracing, propagation) are different reasons to
edit the code.

A group is computed against the model as it was at that moment, so it
refuses to act once the model has been resized since (`tensor_layout`):
channel `i` of a layer is not the same channel after another prune has
renumbered it. Two groups computed up front for two branches of a `cat`,
then applied one after the other, used to remove the wrong input channels
of the layer after the `cat` with the second one; and `.prune()` called
twice on one group pruned the same positions again, silently.
"""
from __future__ import annotations

import itertools
from types import MappingProxyType
from typing import Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune

from .module_rules import ModuleRule, find_module_rule

SpecialHandler = Callable[[nn.Module, torch.Tensor], nn.Module]


def tensor_layout(model: nn.Module) -> dict[str, torch.Size]:
    """Every parameter's and buffer's qualified name and shape: what a prune
    changes, however it was done. `torch.nn.utils.prune`'s masking is read
    through -- `weight_orig` stands for `weight`, its `weight_mask` buffer
    isn't listed -- since masking changes no shape. `DependencyGraph`
    re-traces when this changes; a `PruningGroup` refuses to act."""
    layout = {}
    for prefix, module in model.named_modules():
        masked = {h._tensor_name for h in module._forward_pre_hooks.values() if isinstance(h, torch_prune.BasePruningMethod)}
        tensors = itertools.chain(module.named_parameters(recurse=False), module.named_buffers(recurse=False))
        for name, t in tensors:
            if name.endswith("_mask") and name[: -len("_mask")] in masked:
                continue
            if name.endswith("_orig") and name[: -len("_orig")] in masked:
                name = name[: -len("_orig")]
            layout[f"{prefix}.{name}" if prefix else name] = t.shape
    return layout


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
    from that module's *output* channels; `input_targets[name]` from its
    *input* channels. Which weight dimension each one is belongs to the
    module's rule (`ModuleRule.output_weight`, `rebuild`): dim 0 and dim 1
    for every built-in type, not for a `ConvTranspose2d`. A module can
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
    rebuilt. The three target maps are read-only views: a caller writing
    into them directly used to bypass that disagreement check.

    Apply a group before computing the next one. `.prune()`, `.mask()` and
    `.commit_and_compress()` raise `RuntimeError` once the model has been
    resized since the group was computed (see the module docstring)."""

    def __init__(self, model: nn.Module, module_rules: Mapping[type, ModuleRule] | None = None):
        self.model = model
        self._output_targets: dict[str, torch.Tensor] = {}
        self._input_targets: dict[str, torch.Tensor] = {}
        self._special_targets: dict[str, tuple[SpecialHandler, torch.Tensor]] = {}
        self.module_rules = module_rules
        self._layout = tensor_layout(model)

    @property
    def output_targets(self) -> Mapping[str, torch.Tensor]:
        return MappingProxyType(self._output_targets)

    @property
    def input_targets(self) -> Mapping[str, torch.Tensor]:
        return MappingProxyType(self._input_targets)

    @property
    def special_targets(self) -> Mapping[str, tuple[SpecialHandler, torch.Tensor]]:
        return MappingProxyType(self._special_targets)

    def add_output_target(self, name: str, idx: torch.Tensor) -> bool:
        """Remove output channels `idx` of module `name`. Returns False if
        this exact target was already recorded; raises `ValueError` if a
        *different* one was."""
        return _record_target(self._output_targets, name, idx)

    def add_input_target(self, name: str, idx: torch.Tensor) -> bool:
        """Remove input channels `idx` of module `name` (see
        `add_output_target`)."""
        return _record_target(self._input_targets, name, idx)

    def add_special_target(self, name: str, handler: SpecialHandler, keep_idx: torch.Tensor) -> None:
        """Rebuild module `name` as `handler(module, keep_idx)` on `.prune()`,
        instead of through its module rule."""
        self._special_targets[name] = (handler, keep_idx)

    def _check_current(self) -> None:
        if tensor_layout(self.model) != self._layout:
            raise RuntimeError(
                "the model has been resized since this pruning group was computed (by this group's own .prune(), "
                "or another group's), so its channel indices may no longer name the channels they did -- call "
                "DependencyGraph.get_pruning_group again for a group that matches the model as it is now"
            )

    def prune(self) -> nn.Module:
        """Rebuild every targeted module with the recorded indices removed,
        and write each one back onto `self.model` in place. Returns the
        (mutated) model for convenience.

        Every replacement is built before any is swapped in, so a rule or
        handler that refuses (a weight computed by weight norm, say) leaves
        the model as it was. Each used to be swapped in as soon as it was
        built, so a refusal halfway through left a half-pruned model that
        failed on its next forward pass. (A handler that mutates its module
        in place before refusing is its own business.)"""
        self._check_current()
        replacements = {}
        for name in dict.fromkeys([*self._output_targets, *self._input_targets]):
            module = self.model.get_submodule(name)
            replacements[name] = self._rule(name, module).rebuild(
                module, self._output_targets.get(name), self._input_targets.get(name)
            )
        for name, (handler, keep_idx) in self._special_targets.items():
            replacements[name] = handler(self.model.get_submodule(name), keep_idx)

        for name, new_module in replacements.items():
            _set_submodule(self.model, name, new_module)
        return self.model

    def _rule(self, name: str, module: nn.Module) -> ModuleRule:
        rule = find_module_rule(module, self.module_rules)
        if rule is None:
            raise TypeError(f"{name!r}: don't know how to prune a {type(module).__name__}")
        return rule

    def mask(self) -> None:
        """Phase 1 of the two-phase mask-then-compress workflow (see
        `masking.py`'s module docstring), extended here to a whole
        dependency group instead of one layer at a time: zero every
        *output*-target module's pruned channels in place, through its
        module rule's `mask` (by default `masking.mask_channels`'s
        reparametrization), rather than immediately rebuilding smaller
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
        module unmasked while fine-tuning -- call `.prune()` instead. A
        module rule can refuse too (a `BatchNorm` with running statistics
        but `affine=False`); any
        modules masked before the refusal are still rebuilt correctly by
        `.prune()`, which only reads their kept channels.
        """
        self._check_current()
        if self._special_targets:
            raise NotImplementedError(
                f"{sorted(self._special_targets)} are rebuilt by a special handler, which can't be "
                f"masked first -- call .prune() on this group instead"
            )
        for name, idx in self._output_targets.items():
            if idx.numel() == 0:
                continue
            module = self.model.get_submodule(name)
            try:
                self._rule(name, module).mask(module, idx)
            except NotImplementedError as exc:
                raise NotImplementedError(f"{name!r}: {exc}") from None

    def commit_and_compress(self) -> nn.Module:
        """Phase 2: bake every mask `.mask()` applied into real zeros on
        each `output_targets` module, through its module rule's
        `commit_mask` (the counterpart of the `mask` that applied it), then
        run the same rebuild `.prune()` does. Equivalent to calling `.prune()`
        directly on an unmasked group -- see
        `tests/test_masking.py::test_compress_masked_conv_bn_matches_direct_surgery`
        for the single-layer version of this same equivalence proof -- except
        the model can be fine-tuned with the mask active in between `.mask()`
        and this call. `commit_mask` is a no-op for a module that was never
        masked, so this is also safe to call on a group `.mask()` was never
        called on (identical to plain `.prune()` in that case)."""
        self._check_current()
        for name in self._output_targets:
            module = self.model.get_submodule(name)
            self._rule(name, module).commit_mask(module)
        return self.prune()
