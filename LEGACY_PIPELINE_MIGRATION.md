# Migrating from the original driver scripts to `legacy_pipeline`

`legacy_pipeline/` replaces `channel_pruning_saliency.py`,
`channel_pruning_distance.py`, `vgg_channel_pruning_saliency.py`,
`vgg_channel_pruning_dist.py`, and the two never-finished kernel-level
scripts (`kernel_pruning_saliency.py`, `vgg_kernel_pruning_saliency.py`).
`kernel_pruning_similarites.py`'s scoring function
(`compute_distance_score_kernel`) was one of the few genuinely correct
functions in the old codebase, but kernel-level (as opposed to channel-level)
structural surgery isn't implemented anywhere in this package yet — see
"What's not covered" below.

Full defect-by-defect detail on what was wrong with the originals is in
`GITHUB_AUDIT.md`, sections 4, 5, and 11. This document is about what
replaced it and why, and how to move a workflow from the old scripts to the
new package.

## The design change, not just the bug fixes

The original scripts' intent — confirmed directly, not inferred — was:
**during pruning, mask the weights being pruned; once the whole pruning
schedule is finished, build a compressed model and copy the unmasked
(surviving) weights across from the original.** That's a real, sound
two-phase design. It just never worked, because of the specific bugs listed
in `GITHUB_AUDIT.md` section 11: masking used a hand-written
`torch.nn.utils.prune.BasePruningMethod` subclass whose `compute_mask` had
to reconstruct the correct mask from scratch on every call using a global
`layer_number` that was supposed to track which layer was active — and the
line that updated it was commented out in all six scripts, so it silently
stayed 0 forever.

`legacy_pipeline` implements the same two-phase design the original scripts
were reaching for, using `torch.nn.utils.prune.custom_from_mask` (the
current, stable PyTorch API for applying a precomputed mask) instead of a
`BasePruningMethod` subclass:

**Phase 1 — mask** (`prunelib.mask_vgg_layer`, called once per layer per
iteration from `legacy_pipeline.pipeline.run_pruning`): compute a saliency
score, mask the weakest surviving channels via reparametrization, leave
every tensor's shape untouched. Safe to fine-tune against — gradients
naturally don't flow to masked positions, since PyTorch's reparametrization
makes the effective weight `weight_orig * mask`, so `d(loss)/d(weight_orig)`
is itself zero wherever `mask` is zero.

**Phase 2 — compress** (`prunelib.compress_masked_vgg`, called once at the
end): commit every layer's accumulated mask (bake zeros in permanently),
read back which channels ended up all-zero, and build the actual smaller
network — one real, physically reduced `torchvision.models.vgg16` instance,
not a same-shaped model with some zeros in it.

`tests/test_masking.py::test_compress_masked_conv_bn_matches_direct_surgery`
and `tests/test_vgg_masking.py::test_compress_masked_vgg_matches_masked_accuracy_numerically`
both confirm this numerically: a masked model and the compressed model built
from it produce identical output, because compression only removes channels
that were already contributing exactly zero.

### A correctness fix this design needed that wasn't obvious up front

Naively, "mask another `prune_step` fraction of channels each iteration"
means: score every channel, pick the lowest `prune_step * n_channels`, mask
them. Applied literally on iteration 2 and later, this breaks — an
already-masked channel's weight is zero, which is the lowest possible score
under every criterion in `saliency.py`, so it wins re-selection on every
subsequent iteration. Most of each iteration's "newly pruned" budget would
actually be spent re-selecting channels that were already gone, and the
schedule would advance far slower than the requested `prune_step` per
iteration implies.

`mask_vgg_layer` restricts scoring and selection to
`prunelib.surviving_channels` — channels not already masked — so every call
masks approximately `prune_step * n_channels_original` *new* channels.
`tests/test_vgg_masking.py::test_mask_vgg_layer_excludes_already_masked_channels_on_repeat_calls`
is the regression test for this; without the fix it fails by showing the
second and third calls masking far fewer channels than the first.

## Other corrections along the way

| Original file | Problem | Fix |
|---|---|---|
| `train_model.py::evaluate()` | D14: reassigned `outputs` inside the batch loop instead of accumulating — reported only the last batch | `legacy_pipeline/train.py::evaluate()` accumulates loss/correct/sample counts across the whole loader, divides once at the end |
| `train_model.py::fit_one_cycle()` | D15: called `evaluate()` once per training batch, not once per epoch | `legacy_pipeline/train.py::fit_one_cycle()` evaluates once, after the epoch's training loop |
| `train_model.py`'s L1 term | `sum(nn.L1Loss()(param, zeros) for param in model.parameters())` averages *within* each parameter tensor before summing across tensors — implicitly weights a small bias vector the same as a huge weight matrix | `legacy_pipeline/train.py::_l1_penalty()` sums raw magnitudes globally |
| `load_model.py::freeze()` | Hardcoded a parameter *index* ("if count == 30") correct for exactly one `num_classes` value | `legacy_pipeline/model.py::freeze_all_but_classifier()` freezes everything except `model.classifier` directly — can't drift out of sync with the architecture |
| `load_dataset.py::data_loader_eval()` | One of its two transform pipelines used `transforms.CenterCrop` as a class, not an instance (`transforms.CenterCrop` instead of `transforms.CenterCrop(224)`) — silently wrong preprocessing, not an error, since `Compose` just calls whatever's handed to it | `legacy_pipeline/data.py` has exactly one `_eval_transform()` builder used everywhere eval-style preprocessing is needed |
| `load_model.py`'s `pretrained=True/False` | Deprecated boolean API, removed in current `torchvision` | `prunelib.build_vgg16` uses `weights=VGG16_Weights.IMAGENET1K_V1` |
| Six driver scripts' path/config setup | ~15 lines of copy-pasted `dataset_dir`/`logDir`/module-level-global setup duplicated near-verbatim in every file | `legacy_pipeline/config.py::PruningConfig` — one dataclass, constructed once, passed explicitly |

## How to run it

```bash
pip install -e ".[vision-experiments,dev]"   # torchvision + pytest

# Mechanical check first -- FakeData, random-init weights, no downloads, ~1-2 min:
python -m legacy_pipeline.pipeline --tiny-check

# The real thing -- downloads CIFAR-10 + ImageNet VGG16 weights:
python -m legacy_pipeline.pipeline --method max_k --dataset CIFAR10 --dataset-dir ./data
```

Or from Python:

```python
from legacy_pipeline import PruningConfig, run_pruning

cfg = PruningConfig(dataset_name="CIFAR10", method="max_k")
results, compressed_model = run_pruning(cfg)
```

`results` has one row per masking iteration (`phase="masked"`, `params`
unchanged from baseline — masking doesn't remove parameters, only the final
compression step does) plus a final `phase="compressed"` row where `params`
actually drops. Both the printed log and `cfg.results_csv_path` make this
explicit so it's never ambiguous which number reflects a real, physically
smaller model.

## What's not covered

- **Kernel-level pruning** (as opposed to channel-level) isn't implemented
  in `prunelib` or `legacy_pipeline` at all. The original
  `kernel_pruning_saliency.py` / `vgg_kernel_pruning_saliency.py` never got
  past a placeholder (`GITHUB_AUDIT.md` section 11), and
  `kernel_pruning_similarites.py`'s scoring function
  (`compute_distance_score_kernel`) is correct but has no counterpart here.
  If this is needed, `prunelib.masking`'s primitives (`mask_channels`,
  `commit_mask`) generalize to it — the piece that doesn't yet exist is
  kernel-level *structural surgery* (removing one kernel from a specific
  input-channel group without removing the whole output channel), which
  `prunelib.surgery` doesn't have an equivalent of.
- **K-Means clustering-based selection** (Paper 2's Manhattan/Euclidean/
  Cosine comparison) — `prunelib.scanners` has the distance metrics, not the
  clustering + per-cluster selection loop. See `KT.md` section 6/7.
- **Distance/similarity-based channel selection** for VGG specifically
  (the corrected replacement for `channel_pruning_distance.py` /
  `vgg_channel_pruning_dist.py`) isn't wired into `legacy_pipeline` — only
  `method="max_k"|"l1"|"l2"|"random"` (saliency-style, one score per
  channel) are. `prunelib.pairwise_distance_matrix` has the metric; a
  `mask_vgg_layer`-equivalent that selects by pairwise similarity rather
  than a per-channel score would need its own selection logic (pick one
  channel from each close pair to drop), not just a different `method=`
  argument to the existing function.
