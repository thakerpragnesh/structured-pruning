# Knowledge Transfer — structured-pruning

For anyone picking up this codebase — including future-you six months from
now. Read this before changing `prunelib/`.

## 1. What this is and why it exists — and what it *isn't*, since this changed

**Scope note, read this first if you haven't looked at this repo in a
while:** this package started as a full replacement for the buggy CNN
pruning scripts in `pruning_framwork_v4`. Those scripts were subsequently
fixed *directly, in that repo*, with more complete method coverage
(K-means, SVD, hybrid sequencing) than this package's `prunelib` has.
**`pruning_framwork_v4` is now canonical for CNN pruning.** This package's
job is narrower: the Transformer-extension experiments, and a tested
reference implementation of the mask-then-compress design (section 4
covers both). See the top of `README.md` for the current framing — if a
change you're considering is "add K-means to `prunelib`," it belongs in
`pruning_framwork_v4` instead, not here.

The history below is why the code looks the way it does; it's not a claim
about what this repo currently supersedes.

This package is a reference implementation of the structured-pruning
methods from six of my papers (see `PUBLICATIONS.md`), originally built
because four earlier repos (`pruning_framwork`, `_v2`, `_v3`, `_v4`)
contained real, confirmed bugs — including one in the exact method the
papers are named after.

That history matters for how this code is written, not just as trivia. Every
non-obvious design choice in `prunelib/` exists because of a specific,
verified bug in the old code. If a change looks like it's undoing one of
these choices, it's probably reintroducing the bug. Section 3 maps each
choice to the defect it prevents.

**The short version of the history:** `pruning_framwork_v4/facilitate_pruning.py`
computed a Max-3 saliency score into a tensor called `channel_norm`, then
selected which channels to prune by comparing a *different* tensor,
`channel_norm_temp`, which was never updated and still held the plain L1
norm. The score was computed, stored, returned — and silently ignored. Six
more defects (D2–D7) were found in the same file and one neighboring one.
None of them were exotic — an off-by-one, a top-k tracker that only updates
on new maxima, a boolean tensor passed to `torch.norm()`. All are listed in
detail in section 3. All are pinned by a test in `tests/`.

The real irony: the *papers* are fine. Max-3 and L1 produce measurably
different accuracy curves in the published results, which they couldn't if
v4's broken code had generated them. Working Max-3 code existed — it just
was never pushed to GitHub. It isn't recoverable; this package's `prunelib`
was the reference implementation until `pruning_framwork_v4` got fixed in
place — see the scope note above for where things stand now.

## 2. Repository layout

```
prunelib/
    saliency.py   (150 lines)  Max-k / L1 / L2 / random scoring, single dispatch,
                               Conv2d or Linear weights
    surgery.py    (216 lines)  Conv/BN/FFN/attention-head structural surgery
                               (in-place substitution)
    masking.py    two-phase mask-then-compress workflow (torch.nn.utils.prune)
    graph.py      (661 lines) torch.fx dependency resolution -- generic add/cat/
                               flatten/depthwise surgery, no seam passed by hand;
                               plus prune_model() (one-call generic pruning),
                               PruningGroup.mask()/.commit_and_compress() (two-phase
                               workflow for a whole dependency group), and
                               special_handlers/LeafTracer (attention-block hook)
    quantization.py (182 lines) Post-training Float16 / linear INT8 / Fixed-Point32
                               quantization (thesis Ch. 6.6) -- separate compression
                               stage from pruning, applied after it
    vgg.py        VGG wiring: build_vgg16, prune_vgg_layer, mask_vgg_layer, compress_masked_vgg
    scanners.py   (70 lines)   Distance metrics + co-activation scanning
    evaluate.py   (59 lines)   Parameter counts, measured latency, estimated size at a bit-width
experiments/
    00_demo.py                 (76 lines)  full pipeline, seconds, no dependencies beyond torch
    01_vgg_cifar10_sweep.py   (236 lines)  VGG16/CIFAR-10 — see section 5, this one has three run modes
    02_bert_sst2_sweep.py      (77 lines)  BERT FFN pruning
    03_head_redundancy.py      (63 lines)  attention head distance scan
    04_coactivation.py         (44 lines)  synthetic co-activation demo
    05_ordering.py             (69 lines)  does the CNN ordering result transfer?
    06_generic_pruning.py     (130 lines)  DependencyGraph on a real ResNet-18
    07_quantization.py         (85 lines)  prune_model() + all three quantization methods, one pipeline
archive/
    legacy_pipeline/        corrected rebuild of pruning_framwork_v4's original
        config.py, data.py, driver scripts. Superseded by direct fixes to that
        model.py, train.py, repo (which has more complete method coverage) --
        pipeline.py         kept here as a tested mask-then-compress example,
                             not a live alternative. Moved under archive/
                             2026-09-21. See LEGACY_PIPELINE_MIGRATION.md.
tests/          75 tests across 9 files, one per historical defect or behavior
```

Total: ~2,900 lines. Small on purpose — every module does one thing.

## 3. Design decisions, and the defect each one prevents

If you're refactoring `prunelib`, read this table first.

| Decision | Defect it prevents | Where |
|---|---|---|
| `compute_score()` is the *only* way to get a saliency score; `select_prune_indices()` takes a score tensor, never a weight tensor | D1: score computed one place, selection reads a different, stale tensor | `saliency.py` |
| `torch.topk` instead of a hand-rolled "track top-3" loop | D2: manual tracker only rotated on a *new* max, silently dropping mid-range values | `saliency.py::max_k_saliency` |
| Per-kernel top-k summed across input channels in one vectorized op (`.sum(dim=2)` then `.sum(dim=1)`), not an accumulator reset "once per output channel" but mutated inside an inner loop | D3: score accumulated from monotonically-growing partial maxima across input channels — not the intended quantity at all | `saliency.py::max_k_saliency` |
| Kernel height and width read from their own tensor dimensions (`kh, kw = weight.shape[2], weight.shape[3]`), never reusing one dimension for both | D4: original iterated `range(size[2])` for *both* height and width, dropping columns on non-square kernels | `saliency.py::max_k_saliency` |
| `prune_ffn_block` and `prune_conv_bn` raise `ValueError` immediately if shapes don't match, before touching any tensor | D5: a shape bug surfaced as a confusing `IndexError` deep inside an unrelated loop instead of at the actual mistake | `surgery.py`, both functions' first lines |
| Surgery is pure advanced indexing (`weight.index_select(0, keep_idx)`); no manual per-channel copy loop with a hand-maintained destination counter | D6: the old copy loop's destination index never incremented, so every surviving channel wrote to index 0 | `surgery.py::prune_conv_bn` |
| Same as D6 — no manual loop, so there's no "off-by-one before first use" to have | D7: `fin_new = fin_org` then `fin_new += 1` before the variable was ever read | `surgery.py` (structurally can't recur) |
| Pruning a VGG layer replaces modules in-place inside `model.features` rather than rebuilding a whole model from a `feature_list` and copying weights in | This *is* D6/D7's root cause, one level up — the old code's entire "reconstruct then copy" pattern is gone, not just patched | `experiments/01_vgg_cifar10_sweep.py::prune_vgg_layer`, `vgg.py::compress_masked_vgg` |
| Masking uses `torch.nn.utils.prune.custom_from_mask` — one call, full mask computed up front — instead of subclassing `BasePruningMethod` with a `compute_mask` that reconstructs the mask from global state on every call | The systemic bug across *every* driver script in old `pruning_framwork_v4`: a `layer_number` global was supposed to tell `compute_mask` which layer it was on, and the line updating it was commented out in all six scripts, so it silently stayed 0 forever (see `GITHUB_AUDIT.md` section 11) | `masking.py::mask_channels` |
| `mask_vgg_layer` scores and selects only from `surviving_channels` (not-yet-masked channels), never all channels | Without this, an already-masked channel's weight is zero — the lowest possible score under every criterion — so it wins re-selection on every later iteration, and the pruning schedule advances far slower than the requested fraction implies | `vgg.py::mask_vgg_layer` |
| "Which indices are *not* in this index tensor" is a boolean mask + `nonzero()`, never `set(idx.tolist())` plus a Python `range()` comprehension | Not a historical defect, but a recurring anti-pattern found and fixed 2026-09-22 in three places (`masking.py::surviving_channels`, `vgg.py::prune_vgg_layer`, `saliency.py::keep_indices`) — same result, but drops to a Python-level loop over every channel instead of one vectorized op. If you write `set(...)` + a `range()` comprehension against a channel/index tensor anywhere in this codebase, that's the pattern to replace | `masking.py`, `saliency.py`, `vgg.py` |

If you ever find yourself writing a loop that tracks a running max/min by
hand, or a loop with a manually-incremented destination index into a
tensor — stop. Both patterns have already caused a real bug in this
project's history. Use `torch.topk`/`torch.sort` and advanced indexing
instead.

## 4. Module walkthrough

### `prunelib/saliency.py`

Four scoring functions (`max_k_saliency`, `l1_saliency`, `l2_saliency`,
`random_saliency`), all with signature `(weight: [out,in,kh,kw]) -> [out]`,
routed through `compute_score(weight, method=...)`. `select_prune_indices`
and `keep_indices` operate on the *score* tensor, not the weight — this
separation is what makes D1 structurally impossible to reintroduce, since
there's no code path where a caller has both a weight tensor and a stale
score tensor in scope at the same time.

`max_k_saliency` is the one to understand well: it flattens each kernel to
1D, takes the top-`k` magnitudes with `torch.topk`, sums them (`kernel_scores`,
shape `[out, in]`), then sums across the input-channel axis exactly once
(`channel_scores`, shape `[out]`). That's the whole algorithm. If a change
here doesn't fit in three lines of einsum-shaped reasoning, it's probably
reintroducing a shape bug — this function has been the site of 4 of the 7
historical defects.

### `prunelib/surgery.py`

Two functions: `prune_conv_bn` (Conv2d + optional BatchNorm2d + optional next
Conv2d) and `prune_ffn_block` (Linear → activation → Linear). Both build
*new* modules of the correct smaller size and copy the kept rows/columns in
— they never mutate the originals. Both validate the seam (that the tensor
shapes on either side of the cut actually match) before doing anything, and
raise `ValueError` with a specific message if not.

One deliberate design choice worth knowing: `prune_conv_bn` copies
BatchNorm's running mean/var across to the new, smaller BN layer. The
original code didn't do this at all — pruned models ran without error but
produced silently wrong activations until the next full training pass reset
the running stats. This isn't "more correct" in some abstract sense, it's
specifically there because that failure mode doesn't throw, so nothing would
tell you it happened.

### `prunelib/masking.py`

The two-phase mask-then-compress workflow: `mask_channels` zeroes a
channel's weight *and* bias via `torch.nn.utils.prune.custom_from_mask`
(non-destructive — shapes don't change, and it composes correctly across
repeated calls on the same layer, which is what makes it safe to call once
per pruning iteration). `commit_mask` bakes the zeros in permanently.
`compress_masked_conv_bn` reads back which channels ended up all-zero and
calls `surgery.prune_conv_bn` on the survivors — same tested surgery code
as everywhere else, just fed indices derived from a committed mask instead
of a fresh score.

The point of this two-phase split: you can fine-tune or evaluate with the
mask active, across as many iterations as you want, before committing to
an architecture change. Gradients naturally don't reach masked positions
(`weight = weight_orig * mask`, so `d(loss)/d(weight_orig) = d(loss)/d(weight) * mask`,
which is zero wherever the mask is zero) — no extra bookkeeping needed to
keep pruned channels from "coming back" during fine-tuning.

`tests/test_masking.py::test_compress_masked_conv_bn_matches_direct_surgery`
is the test to read if you're unsure this is equivalent to just calling
`prune_conv_bn` directly — it proves the two paths produce numerically
identical output.

### `prunelib/graph.py`

Added 2026-09-22 to close a gap every other module in `prunelib` leaves
open: `prune_conv_bn` needs `next_conv=` passed in by hand, `vgg.py` only
works because it hardcodes that VGG's `.features` is a flat `Sequential` —
nothing before this traced an arbitrary `nn.Module` and worked out its own
wiring. `DependencyGraph(model, example_input)` traces the model once with
`torch.fx`; `get_pruning_group(layer, keep_idx)` walks forward from a prune
decision to find every other layer it couples to — a residual/skip partner
(via elementwise add, walked backward to whichever conv/Linear actually
produced the other branch), a `torch.cat` (indices get shifted by the other
branches' channel counts), the conv-output-flattened-into-Linear classifier
boundary (channel indices expand by the spatial size), and depthwise convs
(indices pass straight through, since input and output channel count are
the same dimension there). Anything it doesn't recognize — a non-depthwise
grouped conv, an attention block, an unrecognized module — raises
`NotImplementedError` rather than silently mis-wiring something; see the
module's own docstring for the full list of what's deliberately out of
scope (no automatic attention-block discovery; only one `cat` branch pruned
at a time; a few other named limitations).

One real deviation from the rest of this library, called out in the
docstring: `PruningGroup.prune()` mutates the model in place instead of
returning new modules for the caller to wire up. That's deliberate — the
whole point of this module is that the caller shouldn't need to know the
graph shape well enough to do that wiring themselves.

`experiments/06_generic_pruning.py --tiny-check` is the test to read if
you want proof this holds on something real: it prunes a conv inside a
`torchvision.models.resnet18` `BasicBlock` with no seam passed by hand, and
the resulting `PruningGroup` correctly cascades through the rest of that
residual stage (both blocks share the same channel count) and into the next
stage's transition convs — none of that cascading was specifically
designed for, it falls out of the forward-walk naturally.

**Added 2026-09-23, three extensions, none changing the behavior above:**

- `prune_model(model, example_input, layer, prune_fraction, method=...)` —
  the `vgg.py::prune_vgg_layer` convenience (score → select → surgery, one
  call), generalized to any model `DependencyGraph` can trace instead of
  hardcoding VGG's flat `.features`. `experiments/07_quantization.py` uses
  it as its pruning step; `test_graph.py::test_prune_model_scores_and_prunes_in_one_call`
  is the unit test.
- `PruningGroup.mask()` / `.commit_and_compress()` — the two-phase
  mask-then-compress workflow from `masking.py` (see section 4's entry for
  it), extended to a whole dependency group: `.mask()` zeroes every
  output-target module via `masking.mask_channels` instead of immediately
  rebuilding, so you can fine-tune with the group's channels masked before
  committing; `.commit_and_compress()` bakes the zeros in and runs the same
  rebuild `.prune()` does. `test_graph.py::test_mask_then_commit_and_compress_matches_direct_prune`
  proves the two paths produce identical weights, mirroring
  `test_masking.py::test_compress_masked_conv_bn_matches_direct_surgery`'s
  reasoning one level up.
- `special_handlers` (a `{type: handler}` map on `DependencyGraph.__init__`)
  and `LeafTracer` — the extension point for attention-block pruning this
  module's docstring disclaims doing automatically. When `get_pruning_group`'s
  *starting* layer is an instance of a registered type, the handler (not the
  Conv2d/Linear/BatchNorm logic) decides how to rebuild it — see
  `test_graph.py::test_special_handler_prunes_attention_block_via_prune_attention_heads`
  for a worked example wiring a handler to `surgery.prune_attention_heads`.
  `LeafTracer` exists because a custom block class isn't a leaf under
  `fx.Tracer`'s default rule, so without it `DependencyGraph` would trace
  straight through the block and never see it as one node to hand to the
  handler. Deliberately scoped to the *starting* layer only — a block
  encountered mid-propagation (as a consequence of some other layer's
  prune) still raises `NotImplementedError`, same as before; translating an
  arbitrary upstream channel prune into a head-pruning decision is a
  different, harder problem this hook doesn't attempt.

### `prunelib/quantization.py`

Added 2026-09-23 to close the gap section 10.2 (below) used to describe in
full: quantization wasn't implemented anywhere in this codebase, despite
being half the thesis's title. Three post-training methods, reference
implementations of thesis Ch. 6.6: `quantize_float16`/`dequantize_float16`
(direct cast, the thesis's best accuracy/memory tradeoff), `quantize_int8_linear`/
`dequantize_int8_linear` (the standard scale/zero-point affine mapping, Eq.
6.7–6.10, returning an `Int8Tensor` struct since realizing INT8's actual 4x
memory reduction means storing `int8` values plus scale/zero-point rather
than a same-shape float tensor), and `quantize_fixed_point32` (1-sign/3-
integer/28-fractional bit format, thesis's own finding that it's markedly
worse than Float16 since a fixed exponent can't adapt to a layer's actual
weight distribution — kept for reproducing that comparison, not as a
recommendation). `quantize_model_` applies Float16 or Fixed-Point32 to
every Linear/Conv2d/BatchNorm weight and bias in a model, in place — the one
function in this module (besides `masking.commit_mask`, its precedent)
that deviates from `prunelib`'s "never mutate what's passed in" convention,
since quantization changes no tensor's shape, so there's no smaller module
to construct and return instead. `evaluate.estimate_size_bytes(module,
bits_per_param)` complements it — a quick way to compare precision options'
memory footprint without writing a checkpoint to disk for each, the helper
section 10.2 used to say `evaluate.py` was the natural home for.
`experiments/07_quantization.py` runs the whole pipeline (`prune_model` then
all three methods) end to end; `tests/test_quantization.py` is the unit
suite, including a brute-force comparison against the Eq. 6.7–6.9 formula
(`test_int8_quantize_matches_the_documented_formula`) in the same style as
`saliency.py`'s brute-force tests.

### `prunelib/vgg.py`

VGG-specific wiring on top of the layer-level primitives above.
`build_vgg16` wraps `torchvision.models.vgg16` (using `weights=`, not the
deprecated `pretrained=True/False`). `prune_vgg_layer` is the direct,
one-shot version (score → select → surgery, immediately). `mask_vgg_layer`
/ `compress_masked_vgg` are the two-phase version — see the design-decision
table in section 3 for the correctness fix `mask_vgg_layer` needed
(excluding already-masked channels from re-selection).

Both `prune_vgg_layer` and `mask_vgg_layer` refuse to touch the *last*
conv layer (it feeds `classifier[0]`, a `Linear`, not another `Conv2d` —
see section 6).

### `archive/legacy_pipeline/`

Moved under `archive/` 2026-09-21 (was `legacy_pipeline/` at the repo root;
import as `archive.legacy_pipeline`) to make explicit that this package is
`prunelib` plus the Transformer experiments first, not a pipeline you run in
place — see README.md's opening line. A corrected rebuild of
`pruning_framwork_v4`'s original six driver scripts, done in parallel with
this package, before those scripts got fixed directly in that repo. **Now
superseded**: `pruning_framwork_v4`'s own fix has more complete method
coverage (K-means, SVD, hybrid sequencing) than `legacy_pipeline` does, since
`legacy_pipeline` is built on this package's `prunelib` (Max-k/L1/L2/random
only). Don't extend `legacy_pipeline` to add parity with
`pruning_framwork_v4` — that work belongs in `pruning_framwork_v4` itself.
What's still worth reading here:
`pipeline.py::run_pruning` is a complete, tested example of the
mask-then-compress workflow applied to a whole VGG16 end to end (mask
every prunable layer per iteration → fine-tune → evaluate → repeat →
compress once at the end), and `tests/test_legacy_pipeline.py` is the
regression test for that. See `LEGACY_PIPELINE_MIGRATION.md` for the full
defect-by-defect mapping from the original scripts.



`pairwise_distance_matrix(vectors, metric)` implements Manhattan/Euclidean/
Cosine — the three metrics compared in Paper 2. **Important gap:** this
computes *distances only*. It does not implement K-Means clustering itself,
or the "keep the highest-L1-norm channel from each cluster" selection rule
the paper actually uses for K-Means-based pruning. If you need to reproduce
that specific experiment, you'll need to add a `k_means_select()` function
(sklearn's `KMeans` on the distance-derived feature matrix, or normalized
channel vectors directly, then select per-cluster by L1 norm) — see section
6 for where this fits.

`CoActivationScanner` computes pairwise Jaccard similarity on boolean firing
masks, with a `firing_rate_ceiling` (default 0.9) that excludes near-
universal-firing units before computing similarity. This exists because a
unit that fires on 95%+ of tokens will show spuriously high overlap with
*any* other frequently-firing unit — that's an artifact of both units being
non-selective, not evidence of redundancy. This hasn't been validated
against real model activations yet (see section 6) — the logic is tested
against synthetic masks in `tests/test_scanners.py` and demonstrated in
`experiments/04_coactivation.py`, but nobody has confirmed the 0.9 default
is the right threshold on an actual fine-tuned model.

### `prunelib/evaluate.py`

`count_params`, `count_encoder_params` (params under a named submodule, e.g.
`.encoder`, excluding embeddings), and `measure_latency` (wall-clock,
warmup + averaged iterations, CPU by default). Nothing subtle here, but
note: `measure_latency`'s numbers are hardware- and batch-size-dependent.
Don't hardcode a specific multiplier anywhere that isn't clearly labeled
with the machine/conditions it came from — see the README's note about the
1.8x vs 2.13x latency discrepancy for why this matters in practice.

### `experiments/`

Three-tier pattern used across the VGG and BERT experiments:

- **`--smoke`** — fully synthetic data and a tiny hand-built model (no
  `torchvision`/`transformers` model classes involved for `01`). Runs in
  under a second. This is what CI runs on every push.
- **`--tiny-check`** (currently only on `01`) — the *real* model class
  (`torchvision.models.vgg16`) with randomly-initialized weights and
  `FakeData` instead of a real dataset. No network access needed, but it
  exercises the actual code path `run_full()` uses. Takes minutes, not
  seconds — not run in CI, but should be run manually after any change to
  `prune_vgg_layer` or `build_vgg16`.
- **no flag / `run_full()`** — the real thing. Downloads real weights and a
  real dataset. Needs internet access and has not been executed anywhere
  yet (see section 6) — everything up to this point has only been verified
  mechanically.

If you add a fourth experiment that touches a real pretrained model, follow
this same three-tier pattern rather than inventing a new one.

## 5. Testing conventions

Every test in `tests/` is named after either a historical defect (`test_d1_...`,
`test_d6_...`) or a specific behavior guarantee (`test_ffn_block_shrinks_and_preserves_values`).
When you fix a bug in this codebase, the workflow is:

1. Write a test that fails against the buggy code, named `test_dN_<what_broke>`
   if it's a regression of a known historical pattern, or descriptively
   otherwise.
2. Fix the bug.
3. Confirm the test passes and nothing else broke: `pytest tests/ -v`.

This means the test suite doubles as a defect log — reading `tests/` top to
bottom tells you most of this codebase's incident history without needing
`GITHUB_AUDIT.md` open at the same time. Keep it that way: don't delete a
defect-named test even if you refactor the code it guards, unless you're
certain the refactor makes the bug class structurally impossible (as, e.g.,
switching to `torch.topk` made D2 impossible to reintroduce even accidentally).

Run everything: `PYTHONPATH=. pytest tests/ -v` (75 tests, ~30-35s total on
CPU; the pure-`prunelib` tests alone (including `test_graph.py`,
`test_quantization.py`, and `test_evaluate.py`, none of which need
`torchvision`/`transformers`) are still ~2-3s, the rest is the handful
of tests that build a real `torchvision.vgg16` or a real HF BERT model).

The two slowest tests train/evaluate a real (random-init) VGG16 against
`archive/legacy_pipeline`'s FakeData path — `PruningConfig.fakedata_train_size`
/ `fakedata_test_size` (default 32/16, matching the old hardcoded values)
exist specifically so a test can shrink the synthetic dataset instead of
paying full VGG16-forward-pass cost on every batch. Fixed 2026-09-22:
`test_run_pruning_end_to_end_tiny_check` had been silently running at the
config default `image_size=224` despite being the "tiny check" — its sibling
test already used 32. That one line, plus shrinking both tests' FakeData
sizes via the new config fields, cut the full suite from ~59s to ~26-31s
without weakening what either test verifies (still 2 distinguishable epochs,
still `len(train_loader) > 1`, still a real end-to-end pipeline). If you add
a new test that spins up a real VGG16, use these fields rather than
hardcoding a smaller `image_size`/dataset size inline.

## 6. Known gaps — read this before claiming something works

Being explicit about what's *not* done is as important as documenting
what is, given this project's history of a paper's headline result having no
correct public implementation. Current state, honestly:

- **K-Means, SVD, and hybrid-sequencing criteria are not in this package's
  `prunelib`, and that's now by design, not a gap to fill here.** They
  exist, fixed and tested, in `pruning_framwork_v4`. `scanners.py` has the
  distance metrics `pairwise_distance_matrix` needs, but adding a
  `k_means_select()` here would create a second, competing implementation
  of something `pruning_framwork_v4` already does more completely — if you
  want K-means/SVD/hybrid pruning, use that repo, don't rebuild it here.
- **`prune_vgg_layer` can't prune the last conv layer** — it feeds
  `classifier[0]` rather than another `Conv2d`, and resizing that Linear
  layer's `in_features` isn't wired up. Straightforward to add; just not
  done.
- **`run_full()` in `experiments/01` has never been executed.** It's real
  code, verified mechanically via `--tiny-check` (real model class, no
  network), but nobody has run it against actual CIFAR-10 and actual
  ImageNet-pretrained weights yet. Don't cite a specific accuracy-drop
  number from this codebase until that's been done.
- **`experiments/02` and `03` are the same situation** — real
  `transformers` model classes, verified in `--smoke`, never run against a
  real fine-tuned checkpoint or real SST-2 data.
- ~~**FFN saliency scoring is a reshape hack, not a first-class API.**~~
  **Fixed 2026-09-21.** `saliency.py`'s scorers now accept a 2D Linear weight
  `[out_features, in_features]` natively (via `_channel_view`), alongside the
  original 4D Conv2d shape. `experiments/02_bert_sst2_sweep.py::_score_ffn_neurons`
  no longer reshapes around it.
- ~~**No attention-head *pruning* surgery exists**~~ **Fixed 2026-09-21.**
  `surgery.py` now has `prune_attention_heads`, the equivalent of
  `prune_conv_bn` for a multi-head attention block (touches Q/K/V rows and
  the output projection's columns simultaneously). `experiments/03` still
  only *detects* redundancy (distance-based); wiring detected pairs into
  `prune_attention_heads` calls is still open.
- **`CoActivationScanner`'s 0.9 firing-rate ceiling is a design choice, not
  a validated threshold.** Only tested against hand-constructed synthetic
  masks so far.
- ~~**Every surgery function needs the caller to already know the seam.**~~
  **Fixed 2026-09-22.** `prunelib.graph.DependencyGraph` traces an arbitrary
  model with `torch.fx` and works out which other layers a prune decision
  couples to on its own (residual/skip, `cat`, the flatten-into-classifier
  boundary, depthwise convs) — see its module walkthrough entry above and
  `graph.py`'s own docstring for what's still explicitly out of scope:
  non-depthwise grouped convs raise rather than attempt, and only one `cat`
  branch is safely pruned at a time within a single `get_pruning_group` call.
- ~~**`DependencyGraph` isn't wired into the two-phase mask-then-compress
  workflow — only one-shot surgery.**~~ **Fixed 2026-09-23.**
  `PruningGroup.mask()` / `.commit_and_compress()` extend `masking.py`'s
  two-phase design to a whole dependency group — see the `graph.py`
  walkthrough entry above.
- ~~**No automatic attention-block discovery in `DependencyGraph`, and no
  hook for one.**~~ **Partially addressed 2026-09-23** — `special_handlers`
  (see the `graph.py` walkthrough entry above) is a real, tested extension
  point now, not just a described-but-absent idea. What's still true, and
  still deliberate: it only fires when the *registered* type is the
  `get_pruning_group` call's starting layer, not when it's discovered
  automatically mid-graph or found without the caller naming the type up
  front — genuine automatic attention-block *discovery* (recognizing a
  reshape/matmul/softmax/matmul sequence as "one attention head" without
  being told) is still the "much harder problem" `graph.py`'s module
  docstring disclaims, and still isn't attempted.
- **Quantization now exists (`prunelib/quantization.py`, added 2026-09-23),
  closing what section 10.2 used to describe as "not present in this
  codebase at all."** All three thesis Ch. 6.6 methods are implemented and
  tested (Float16, linear INT8, Fixed-Point32) — see the module walkthrough
  entry above. What's still a gap, honestly: none of them have been run
  against a real fine-tuned model to reproduce the thesis's own accuracy-drop
  numbers (1.34% for Float16, 3.26% for Fixed-Point32 on a pruned VGG16) —
  `experiments/07_quantization.py` verifies the mechanism (precision loss is
  real, sizes shrink as expected) on a synthetic model and random data, same
  caveat section 6 already states for `experiments/01`'s `run_full()`. Don't
  cite a specific post-quantization accuracy number from this codebase
  either, for the same reason.

## 7. Extending this codebase — where new work goes

- **New scoring method** (e.g. a second-order/Hessian-based saliency): add
  a function to `saliency.py` matching the existing signature
  (`weight: [out,in,kh,kw] -> Tensor[out]`), register it in `_METHODS`, add
  a test in `tests/test_saliency.py` following the existing pattern
  (independent brute-force comparison where feasible, like
  `test_d2_matches_bruteforce_topk`).
- **New surgery type** (`prune_attention_heads` is the worked example now):
  add to `surgery.py`, validate the seam and raise before mutating anything
  (follow `prune_ffn_block`'s pattern exactly), add a test that checks
  *values* survive correctly post-surgery, not just shapes (see
  `test_d6_conv_values_are_correct_not_just_shape` for why shape-only tests
  aren't enough — that's literally how D6 shipped originally), and where
  feasible verify against a real HF model forward pass, not just plain
  `nn.Linear` (see `test_prune_attention_heads_against_a_real_hf_bert_model`).
- **New experiment**: follow the three-tier `--smoke` / `--tiny-check` /
  full pattern from section 4. Add the `--smoke` invocation to
  `.github/workflows/tests.yml` so CI actually exercises it.
- **Extending `prunelib.graph.DependencyGraph`**: `special_handlers`
  (`DependencyGraph(..., special_handlers={SomeAttentionClass: your_handler})`,
  paired with `LeafTracer([SomeAttentionClass])` so the traced graph actually
  sees that class as one node) is a real, tested attention-block delegation
  point now — see the `graph.py` walkthrough entry in section 4 and
  `test_graph.py::test_special_handler_prunes_attention_block_via_prune_attention_heads`
  for the worked example. It's scoped to the *starting* layer of a
  `get_pruning_group` call, not to automatic discovery mid-graph — if you
  want a specific attention class recognized as a prunable starting point
  without the caller naming it, that's the actual remaining gap, not the
  handler mechanism itself. `PruningGroup.mask()` / `.commit_and_compress()`
  (also added 2026-09-23) already wire `DependencyGraph`-based pruning into
  the two-phase mask-then-compress workflow — that's no longer open either.
- **New quantization method** (e.g. logarithmic or k-means-based
  quantization, which the thesis flags as better fits for skewed weight
  distributions than linear INT8 — see section 10.2): add a function to
  `quantization.py` matching the existing per-tensor signature (`Tensor ->
  Tensor`, or a small dataclass like `Int8Tensor` if it needs auxiliary
  metadata to dequantize), add a test following `test_quantization.py`'s
  pattern (brute-force comparison against the method's own formula where
  feasible, like `test_int8_quantize_matches_the_documented_formula`; a
  clipping/wrapping-at-the-boundary test if the format has a bounded range,
  like `test_fixed_point32_clips_out_of_range_values_instead_of_wrapping`).
  If it round-trips to a same-shape, same-dtype-castable tensor, also wire
  it into `quantize_model_`'s `_MODEL_METHODS`; if it needs a struct like
  INT8 does, document why it's excluded from `quantize_model_`, the way
  `quantize_model_`'s own docstring does for INT8.
- **A new CNN pruning criterion (K-means, SVD, second-order, anything
  matching the papers) belongs in `pruning_framwork_v4`, not here.** That
  repo is canonical for CNN methods and already has more coverage than
  this package's `prunelib`. Adding one here would create a second,
  divergent implementation of the same idea — see the scope note at the
  top of section 1.
- **Transformer-extension work** (wiring `experiments/03`'s detected
  redundant-head pairs into actual `prune_attention_heads` calls — now
  straightforward via a `special_handlers` handler, see above — deciding a
  real criterion rather than raw query-weight distance, validating
  `CoActivationScanner`'s threshold against real activations) is this repo's
  actual remaining job — see the gaps list in section 6 for what's still
  open there. Also check `transformer_pruning` first, since it's the newer,
  more actively developed line of the same work; avoid duplicating effort
  across both.

## 8. Onboarding checklist

**First: check the scope note at the top of section 1.** If you came here
to add or fix a CNN pruning criterion, you probably want
`pruning_framwork_v4` instead.

```bash
git clone https://github.com/thakerpragnesh/structured-pruning.git
cd structured-pruning
pip install -e ".[dev,vision-experiments,transformer-experiments]"
pytest tests/ -v                        # 75 tests
python experiments/00_demo.py           # full pipeline, seconds
python experiments/01_vgg_cifar10_sweep.py --smoke
python experiments/06_generic_pruning.py --smoke
python experiments/07_quantization.py   # pruning + all three quantization methods
```

Read section 3 of this document before touching `prunelib/saliency.py`,
`prunelib/surgery.py`, `prunelib/masking.py`, or `prunelib/graph.py`. Read section 6 before
writing anything that implies full VGG16/CIFAR-10 results or BERT
fine-tuning results are already produced by this codebase — the mask-then-
compress *mechanism* is tested and correct; specific accuracy numbers from
real training runs haven't been produced here yet.

## 9. Glossary

- **Saliency score** — a per-channel (or per-neuron) number estimating how
  much that channel contributes to the model's output; lower means "safer
  to prune."
- **Structural surgery** — physically resizing a layer (fewer real rows/
  columns in a real tensor) as opposed to zero-masking, which keeps the
  tensor the same size and just sets some entries to zero. Surgery makes
  the model smaller and faster; masking only makes it sparser.
- **Seam** — the shape contract between two connected layers (a conv's
  output channels matching the next conv's input channels, or an FFN's
  `fc1.out_features` matching `fc2.in_features`). "Seam mismatch" is this
  codebase's term for what happens when surgery on one layer isn't
  propagated to the layer downstream of it.
- **FFN block** — the two-Linear-layer (expand, then project back down)
  feed-forward sub-layer inside each Transformer block.
- **Co-activation** — two units (neurons, heads) firing on the same
  inputs; a redundancy signal independent of whether their weights look
  similar.
- **Firing rate** — the fraction of tokens/inputs a given unit is "active"
  on, used to exclude near-universal-firing units from co-activation
  analysis (see section 4).
- **Mask-then-compress** — the two-phase pruning design in `masking.py`:
  zero a channel's contribution via reparametrization first (reversible,
  same tensor shapes, safe to fine-tune against), and only physically
  resize the model once, at the end, via `compress_masked_conv_bn`. As
  opposed to one-shot surgery (`prune_conv_bn` called directly), which
  resizes immediately on every pruning decision.

## 10. Thesis material not reflected in this codebase

Pragnesh Thaker's Ph.D. thesis, *Pruning and Quantization Techniques for Deep
Neural Network Acceleration* (NITK Surathkal, July 2025), is the primary
source behind this repo — `prunelib`'s Max-k saliency, K-Means/distance-metric
scanning, and hybrid pruning all trace back to it, and the numbers in
`README.md` and `PUBLICATIONS.md` are drawn from it. This section covers what
the thesis describes that **isn't** implemented or documented anywhere in
this codebase: methods `prunelib` doesn't have (SVD pruning, kernel-level
pruning, FC-neuron pruning, custom regularization, quantization), the
original framework design the current code replaced, and results (external
benchmark comparison, future work) not summarized elsewhere in this repo's
docs. General ML background from the thesis — CNN architecture history
(LeNet through ResNet), optimizer survey (SGD through Adam), training
techniques (dropout, batch norm, cyclic LR), and the literature review of
other authors' pruning papers — is deliberately left out here as generic
textbook content rather than codebase-specific knowledge.

### 10.1 Pruning methods in the thesis that `prunelib` doesn't have

**SVD channel pruning** (thesis Ch. 4.5) — condenses each output channel's
kernel into a single scalar via a bit-weighted sum (`Condense_Value = Σ
2^(k1×3+k2) · W[i][j]`, thesis Eq. 4.10), runs Singular Value Decomposition
across channels, and uses the singular-value magnitude as the saliency
score — channels with the smallest singular values are pruned. This is a
genuinely different saliency signal from `saliency.py`'s L1/L2/Max-k (which
all operate directly on raw weight magnitudes, never touch SVD). Thesis
result (Table 4.12, VGG16/CIFAR10): 26.19% FLOPs / 36.49% parameter
reduction at a 1.31% accuracy drop — weaker than Max-k but still usable. Not
present anywhere in `prunelib`.

**Kernel-level pruning** (thesis Ch. 4.4, Algorithm 4.4/4.5, and Ch. 5.3 for
the similarity variant) — masks individual *kernels* within a channel (the
`[in_channel, kh, kw]` slice for one output channel) rather than the whole
channel. Because each kernel in a channel operates on a different input
channel and their outputs are summed, kernels can't simply be deleted and
the tensor reshaped the way channels can — the thesis's approach is to zero
the kernel's weights and skip it at compute time via a data selector, so the
*parameter count* doesn't shrink but FLOPs do. Thesis results
(VGG16/CIFAR10): saliency-based kernel pruning reached 27.74% FLOPs
reduction at a 0.96% accuracy drop (Table 4.6) before the 4th iteration
exceeded threshold; the similarity-based kernel variant (Table 5.6) was
markedly worse (19.57% FLOPs at a 1.49% drop — the thesis's own conclusion,
Ch. 5.3.1, is that grouping kernels by cross-channel similarity "is not
showing fruitful results" since kernels on different input channels
legitimately encode different information). Neither the masking-based
kernel pruning workflow nor a kernel-level saliency/similarity score exists
in `prunelib.saliency` or `prunelib.surgery` — both only operate at
whole-channel (or whole-head, for attention) granularity.

**Fully-connected / neuron pruning** (thesis Ch. 6.5) — prunes neurons out of
a classifier's dense layers using an L1 saliency score per neuron: the sum
of the absolute values of that neuron's incoming *and* outgoing connection
weights (not just one side, which is what makes this different from a naive
per-layer L1 prune). Applied to VGG16's classifier head, 5% of neurons were
removed per fully-connected layer per iteration, for 5 iterations, within a
1% accuracy-drop budget. `prunelib.surgery.prune_ffn_block` could physically
implement the resulting surgery (it already shrinks a Linear → activation →
Linear pair), but nothing in `prunelib` computes this specific
incoming+outgoing neuron saliency score, and `vgg.py` only ever touches
`model.features` (the conv stack) — `model.classifier` (VGG's 3 dense
layers) is never pruned by this codebase.

**Custom regularization (CSD — Custom Standard Deviation)** (thesis Ch. 6.1,
Eq. 6.1–6.4; this is `PUBLICATIONS.md` paper #2) — a training-time
group-regularization loss, not a pruning-time surgery step, so it's a
different kind of gap: `prunelib` has no training loop at all (that lived in
`archive/legacy_pipeline/train.py`, and even that has no regularization term
beyond a generic L1 penalty). The CSD loss for channel *i* in layer *l* is
the ratio of the channel's L1 norm to a per-channel "custom standard
deviation" — sum of absolute deviations from the channel's mean weight, not
the usual squared-deviation variance:

```
NC_i^l  = Σ_j Σ_k1 Σ_k2 |W[i,j,k1,k2]|
CSD_i^l = Σ_j Σ_k1 Σ_k2 |W[i,j,k1,k2] - Mean|
RegularizationLoss^l = Σ_i (NC_i^l / CSD_i^l)
```

A channel with low internal weight variation (small `CSD`) gets a *large*
regularization term relative to its norm, pushing it toward zero faster —
i.e. the loss is designed to penalize channels whose weights are already
nearly redundant with each other, not just small in magnitude. Fine-tuning
with this loss before saliency-based pruning improved VGG16's L1-pruning
result from 53.80%/39.04% (no regularization) to 61.91%/46.14%
FLOPs/parameter reduction at a comparable ~0.95% accuracy drop (thesis Table
6.4). Nothing in `prunelib` or `archive/legacy_pipeline` implements this
loss.

### 10.2 Quantization — now implemented in `prunelib/quantization.py` (2026-09-23)

The thesis title is "Pruning *and Quantization* Techniques," and Ch. 6.6
covers three quantization approaches applied *after* pruning, as a separate
compression stage. All three are now in `prunelib/quantization.py` — see the
module walkthrough entry in section 4 for the API, and section 6 for what's
still an honest gap (none have been run against a real fine-tuned model to
reproduce the thesis's own accuracy-drop numbers below). `archive/legacy_pipeline`
and the pre-existing experiments (00 through 06) are unaffected; this repo's
scope is otherwise still pruning-focused, with quantization as a separate,
final stage you apply to whatever model pruning produced.

**Float32 → Float16** (`quantize_float16`/`dequantize_float16`). A direct
cast of every weight from 32-bit to 16-bit floating point (1 sign bit, 5
exponent bits, 10 mantissa bits — range ±10^±5, 3–4 decimal digits of
precision). Applied to the final hybrid-pruned VGG16 model, the thesis
measured this as halving memory footprint for a 1.34% accuracy drop (Ch.
6.6.4) — the thesis reports this as a good trade since AI accelerators and
mobile processors widely support Float16 inference natively.

**Float32 → Fixed-Point32** (`quantize_fixed_point32`). Parameters are cast
into a fixed 1-sign-bit / 3-integer-bit / 28-fractional-bit format; any
value outside the representable range is clipped rather than wrapped
(`test_fixed_point32_clips_out_of_range_values_instead_of_wrapping` pins
this specifically). The thesis measured this as markedly worse than Float16
— a 3.26% accuracy drop on the same pruned VGG16 — because, unlike floating
point, a fixed-point format can't dynamically rescale its exponent to fit
the actual distribution of a layer's weights, so precision is wasted or
values are lost at the tails. One simplification from the thesis worth
flagging: the thesis describes a per-layer weight *normalization* step
before the fixed-point cast; `quantize_fixed_point32` here is the cast only
(round to the nearest representable value, then clip) with no normalization
step — closer to the "cast, don't rescale" comparison the thesis draws
against Float16 than a full reproduction of Ch. 6.6.2's exact procedure. If
you need the normalization step too, it belongs as a separate, composable
function (normalize, then call `quantize_fixed_point32` on the result), not
folded into this one.

**Linear INT8 quantization** (`quantize_int8_linear`/`dequantize_int8_linear`,
Eq. 6.7–6.10) — the standard scale/zero-point affine mapping, which the
thesis includes as a reference technique rather than one it ran end-to-end
on the pruned VGG16:

```
scale = (r_max - r_min) / (q_max - q_min)
zero_point = round(q_min - r_min / scale)
x_int8 = round(x_fp32 / scale) + zero_point
```

with dequantization `x_fp32 ≈ scale × (x_int8 - zero_point)`, and `[q_min,
q_max] = [-128, 127]` for signed INT8 — `test_int8_quantize_matches_the_documented_formula`
is an independent brute-force check against exactly this formula, in the
style of `saliency.py`'s brute-force tests. The one addition beyond the
thesis's formula: a constant tensor (`r_max == r_min`) would divide by zero
in it, so that case is handled separately (see `quantize_int8_linear`'s own
docstring for how). The thesis notes the general formula assumes a roughly
uniform weight distribution and flags logarithmic or k-means-based
quantization as better fits for skewed distributions — neither is
implemented here; see section 7 if you want to add one.

`evaluate.estimate_size_bytes(module, bits_per_param)` is the bit-width/
precision helper this section used to say `evaluate.py` was the natural
home for, now added alongside `count_params` and `measure_latency`.

### 10.3 The original four-module framework (thesis Ch. 3) — useful mainly as the defect record's primary source

The thesis formalizes, in pseudocode, the exact framework `GITHUB_AUDIT.md`
and this document's section 1/3 describe from the outside. It's organized
as four modules — **Load Model** (save/load/create-custom-model),
**Initialize Pruning** (convolution-layer indexing, prune-count-list
generation), **Facilitate Pruning** (mask matrix creation, apply-pruning,
deep-copy-non-zero-channels), **Train Model** (fit-one-cycle, freeze-model,
freeze-selected-layer) — and this codebase's `prunelib` is a from-scratch,
corrected reimplementation of the same four responsibilities, not a
refactor of this code.

Two pieces of this pseudocode are worth knowing about specifically because
they *are* the bugs section 3 documents, now given formal shape:

- **Algorithm 3.9 ("Pruning Algorithm Class")** subclasses
  `torch.nn.utils.prune.BasePruningMethod` and calls
  `PruningAlgorithm.apply(module, name)` to mask channels. This is the exact
  design `masking.py`'s docstring calls out as the root cause of the
  `layer_number`-global bug (a `compute_mask` callback reconstructing a mask
  from global state on every call) — `prunelib.masking.mask_channels`
  replaced this whole pattern with one
  `torch.nn.utils.prune.custom_from_mask` call using a mask computed once,
  up front.
- **Algorithm 3.10 ("Deep Model Copy Channel wise")** is, almost verbatim,
  defect D6: it iterates output channels, checks `norm of source channel
  parameters is not zero`, and copies surviving channels into the
  compressed model at a running `out_ch_new` counter that increments only
  inside the `if not zero` branch — the exact hand-maintained
  destination-index pattern `surgery.py`'s module docstring says caused the
  original `deep_model_copy_channelwise` bug (the original never
  incremented, so every survivor overwrote index 0). Seeing the thesis's
  own pseudocode for this algorithm confirms this document's account of D6
  is describing a real, specified algorithm, not just a colloquial
  retelling.

The framework's **prune-count-list algorithm** (thesis Algorithm 3.7) also
formalizes something `prunelib` does differently on purpose: the thesis
schedules pruning per *block* (VGG16's `[2,2,3,3,3]` conv-layer grouping)
with a linearly increasing prune probability across blocks (deeper blocks
get pruned more aggressively, on the stated grounds that they carry more
redundancy) — `prunelib.vgg.mask_vgg_layer` instead takes an explicit
`prune_fraction` per call and applies it uniformly to whichever layer the
caller names, with no built-in per-block schedule. Anyone wanting to
reproduce the thesis's exact schedule in this codebase would need to
compute that per-layer fraction list themselves and drive `mask_vgg_layer`
with it in a loop — nothing in `prunelib` generates it automatically.

### 10.4 External benchmark comparison, and the thesis's own future work

**Head-to-head with published pruning methods (thesis Table 6.12,
VGG16/CIFAR10).** `README.md` and `PUBLICATIONS.md` report this thesis's own
numbers, but not how they stack up against other published pruning
techniques — the thesis includes that comparison directly:

| Method | Baseline acc. | Pruned acc. | Acc. drop | Param ↓ | FLOPs ↓ |
| --- | --- | --- | --- | --- | --- |
| SSS (2018) | 93.96% | 93.02% | 0.94% | 73.8% | 41.6% |
| GAL (2019) | 93.96% | 93.77% | 0.19% | 77.6% | 39.6% |
| HRank (2020) | 93.96% | 93.43% | 0.53% | 82.9% | 53.5% |
| CHIP (2022) | 93.96% | 93.86% | 0.10% | 81.6% | 58.1% |
| **Max-3 (this thesis)** | 93.28% | 92.37% | 0.97% | 46.14% | 61.91% |
| **K-Means (this thesis)** | 93.28% | 92.29% | 0.99% | 40.00% | 40.00% |
| **Hybrid (this thesis)** | 93.28% | 92.47% | 0.91% | 35.00% | 58.34% |

The thesis's own reading of this table: SSS/GAL/HRank/CHIP remove a much
larger fraction of *parameters* (73–83%) than this thesis's methods do, but
this thesis's Max-3 and Hybrid methods reach comparable or better *FLOPs*
reduction (58–62% vs. 40–58%) with a similar accuracy drop — i.e. the
methods here are relatively more effective at cutting compute than at
cutting raw parameter count, which tracks with Max-3/Hybrid being
channel-pruning methods concentrated on convolutional layers (per Ch. 1.5,
convs are ~5% of VGG16's parameters but ~98% of its FLOPs). None of this
comparison table, or the SSS/GAL/HRank/CHIP citations, appear anywhere in
this repo.

**The thesis's stated future work (Ch. 7.2)**, distinct from this document's
section 6 ("Known gaps") and section 7 ("Extending this codebase"), which
describe gaps in the *code*, not the author's own research roadmap:

- Improving SVD-based reconstruction precision/stability post-pruning, to
  reduce retraining cost while preserving accuracy.
- More robust, possibly learned, evaluation metrics for channel similarity
  and saliency quality — the thesis notes existing metrics (L1 norm,
  distance metrics) may not fully capture a channel's functional
  redundancy.
- Extending pruning/quantization beyond CNNs to transformer architectures
  and task-specific models (medical imaging, autonomous driving, edge
  deployment).
- **Adaptive hybrid pruning** that dynamically balances saliency-based vs.
  similarity-based pruning during training/fine-tuning based on model
  complexity, data distribution, and hardware constraints — a more
  automated version of what Algorithm 6.1 (the hybrid-pruning loop, thesis
  Ch. 6.3) does by hand today.
- Hardware-aware optimization: pruning/quantization decisions driven by
  target-device characteristics (memory access patterns, parallelism,
  precision support) rather than architecture-agnostic thresholds.
- Mixed-precision and quantization-aware training, as a follow-on to the
  post-training quantization covered in section 10.2 above.

This roadmap is worth reading before proposing new work in either
`structured-pruning` or `transformer_pruning` (per section 7) — it's the
author's own sense of what's still open, independent of what's specifically
unfinished in this particular codebase.
