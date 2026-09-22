# structured-pruning

**`structured-pruning` is the library-first repo for pruning modern
(Transformer) architectures** — a pip-installable `prunelib` package plus
tested experiments, not a config.ini-driven pipeline you edit and run in
place. For CNN channel-pruning from my Ph.D. work — Max-3 saliency,
L1/SVD/K-means criteria, hybrid multi-criterion sequencing — the canonical
repo is [`pruning_framwork_v4`](https://github.com/thakerpragnesh/pruning_framwork_v4)
instead: that repo has all five criteria working and verified, where this
one only has Max-k/L1/L2/random. This repo's actual job is the two things
below.

## What this repo is for

**1. Extending the same saliency and redundancy ideas to Transformers.**
`experiments/02` through `05` and the corresponding pieces of `prunelib/`
apply Max-k-style saliency to BERT FFN neurons, and Manhattan/Euclidean/
Cosine distance to attention heads — the CNN methods don't have a
Transformer equivalent anywhere else on my account. (Newer, more actively
developed Transformer work now lives in
[`transformer_pruning`](https://github.com/thakerpragnesh/transformer_pruning);
this repo's Transformer experiments are the earlier version of that line
of work.)

**2. A tested reference implementation of the mask-then-compress design.**
`prunelib/masking.py` implements the two-phase workflow — mask channels
during pruning via `torch.nn.utils.prune.custom_from_mask`, physically
compress once at the end — cleanly and with a real test suite (36 tests,
CI on every push). If you're implementing that pattern elsewhere
(including in `pruning_framwork_v4`, which arrived at a similar design
independently), this is a working, tested reference for it.

**What's now redundant:** `archive/legacy_pipeline/` (moved there 2026-09-21,
formerly `legacy_pipeline/` at the repo root — see `LEGACY_PIPELINE_MIGRATION.md`)
was a corrected rebuild of `pruning_framwork_v4`'s original driver scripts,
done in parallel with this package. Since then, those same scripts got fixed
directly in `pruning_framwork_v4` itself — and more completely, since that
fix added K-means/SVD/hybrid sequencing that `legacy_pipeline` (built on this
repo's Max-k/L1/L2/random-only `prunelib`) never had. It moved under
`archive/` rather than staying at the top level to make that explicit: this
repo is `prunelib` plus the Transformer experiments first, with
`legacy_pipeline` kept only for its test suite and as a record of the
mask-then-compress design being applied to a full VGG16 — not as something
to run instead of `pruning_framwork_v4`, and not part of the library's public
surface (`from prunelib import ...` never touches it).

See `GITHUB_AUDIT.md` for the original bug-by-bug history of why this
package was built, and `KT.md` if you're extending the code here
specifically (the Transformer experiments or `prunelib/masking.py`).

## Install

```bash
pip install -e .
```

## Quickstart

```bash
python experiments/00_demo.py
```

```
prune fraction: 50%  (64 -> 32 channels)
params:  38,848 -> 19,456  (49.9% reduction)
latency: 5.518ms -> 2.591ms  (2.13x)
```

Every experiment has a `--smoke` flag that runs the full pipeline on
synthetic data / randomly-initialized models in seconds, with no downloads —
useful for verifying the machinery before spending GPU time on the real run,
and what CI runs on every commit.

## API

```python
from prunelib import compute_score, select_prune_indices, prune_conv_bn, prune_ffn_block

scores = compute_score(conv.weight, method="max_k", k=3)   # or "l1", "l2", "random"
keep_idx = ...  # complement of select_prune_indices(scores, n_to_prune)
new_conv, new_bn, new_next_conv = prune_conv_bn(conv, keep_idx, bn=bn, next_conv=next_conv)
```

For Transformer FFN blocks:

```python
from prunelib import prune_ffn_block
new_fc1, new_fc2 = prune_ffn_block(fc1, fc2, keep_idx)
```

`compute_score` also takes a Linear weight `[out_features, in_features]`
directly (no reshaping to a fake conv tensor needed) — e.g. `compute_score(
fc1.weight, method="max_k", k=3)`.

For attention heads:

```python
from prunelib import prune_attention_heads
new_q, new_k, new_v, new_out = prune_attention_heads(
    query, key, value, output, keep_heads=keep_heads, num_heads=num_heads,
)
```

For redundancy scanning (channels, attention heads, or arbitrary activation
patterns):

```python
from prunelib import pairwise_distance_matrix, CoActivationScanner
dist = pairwise_distance_matrix(vectors, metric="manhattan")  # or "euclidean", "cosine"
scanner = CoActivationScanner(firing_rate_ceiling=0.9)
similarity = scanner.jaccard(activation_mask)
```

Everything above needs the caller to already know the seam (`next_conv=`,
which two Linears form an FFN block). `prunelib.graph.DependencyGraph`
doesn't — it traces an arbitrary model with `torch.fx` and works out which
other layers a prune decision couples to (a residual/skip partner, a
downstream conv/Linear, a `cat`, a flatten into a classifier head) on its
own:

```python
from prunelib import DependencyGraph

dep = DependencyGraph(model, example_input)          # traced once, reused for as many prunes as you like
group = dep.get_pruning_group(model.layer2[0].conv2, keep_idx)
group.prune()                                         # mutates model in place -- see graph.py's docstring for why
```

It handles Conv2d/BatchNorm/Linear chains coupled by elementwise add,
`torch.cat`, the conv-output-flattened-into-Linear boundary, and depthwise
convs — see `prunelib/graph.py`'s module docstring for exactly what it does
and doesn't handle (it does not attempt *automatic* attention-block
discovery; a `special_handlers` hook lets you register one manually, see
below).

`prune_model` is the one-call convenience on top of `DependencyGraph` —
score, select, and prune a layer plus everything it's coupled to, for any
model the graph can trace (not just VGG's flat `.features`):

```python
from prunelib import prune_model

group = prune_model(model, example_input, model.layer2[0].conv2, prune_fraction=0.3, method="l1")
```

`DependencyGraph` also has a two-phase mode, mirroring the mask-then-compress
workflow below at the level of a whole dependency group instead of one layer:

```python
group = dep.get_pruning_group(layer, keep_idx)
group.mask()                 # zero the group's channels, shapes unchanged -- safe to fine-tune against
# ... fine-tune / evaluate ...
group.commit_and_compress()  # bake the zeros in and physically shrink everything the group touches
```

And for attention blocks specifically -- which `DependencyGraph` can't wire
up on its own, since `prune_attention_heads` needs four Linears rewired
together at once, not one module resized -- `special_handlers` registers a
handler that takes over when a given block type is the *starting* layer of
a `get_pruning_group` call:

```python
from prunelib import DependencyGraph, LeafTracer

def handle_attention_block(block, keep_heads):
    block.query, block.key, block.value, block.out = prune_attention_heads(
        block.query, block.key, block.value, block.out,
        keep_heads=keep_heads, num_heads=block.num_heads,
    )
    block.num_heads = keep_heads.numel()
    return block

dep = DependencyGraph(
    model, example_input,
    tracer=LeafTracer([MyAttentionBlockClass]),  # needed so the block traces as one node
    special_handlers={MyAttentionBlockClass: handle_attention_block},
)
dep.get_pruning_group(model.attn, keep_heads).prune()
```

## Results

CNN results below are the published, peer-reviewed figures (see
`PUBLICATIONS.md`). Latency is a fresh measurement from `experiments/00_demo.py`
on this package's implementation — **run it on your own machine before citing
a specific multiplier**; wall-clock speedup depends on hardware, batch size,
and how much of the network is pruned, and will not be identical across runs.

| Method | Model | Params ↓ | FLOPs ↓ | Acc. drop | Venue |
|---|---|---|---|---|---|
| Max-3 saliency | VGG16 / CIFAR-10 | 46.1% | 61.9% | <1% | IEEE Access 2024 |
| Max-3 saliency | ResNet56 | 35.2% | 35.2% | <1% | IEEE Access 2024 |
| CSD group regularization | VGG16 | 46.1% | 61.9% | 0.95% | IEEE Access 2024 |
| Manhattan K-Means | VGG16 | 35.2% | 49.1% | 0.98% | IEEE Access 2024 |
| Hybrid ordering (channel→channel→kernel) | VGG16 / Intel IC | 58.4% | 42.8% | 4.4% | IEEE ICCCNT 2023 |

| Component | Status |
|---|---|
| `saliency.py` — Max-k/L1/L2/random, Conv2d and Linear weights | Unit-tested, 10/10 passing |
| `surgery.py` — Conv/BN/FFN/attention-head structural surgery | Unit-tested, verified against a real HF BERT forward pass |
| `graph.py` — `torch.fx`-traced dependency resolution (add/cat/flatten/depthwise), `prune_model`, two-phase `PruningGroup`, `special_handlers` | Unit-tested; `experiments/06 --tiny-check` verified against a real `torchvision.models.resnet18`, including cascading through a whole residual stage |
| `quantization.py` — Float16/INT8/Fixed-Point32 post-training quantization | Unit-tested, including a brute-force formula check for INT8 and a clipping-not-wrapping check for Fixed-Point32; `experiments/07` runs the full pipeline on synthetic data, not yet against a real fine-tuned model |
| `scanners.py` — distance metrics + co-activation | Unit-tested |
| `experiments/01` VGG-CIFAR10 sweep | `--tiny-check` runs the real `torchvision.models.vgg16` class through real `prune_vgg_layer`/`prune_conv_bn` calls end to end (verified, ~3-4 min on CPU); full run (real CIFAR-10 + ImageNet weights) not yet executed |
| `experiments/02` BERT FFN sweep | Pipeline verified in `--smoke` against real `transformers` model classes; full run not yet executed |
| `experiments/03` head redundancy | Pipeline verified in `--smoke`; full run needs a fine-tuned checkpoint |
| `experiments/05` ordering | Scoring-perturbation mechanism verified; full accuracy-drop reproduction not yet run |

That last column is deliberately explicit: the CNN numbers above are the
published, real results. The Transformer-extension experiments are new
infrastructure — correct and tested, but not yet run against real fine-tuned
models. Don't claim results those runs haven't produced yet.

## Two-phase pruning: mask, then compress

The original design intent behind the old driver scripts (confirmed
directly, not inferred) was: mask the weights being pruned during the
schedule, and once it's done, build a compressed model by copying the
unmasked (surviving) weights across. `prunelib.masking` implements exactly
that, using `torch.nn.utils.prune.custom_from_mask` instead of the
hand-rolled `BasePruningMethod` subclasses that caused the original bugs
(see `GITHUB_AUDIT.md` section 11 and `LEGACY_PIPELINE_MIGRATION.md`):

```python
from prunelib import mask_channels, commit_mask, compress_masked_conv_bn, select_prune_indices, compute_score

# Phase 1, safe to call repeatedly across a fine-tuning schedule:
scores = compute_score(conv.weight, method="max_k", k=3)
mask_channels(conv, select_prune_indices(scores, n_to_prune))
# ... fine-tune / evaluate with the mask active as many times as you like ...

# Phase 2, once, when you're ready to commit:
commit_mask(conv)
new_conv, new_bn, new_next_conv, keep_idx = compress_masked_conv_bn(conv, bn=bn, next_conv=next_conv)
```

For a whole VGG16, `archive/legacy_pipeline` wraps this into a complete
pipeline — see `LEGACY_PIPELINE_MIGRATION.md` for how to run it and exactly
what it replaces.

## Quantization

`prunelib.quantization` implements the three post-training quantization
methods from the thesis this repo is based on (Ch. 6.6) as a separate
compression stage, applied *after* pruning:

```python
from prunelib import quantize_model_, quantize_int8_linear, dequantize_int8_linear, estimate_size_bytes

# Float16 or Fixed-Point32: in-place, round-trips every Linear/Conv2d/BatchNorm
# weight and bias through the target precision, cast back to float32.
quantize_model_(pruned_model, method="float16")
quantize_model_(pruned_model, method="fixed_point32", integer_bits=3, fractional_bits=28)

# Linear INT8: per-tensor scale/zero-point, since realizing its actual
# memory savings means storing int8 values plus that metadata, not a
# same-shape float tensor you can substitute in place.
q = quantize_int8_linear(conv.weight)
approx_weight = dequantize_int8_linear(q)

estimate_size_bytes(pruned_model, bits_per_param=16)  # compare precision options without writing a checkpoint per option
```

See `experiments/07_quantization.py` for the whole pipeline (`prune_model`
then all three methods) end to end, and `KT.md` section 10.2 for the
thesis's own numbers (Float16: 1.34% accuracy drop for half the memory;
Fixed-Point32: 3.26%, markedly worse, since a fixed exponent can't adapt to
a layer's actual weight distribution) — not yet reproduced against a real
fine-tuned model in this codebase, only verified mechanically so far.

## Repository layout

```
prunelib/
    saliency.py   Max-k (correct), L1, L2, random -- Conv2d or Linear weights
    surgery.py    conv/BN/FFN/attention-head structural surgery
    masking.py    two-phase mask-then-compress workflow (torch.nn.utils.prune)
    graph.py      torch.fx dependency resolution -- generic add/cat/flatten/depthwise
                  surgery; prune_model, PruningGroup.mask()/.commit_and_compress(),
                  special_handlers/LeafTracer (attention-block hook)
    quantization.py  Float16 / linear INT8 / Fixed-Point32 post-training quantization
    vgg.py        VGG wiring: build_vgg16, mask_vgg_layer, compress_masked_vgg
    scanners.py   distance metrics + co-activation scanning
    evaluate.py   parameter counts, measured latency, estimated size at a bit-width
experiments/
    00_demo.py                  runs in seconds
    01_vgg_cifar10_sweep.py     Max3 vs L1 vs L2 vs random
    02_bert_sst2_sweep.py       FFN pruning on BERT
    03_head_redundancy.py       head similarity across layers
    04_coactivation.py          activation-based redundancy
    05_ordering.py              does the CNN ordering result transfer?
    06_generic_pruning.py       DependencyGraph on a real ResNet-18, no seam passed by hand
    07_quantization.py          prune_model() + all three quantization methods, one pipeline
archive/
    legacy_pipeline/            corrected replacement for the six original
        config.py, data.py, model.py,   driver scripts -- now redundant with
        train.py, pipeline.py           pruning_framwork_v4, see
                                         LEGACY_PIPELINE_MIGRATION.md
tests/          75 tests, each naming the defect or behavior it guards against
```

## Citation

See `CITATION.cff`, or `PUBLICATIONS.md` for full BibTeX entries for all six
papers this code implements or extends.

## License

MIT — see `LICENSE`.
