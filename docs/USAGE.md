# Usage guide

The full `prunelib` API, with examples. The [README](../README.md) has the
overview and quickstart; [`ARCHITECTURE.md`](ARCHITECTURE.md) explains the
design decisions behind it.

## Core API

```python
from prunelib import compute_score, keep_indices, prune_conv_bn, prune_ffn_block

scores = compute_score(conv.weight, method="max_k", k=3)   # or "l1", "l2", "random"
keep_idx = keep_indices(scores, n_to_prune)                 # complement of select_prune_indices(scores, n_to_prune)
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
squeeze-and-excitation gate, a downstream conv/Linear, a `cat`, a flatten
into a classifier head) on its own, whether activations are written as
modules (`nn.ReLU()`) or functions (`F.relu(x)`):

```python
from prunelib import DependencyGraph

dep = DependencyGraph(model, example_input)          # one graph, reused for as many prunes as you like
group = dep.get_pruning_group(model.layer2[0].conv2, keep_idx)
group.prune()                                         # mutates model in place -- see group.py's docstring for why
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

### Extending `prunelib` without editing it

Every pluggable part — saliency scorers, selection rules, distance metrics,
quantization methods, the layer types `DependencyGraph` can prune, and the
ops it can carry a prune through — is a public registry. Register from your own code, and every entry point that
dispatches on it picks the new entry up:

```python
from prunelib import (
    ChannelRole, DistanceMetric, ModuleRule, propagate_elementwise, register_distance_metric,
    register_module_rule, register_op_propagator, register_quantization_method,
    register_saliency_method,
)

@register_saliency_method("taylor")            # now valid as method="taylor" in compute_score,
def taylor_saliency(weight, **kwargs): ...     # select_prune_indices_by_method, prune_model, vgg.py

register_distance_metric("chebyshev", DistanceMetric(  # scanning and K-Means both accept it
    pairwise=lambda x, y: torch.cdist(x, y, p=float("inf")),
    centroid=lambda m: (m.max(0).values + m.min(0).values) / 2,
))

register_quantization_method("log2", my_log2_round_trip)   # quantize_model_(model, method="log2")

class Conv1dRule(ModuleRule):                  # teach DependencyGraph a new layer type
    def role(self, conv): return ChannelRole.MIXING
    def rebuild(self, conv, prune_out, prune_in): ...
    # optional, for a type whose output channels aren't dim 0 of its weight (ConvTranspose2d):
    # output_weight(self, conv) -> weight as [out, in, ...]; mask(self, conv, prune_out); commit_mask(self, conv)
    # optional, for a type whose tensors keep channels last rather than on dim 1 (as Linear's do):
    # channel_dim(self, module) -> -1
register_module_rule(nn.Conv1d, Conv1dRule())  # or DependencyGraph(..., module_rules={...}) for one graph
                                               # prune_model then scores and prunes Conv1d too

register_op_propagator(torch.maximum, propagate_elementwise)  # a residual merged with torch.maximum couples like `+`
                                                              # (or DependencyGraph(..., op_propagators={...}));
                                                              # propagate_channelwise for your own activation function
```

Every `method=` argument also takes the implementation itself, for a
one-off not worth registering: a selector `(weight, prune_amount) ->
prune_idx` for `prune_model` / `prune_vgg_layer` / `mask_vgg_layer`, a
scorer for `compute_score`, a same-shape round-trip for `quantize_model_`.
Whatever answers is held to its contract -- one score per output channel,
exactly `prune_amount` distinct channels, the parameter's own shape -- and
a plugin that breaks it raises, naming itself. Every pruned module is rebuilt by
`surgery.py`'s `slice_conv2d` / `slice_depthwise_conv2d` / `slice_linear` /
`slice_batchnorm`, each of which returns a copy of the original with only
its channels resized: same class (a `Conv2d` subclass stays that
subclass), configuration, device, dtype, train/eval mode and
`requires_grad`. `tests/test_extension_points.py` has a worked example of
each extension point.


## Transformer techniques (`prunelib.transformer`)

Merged from `transformer_pruning`. Where the core API above is stateless
tensor-in/tensor-out, this subpackage works on a whole model (any
HuggingFace BERT/RoBERTa-layout encoder or task model, via
`resolve_adapter`), and adds the pieces weight-only criteria can't see:

| Area | What it adds |
|---|---|
| **Calibration** | `FFNCalibrator` (per-neuron activation moments), `HeadCalibrator` (per-head contribution moments + Michel et al. 2019 gradient importance), `ActivationRecorder` (per-token firing patterns). Padding is masked out everywhere. |
| **FFN criteria** | `SaliencySelector` (any scorer: `TopKMagnitudeScorer`/Max-3, `LpNormScorer`, `CSDScorer`, or any core method via `RegisteredScorer`), `InOutNormSelector`, `ActivationAwareSelector`, `TwinRedundancySelector` (co-activation twins), `WeightClusterRedundancySelector` (K-means on `W_in` direction) |
| **Head criteria** | `RedundantHeadSelector` (OV-circuit or query cosine), `OVNormHeadSelector`, `ActivationAwareHeadSelector`, `GradientHeadSelector` |
| **Cheaper cuts** | Bias compensation (fold a removed neuron's/head's mean output into the bias: zero-ablation becomes mean-ablation) and merging (fold a redundant neuron's output column into its twin) |
| **Model-wide budgets** | `allocation="uniform"` (every layer loses the same fraction) or `"global"` (every unit ranked against every other, with per-layer floors), or your own `AllocationStrategy` |

```python
from prunelib.transformer import (
    ActivationAwareSelector, FFNCalibrator, GradientHeadSelector, HeadCalibrator,
    prune_model_attention, prune_model_ffn,
)

# Heads first: the head cut changes every downstream FFN's input distribution.
cal = HeadCalibrator(model)
prune_model_attention(
    model, GradientHeadSelector(prune_percent=25), allocation="global", normalize="l2",
    stats_by_layer=cal.collect_many(batches),
    gradient_importance_by_layer=cal.collect_gradient_importance(labelled_batches),
    compensate_bias=True,
)
# Then calibrate the FFN on the head-pruned model and cut it.
prune_model_ffn(
    model, ActivationAwareSelector(prune_percent=40), allocation="global",
    stats_by_layer=FFNCalibrator(model).collect_many(batches), compensate_bias=True,
)
```

`python experiments/08_transformer_techniques.py --smoke` runs exactly this
(plus a twin merge) on a random-init HF BERT in seconds.

**Shared with core:** CSD is one implementation (`prunelib.compute_score(w,
"csd")` and `CSDScorer` are the same function, so CSD now works for CNN
channel masking too), and head distances use the core
`pairwise_distance_matrix` registry. The dependency runs one way only:
`prunelib.transformer` imports core, never the reverse
(`tests/transformer/test_package_layout.py` enforces it, along with the
subpackage layering).

Two things share a name across the two layers on purpose. The core
`prunelib.prune_attention_heads(q, k, v, o, keep_heads, num_heads)` is the
tensor-level primitive and takes heads to **keep**. The model-level
`prunelib.transformer.prune_attention_heads(model, layer, head_indices)`
takes heads to **drop** and also updates the layer's head-count
bookkeeping. Likewise, core `"max_k"` treats a Linear weight as a 1x1 conv,
so on an FFN it equals L1, while `TopKMagnitudeScorer(k)` sums each
neuron's k strongest incoming connections.

The research numbers for these criteria have **not** been produced yet.
The harness that produces them is `experiments/research/`: GLUE tasks ×
seeds × criteria × budgets, resumable on free Kaggle/Colab GPUs, with
paired mean ± std tables and plots. See
[`RUNNING_EXPERIMENTS.md`](RUNNING_EXPERIMENTS.md) and
`notebooks/pruning_study_kaggle_colab.ipynb`.


## Two-phase pruning: mask, then compress

Mask the weights being pruned during the schedule, and once it's done,
build a compressed model by copying the unmasked (surviving) weights
across. `prunelib.masking` implements this with
`torch.nn.utils.prune.custom_from_mask` and a mask computed once, up front
(see [`ARCHITECTURE.md`](ARCHITECTURE.md) for why):

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
pipeline — see [`LEGACY_PIPELINE_MIGRATION.md`](LEGACY_PIPELINE_MIGRATION.md) for how to run it and exactly
what it replaces.


## Quantization

`prunelib.quantization` implements the three post-training quantization
methods from the thesis this repo is based on (Ch. 6.6) as a separate
compression stage, applied *after* pruning:

```python
from prunelib import quantize_model_, quantize_int8_linear, dequantize_int8_linear, estimate_size_bytes

# Float16 or Fixed-Point32: in-place, round-trips every floating-point
# parameter through the target precision, cast back to float32
# (module_types=prunelib.quantization.QUANTIZABLE_MODULE_TYPES for the thesis's Linear/Conv2d/BatchNorm scope).
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
then all three methods) end to end, and [`KT.md`](KT.md) section 10.2 for the
thesis's own numbers (Float16: 1.34% accuracy drop for half the memory;
Fixed-Point32: 3.26%, markedly worse, since a fixed exponent can't adapt to
a layer's actual weight distribution) — not yet reproduced against a real
fine-tuned model in this codebase, only verified mechanically so far.
