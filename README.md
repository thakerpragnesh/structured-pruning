# structured-pruning

**`prunelib`: a tested, pip-installable library for structured pruning and
quantization of CNNs and Transformers**, with reproducible experiments built
on it. It implements the Max-3 saliency criterion and K-Means channel
selection with Manhattan, Euclidean and cosine distance from the author's
IEEE Access 2024 papers, adapts the CSD idea from the second paper into a
one-shot score, and extends these criteria to BERT-style Transformers (see
[`PUBLICATIONS.md`](PUBLICATIONS.md)).

## Capabilities

- **Saliency and redundancy criteria.** Max-k, L1, L2, CSD-based and random
  scoring for any `[out, in, *kernel]` weight (Conv1d/2d/3d, Linear);
  K-Means channel selection under Manhattan, Euclidean or cosine distance;
  co-activation and pairwise-distance redundancy scanning.
- **Architecture-agnostic structural pruning.** `DependencyGraph` traces any
  `torch.fx`-traceable model and works out which layers a prune couples to
  (residual adds, squeeze-and-excitation gates, `cat`, flatten into a
  classifier, depthwise convs), then physically shrinks them. Verified on
  torchvision ResNet-18 and MobileNetV3-Small.
- **Mask, then compress.** Mask channels during fine-tuning, then commit and
  physically shrink once at the end; the compressed model's output matches
  the masked model's exactly.
- **Transformer pruning (`prunelib.transformer`).** Calibration-aware FFN-neuron
  and attention-head selection for HuggingFace BERT/RoBERTa-layout models,
  bias compensation, twin-neuron merging, and uniform or global budgets.
- **Post-training quantization.** Float16, linear INT8 and Fixed-Point32,
  applied after pruning, plus size and latency measurement.
- **Extensible without forking.** Scorers, selection rules, distance metrics,
  quantization methods, prunable layer types and graph ops are all public
  registries.
- **Tested.** 390 tests (3 need a CUDA GPU), CI on Python 3.10–3.13, and
  every experiment has a `--smoke` mode that runs in seconds.

CNN pruning with SVD and hybrid multi-criterion sequencing lives in the
companion repo
[`pruning_framwork_v4`](https://github.com/thakerpragnesh/pruning_framwork_v4).

## Install

```bash
pip install -e .                       # core: torch only
pip install -e ".[analysis]"           # + pandas, for NetworkSaliencyScanner.scan()
pip install -e ".[research]"           # + datasets/accelerate/sklearn/matplotlib, for experiments/research
pip install -e ".[dev,transformer-experiments,vision-experiments,research]"   # everything CI runs
```

On an NVIDIA GPU, install a CUDA build of PyTorch first, choosing the CUDA
version your driver supports (`nvidia-smi`; see pytorch.org), then the package:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -e ".[dev,transformer-experiments,vision-experiments]"
```

Every experiment runs on the GPU when there is one and prints `device: cuda`
(`--device cpu` forces the CPU). For Colab or Kaggle, open
`notebooks/run_on_colab_kaggle.ipynb`. It clones the repo, runs the tests
and smoke runs on the GPU, then runs a resumable VGG16 / CIFAR-10 sweep.

## Quickstart

```bash
python experiments/00_demo.py                          # prune a conv block, measure params and latency
python experiments/06_generic_pruning.py --tiny-check  # DependencyGraph on a real ResNet-18
python experiments/08_transformer_techniques.py --smoke  # calibrated head + FFN pruning on BERT
```

```
prune fraction: 50%  (64 -> 32 channels)
params:  38,848 -> 19,456  (49.9% reduction)
latency: 5.518ms -> 2.591ms  (2.13x)
```

Experiments that touch a real pretrained model (`01`, `02`, `03`, `06`) have
a `--smoke` flag that runs the full pipeline on synthetic data /
randomly-initialized models in seconds, with no downloads — useful for
verifying the machinery before spending GPU time on the real run. The rest
(`00`, `04`, `05`, `07`) are synthetic already and run in seconds as-is. CI
runs all of them this way on every commit (see `.github/workflows/tests.yml`).

Prune any traceable model in one call:

```python
from prunelib import prune_model

group = prune_model(model, example_input, model.layer2[0].conv2, prune_fraction=0.3, method="max_k")
```

Prune a BERT model's attention heads and FFN neurons:

```python
from prunelib.transformer import (
    ActivationAwareSelector, FFNCalibrator, GradientHeadSelector, HeadCalibrator,
    prune_model_attention, prune_model_ffn,
)

cal = HeadCalibrator(model)
prune_model_attention(
    model, GradientHeadSelector(prune_percent=25), allocation="global", normalize="l2",
    stats_by_layer=cal.collect_many(batches),
    gradient_importance_by_layer=cal.collect_gradient_importance(labelled_batches),
    compensate_bias=True,
)
prune_model_ffn(
    model, ActivationAwareSelector(prune_percent=40), allocation="global",
    stats_by_layer=FFNCalibrator(model).collect_many(batches), compensate_bias=True,
)
```

The [usage guide](docs/USAGE.md) covers the full API: the tensor-level
primitives, `DependencyGraph` and its two-phase mode, attention-block
handlers, every extension point, and quantization.

## Benchmarks

CNN results below are the published, peer-reviewed figures (see
`PUBLICATIONS.md`). Latency is a fresh measurement from `experiments/00_demo.py`
on this package's implementation — **run it on your own machine before citing
a specific multiplier**; wall-clock speedup depends on hardware, batch size,
and how much of the network is pruned, and will not be identical across runs.

| Method | Model | Params ↓ | FLOPs ↓ | Acc. drop | Venue |
|---|---|---|---|---|---|
| Max-3 saliency | VGG16 / CIFAR-10 | 46.1% | 61.9% | 0.97% | IEEE Access 2024 |
| Max-3 saliency | ResNet56 | 35.2% | 35.2% | <1% | IEEE Access 2024 |
| CSD group regularization | VGG16 | 46.1% | 61.9% | 0.95% | IEEE Access 2024 |
| Manhattan K-Means | VGG16 | 35.2% | 49.1% | 0.98% | IEEE Access 2024 |
| Hybrid ordering (channel→channel→kernel) | VGG16 / Intel IC | 58.4% | 42.8% | 4.4% | IEEE ICCCNT 2023 |

| Component | Status |
|---|---|
| `saliency.py` — Max-k/L1/L2/random, Conv2d and Linear weights | Unit-tested, 10/10 passing |
| `surgery.py` — Conv/BN/FFN/attention-head structural surgery | Unit-tested, verified against a real HF BERT forward pass |
| `masking.py` / `vgg.py` — two-phase mask-then-compress | Unit-tested, including a whole-VGG16 mask → compress run |
| CUDA — every pruning path on a GPU | Both CUDA tests pass and the full suite passes on a GTX 1650 (2026-10-13); every experiment's `--device cuda` output matches its CPU run |
| `graph.py` / `group.py` / `op_rules.py` / `oneshot.py` — `torch.fx`-traced dependency resolution (elementwise merges incl. squeeze-and-excitation, functional activations, cat/flatten/depthwise), two-phase `PruningGroup`, `special_handlers`, `prune_model` | Unit-tested; `experiments/06 --tiny-check` verified against a real `torchvision.models.resnet18`, including cascading through a whole residual stage; torchvision MobileNetV3-Small pruned through its squeeze-and-excitation blocks (`test_graph.py`) |
| `quantization.py` — Float16/INT8/Fixed-Point32 post-training quantization | Unit-tested, including a brute-force formula check for INT8 and a clipping-not-wrapping check for Fixed-Point32; `experiments/07` runs the full pipeline on synthetic data, not yet against a real fine-tuned model |
| `scanners.py` — distance metrics + co-activation | Unit-tested |
| `registry.py` / `selection.py` / `distance.py` / `module_rules.py` — public extension points (scorers, selection rules, metrics, quantization methods, prunable layer types, graph ops) | Unit-tested, including a custom Conv1d rule and a `torch.maximum` residual driving `DependencyGraph` end to end; refactor verified bit-identical against the previous implementation on every pruning path |
| `clustering.py` — K-Means (Manhattan/Euclidean/Cosine) selection, prune lowest-L1 within each cluster | Unit-tested, including a brute-force check of the selection rule and end-to-end through `prune_model` and both VGG paths; paper's accuracy numbers not yet reproduced here |
| `experiments/01` VGG-CIFAR10 sweep | `--tiny-check` runs the real `torchvision.models.vgg16` class through real `prune_vgg_layer`/`prune_conv_bn` calls end to end (verified, ~3-4 min on CPU); full run (real CIFAR-10 + ImageNet weights) not yet executed |
| `experiments/02` BERT FFN sweep | Pipeline verified in `--smoke` against real `transformers` model classes; full run not yet executed |
| `experiments/03` head redundancy | Distance scan plus K-Means head pruning (`prune_attention_heads`) verified in `--smoke` against a real HF BERT forward pass; full run needs a fine-tuned checkpoint |
| `experiments/05` ordering | Scoring-perturbation mechanism verified; full accuracy-drop reproduction not yet run |

The CNN numbers above are the published, peer-reviewed results. The
Transformer experiments are tested infrastructure whose full runs are in
progress; their results will be added here once produced.

## Running the Transformer study

`experiments/research/` runs GLUE tasks × seeds × criteria × budgets,
resumable on free Kaggle/Colab GPUs, and produces paired mean ± std tables
and plots. See [`docs/RUNNING_EXPERIMENTS.md`](docs/RUNNING_EXPERIMENTS.md)
and `notebooks/pruning_study_kaggle_colab.ipynb`.

## Documentation

- [`docs/USAGE.md`](docs/USAGE.md): full API guide with examples
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): design decisions and the invariants the tests pin
- [`docs/RUNNING_EXPERIMENTS.md`](docs/RUNNING_EXPERIMENTS.md): producing publishable numbers
- [`docs/KT.md`](docs/KT.md): maintainer handbook (module walkthrough, known gaps, thesis material)
- [`docs/LEGACY_PIPELINE_MIGRATION.md`](docs/LEGACY_PIPELINE_MIGRATION.md): the archived end-to-end VGG16 pipeline

## Repository layout

```
prunelib/
    registry.py   name -> implementation registries, the shared extension mechanism
    saliency.py   Max-k (correct), L1, L2, CSD, random -- any [out, in, *kernel] weight
    selection.py  weight + budget -> indices to prune, by any registered rule
    clustering.py K-Means channel selection: prune lowest-L1 within each cluster
    distance.py   Manhattan/Euclidean/Cosine metrics (+ matching K-Means centroids)
    indices.py    index-set helpers: complement, expand channel -> block of columns;
                  index sets live on the CPU, moved to the data's device to index it
    surgery.py    slice_* primitives; conv/BN/FFN/attention-head structural surgery;
                  live_tensor (a masked weight as of now, not of the last forward pass)
    masking.py    two-phase mask-then-compress workflow (torch.nn.utils.prune)
    module_rules.py  per-layer-type rules DependencyGraph prunes through (Conv2d,
                  Linear, every BatchNorm/InstanceNorm built in; register more)
    op_rules.py   how a prune passes through + - * /, functional activations,
                  cat, flatten (register more ops)
    graph.py      torch.fx dependency resolution: the trace (one or several inputs)
                  and get_pruning_group; special_handlers/LeafTracer (attention-block hook)
    propagation.py  one prune decision's walk: which layers it touches
                  (merges/cat/flatten/depthwise), on whichever dim holds the channels
    group.py      PruningGroup: applies that decision -- .prune(), or
                  .mask() then .commit_and_compress(); refuses once the model has changed
    oneshot.py    prune_model(): score, select and prune any traceable model in one call
    quantization.py  Float16 / linear INT8 / Fixed-Point32 post-training quantization
    vgg.py        VGG wiring: build_vgg16, mask_vgg_layer, compress_masked_vgg
    scanners.py   pairwise distance matrix + co-activation scanning
    evaluate.py   parameter counts, measured latency, estimated size at a bit-width
    transformer/  model-level Transformer pruning (merged from transformer_pruning),
                  layered lowest-first; each layer imports only those above it:
        models/       adapters (BERT/RoBERTa layout, resolve_adapter), layer discovery
        measurement/  hooks, FFN/head calibration, activation recording
        analysis/     scorers, network scanner, OV-circuit math, twins, k-means
        selection/    contexts, budgets, FFN and head selectors
        surgery/      FFNSurgeon (merge + bias compensation), AttentionSurgeon
        pipeline/     allocation strategies, prune_* entry points
experiments/
    00_demo.py                  runs in seconds
    01_vgg_cifar10_sweep.py     Max3 vs L1 vs L2 vs random
    02_bert_sst2_sweep.py       FFN pruning on BERT
    03_head_redundancy.py       head similarity across layers
    04_coactivation.py          activation-based redundancy
    05_ordering.py              does the CNN ordering result transfer?
    06_generic_pruning.py       DependencyGraph on a real ResNet-18, no seam passed by hand
    07_quantization.py          prune_model() + all three quantization methods, one pipeline
    08_transformer_techniques.py  calibrated head + FFN pruning, twin merging
    research/                   the GLUE pruning study (run_study.py, summarize.py,
                                variants.py, glue.py) -- see docs/RUNNING_EXPERIMENTS.md
notebooks/
    run_on_colab_kaggle.ipynb          CNN pipeline (tests, VGG prune/fine-tune) on Colab/Kaggle
    pruning_study_kaggle_colab.ipynb   the GLUE transformer study on Kaggle/Colab
docs/
    USAGE.md                    full API guide with examples
    ARCHITECTURE.md             design decisions and the invariants tests pin
    RUNNING_EXPERIMENTS.md      step-by-step guide to producing publishable numbers
    KT.md                       maintainer handbook: module walkthrough, known gaps
    LEGACY_PIPELINE_MIGRATION.md  the archived VGG16 mask-then-compress pipeline
archive/
    legacy_pipeline/            end-to-end VGG16 mask-then-compress pipeline,
        config.py, data.py, model.py,   kept as a tested example (CNN pruning is
        train.py, pipeline.py           maintained in pruning_framwork_v4)
tests/          191 core tests, each naming the invariant or behavior it guards
    transformer/  199 tests for prunelib.transformer (BERT-shaped stub model, no HF needed)
```

## Citation

See `CITATION.cff`, or `PUBLICATIONS.md` for full BibTeX entries for all six
papers this code implements or extends.

## License

MIT — see `LICENSE`.
