# Running the pruning study on Kaggle / Colab

How to produce publishable numbers for the `prunelib.transformer` criteria
on free GPUs. The code is in `experiments/research/`; the ready-made
notebook is `notebooks/pruning_study_kaggle_colab.ipynb`.

## What the study measures

Every run fine-tunes **one baseline per (task, model, seed)**, then cuts it
with one **variant** at one **percent**, and records the task metric
**before healing** (what the criterion knew) and **after** a short fine-tune
(what can be recovered). All variants at a seed start from the same
baseline, so differences between rows come from the criterion alone.

| Preset | Research question | Variants |
|---|---|---|
| `rq1-ffn` | Do the thesis's CNN criteria (Max-3, L1/L2, CSD, K-means) transfer to BERT FFN neurons, and how do they compare with output- and activation-aware criteria? | 12 |
| `rq1-heads` | The same question for attention heads (query-weight redundancy vs. OV-circuit, activation, gradient) | 8 |
| `rq2-order` | Does pruning order matter in transformers, as it did for CNNs (ICCCNT 2023)? Heads→FFN vs. FFN→heads, and is re-calibrating between steps necessary? | 5 |
| `quick` | One of each, for pilots | 5 |

`python experiments/research/run_study.py --list-variants` prints every
variant with a one-line description.

Percent semantics: *uniform* variants remove `percent` of every layer;
*global* variants (`*_global_*`, all of `rq2-order`) treat `percent` as a
model-wide budget, so layers lose different amounts. Every row records the
**measured** encoder-parameter reduction, and that measured value is the
one to plot against, not `percent`.

## Step 0 — before any GPU time

1. **Commit and push.** The notebook clones the repo from GitHub, and every
   result row records the git commit (plus a `git_dirty` flag in
   `environment.jsonl`). A result you can't tie to a commit can't be
   reproduced. If the repo is private, either make the `structured-pruning`
   repo public, or upload a zip of the repo as a private Kaggle Dataset and
   point `CODE_DIR` at it.
2. **Run the smoke test locally** (seconds, no downloads, no GPU):
   ```bash
   pip install -e ".[research]"
   python experiments/research/run_study.py --smoke --variants all
   ```
   All 23 variants should print one line each and finish with `done:`.

## Step 1 — set up the platform

### Kaggle (recommended for long runs)

1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** →
   upload `notebooks/pruning_study_kaggle_colab.ipynb`.
2. Right panel → **Session options**:
   - **Accelerator:** *GPU T4 x2* or *GPU P100*. The runner uses one GPU
     either way (see "Platform notes" for why).
   - **Internet:** on. This needs a phone-verified account; without it
     the clone and the dataset/model downloads fail.
3. Edit the **Configuration** cell and run all cells once interactively
   to check that the pilot works.
4. For real runs: **Save Version → Save & Run All (Commit)**. This runs in
   the background for up to the session limit (about 12 h), even with the
   browser closed. When it finishes, `/kaggle/working/study_runs` is the
   version's **Output**.
5. **To continue the next day:** open the notebook → **Add Input** → *Your
   Work* → this notebook (select its output) → **Save & Run All** again. The
   notebook detects the attached output and passes `--resume-from`
   automatically. It merges the earlier results, reuses the cached
   baselines, and runs only what's missing.

Check your remaining weekly GPU quota on your Kaggle profile before
starting a long version.

### Colab (good for pilots and short runs)

1. colab.research.google.com → **File → Upload notebook** → the same
   `.ipynb`.
2. **Runtime → Change runtime type → T4 GPU.**
3. Run all. The notebook mounts Google Drive and writes everything to
   `MyDrive/structured-pruning/study_runs`, so a disconnect loses at most
   the run in progress. Re-running the notebook resumes.
4. Free Colab doesn't run in the background: keep the tab open and
   active. Sessions can end early and GPU access can be throttled after
   heavy use. `TIME_BUDGET_HOURS` should be well under what you expect to
   get (e.g. 3–4).

## Step 2 — a staged run plan

Don't launch everything at once. Each stage tells you whether the next is
worth its quota.

| Stage | Configuration | Purpose |
|---|---|---|
| **A. Pilot** | `TASKS=["mrpc"]`, `SEEDS=[1]`, `VARIANTS=["quick"]`, `PERCENTS=[40]` | Check the baseline is sane (see below) and **read the per-run `seconds`** to cost the rest |
| **B. RQ1 on small tasks** | `TASKS=["mrpc","rte"]`, `SEEDS=[1,2,3]`, `VARIANTS=["rq1-ffn","rq1-heads"]`, `PERCENTS=[20,40,60]` | The core comparison, on the two cheapest tasks |
| **C. RQ2** | same tasks and seeds, `VARIANTS=["rq2-order"]` | Pruning order |
| **D. Larger tasks** | `TASKS=["sst2","qnli"]`, `HEAL_EXAMPLES=8000` (or `HEAL_EPOCHS=0`) | Generality. Healing on the full sets costs minutes per run, and the pre-heal column is the primary result anyway |
| **E. Latency** | see "Latency" below | Speed-up table |

**Costing.** Each result line prints its wall time, and each baseline
prints its training time (also stored in `baselines/*/meta.json`). Total
cost ≈ baselines × train time + runs × per-run time. As a rough guide,
BERT-base on a T4 with fp16 takes a few minutes per baseline on MRPC/RTE
and tens of minutes on SST-2/QNLI. Trust your pilot's measurement over
this guess. The slowest variants are the gradient head criteria (a
backward pass for calibration) and `ffn_cluster*` (k-means per layer).

**Splitting work.** Different tasks can run in different notebooks or
sessions, each with its own work dir; `summarize.py` accepts several
`results.jsonl` files at once. Keep **model, epochs, learning rate, batch
size, calibration size and heal settings identical** across everything
you plan to compare. They're part of the method, and different values
make rows non-comparable.

## Step 3 — sanity checks before trusting a table

- **Baseline in the normal range.** Approximate dev-set scores usually
  reported for fine-tuned `bert-base-uncased`, as sanity bounds rather
  than targets:

  | Task | Metric | Typical range |
  |---|---|---|
  | MRPC | accuracy | ~0.84–0.88 |
  | RTE | accuracy | ~0.63–0.72 |
  | SST-2 | accuracy | ~0.91–0.93 |
  | QNLI | accuracy | ~0.90–0.92 |
  | CoLA | Matthews corr. | ~0.52–0.60 |

  Far below the range means the baseline is undertrained or broken. Fix
  that before running variants.
- **Pruned should not beat the baseline systematically.** If many
  variants score *above* their baseline, the baseline is undertrained
  (seen in testing with 1 epoch on 400 examples).
- **Every criterion is compared against `*_random`.** A criterion that
  doesn't beat random at the same budget has no evidence behind it.
- **Small tasks are seed-sensitive.** BERT fine-tuning on RTE/MRPC
  occasionally degenerates to majority-class prediction for some seeds.
  Use ≥3 seeds, report mean ± std, and if you exclude a degenerate seed,
  state the rule in the paper and apply it to every variant.
- **`errors.jsonl`** lists failed runs. They're retried on the next
  invocation and excluded from summaries (the summary prints a warning).

## Step 4 — summarize

```bash
python experiments/research/summarize.py study_runs/results.jsonl --plot
# several sessions / platforms at once:
python experiments/research/summarize.py kaggle_a/results.jsonl kaggle_b/results.jsonl --out paper_tables --plot
```

The notebook's last cell does this for you. Outputs:

- `summary.csv` — one row per (task, model, variant, percent, heal):
  mean/std over seeds of the metric before and after healing, the
  **paired** drop (per seed against that seed's own baseline), measured
  parameter reduction, and the latency ratio if measured.
- `summary.md` — the same as Markdown tables per task and research
  question, best criterion first within each budget.
- `plots/*.png` — metric vs. pruned %, before and after healing, with
  ±std error bars.

## Latency

Shared cloud GPUs give noisy timings. For a paper's speed-up table, run a
separate latency pass on **one fixed machine** (your own GPU, or a laptop
CPU, stated as such). Baselines are cached, so this pass is cheap:

```bash
python experiments/research/run_study.py --tasks mrpc --seeds 1 \
    --variants ffn_act_global_comp heads_grad_global_comp order_heads_then_ffn \
    --percents 20 40 60 --heal-epochs 0 --latency --latency-batch 32 \
    --work-dir study_runs_latency --resume-from study_runs
```

`--heal-epochs 0` keys these rows separately from the accuracy rows.
Report the latency *ratio* to the baseline (`latency_ratio` in
`summary.csv`), with the device and batch size.

## Platform notes and troubleshooting

- **One GPU on purpose.** On Kaggle's *T4 x2*, HF Trainer would otherwise
  use DataParallel and double the effective batch size, so Kaggle results
  wouldn't match Colab's. The runner sets `CUDA_VISIBLE_DEVICES=0` unless
  you pass `--all-gpus`.
- **transformers 5 and the small `prajjwal1/bert-*` checkpoints.** They
  ship only `vocab.txt`, which transformers 5 can no longer turn into a
  tokenizer. Add `--tokenizer google-bert/bert-base-uncased`: same vocabulary,
  identical token ids (verified to give identical results on transformers
  4.57), and recorded in every result row. Their config also lacks
  `model_type`; the runner loads them as BERT automatically. `bert-base-uncased`
  itself needs neither.
- **Both transformers 4.x and 5.x work.** The runner picks the right warmup
  argument for the installed version (`warmup_ratio` on 4.x, a fractional
  `warmup_steps` on 5.x), so the training recipe is the same on either; the
  version used is recorded in `environment.jsonl`.
- **`ImportError: huggingface-hub>=0.34.0,<1.0 is required`.** A newer
  `datasets` pulled in an incompatible hub. Run
  `pip install "huggingface-hub>=0.34,<1.0"`, or upgrade `transformers`
  to match. The notebook avoids this by installing only packages that
  are missing.
- **CUDA out of memory.** Use `EXTRA_ARGS=["--batch-size", "16"]`. That
  changes the training recipe, so use it for *every* run you'll compare.
- **Kaggle clone/download fails.** Internet is off or the account isn't
  phone-verified.
- **Colab session ended.** Re-run all cells; results on Drive are kept.
- **Disk.** Each BERT-base baseline checkpoint is about 440 MB (e.g. 4
  tasks × 3 seeds ≈ 5 GB). Kaggle's output limit is 20 GB. Use
  `--no-save-baselines` only if you'll finish a (task, seed) in one session.

## What to report in the paper

For reproducibility, from `environment.jsonl` and the run settings:

- the GPU model(s) and the torch / transformers / datasets versions;
- the git commit (with `git_dirty: false`);
- model checkpoint, max sequence length (128), baseline epochs, learning
  rate, batch size, warmup ratio (0.1), weight decay (0.01), fp16;
- calibration set size (default 512 training examples) and heal settings;
- seeds, with mean ± std, and the paired-drop protocol;
- that pruned accuracy is measured with no fine-tuning, and healed
  accuracy after the stated heal budget.

## Command reference

```bash
python experiments/research/run_study.py --list-variants
python experiments/research/run_study.py --smoke [--variants all]
python experiments/research/run_study.py \
    --tasks mrpc rte --model google-bert/bert-base-uncased --seeds 1 2 3 \
    --variants rq1-ffn rq1-heads --percents 20 40 60 \
    --epochs 3 --heal-epochs 1 [--heal-examples 8000] \
    --work-dir study_runs [--resume-from PREVIOUS_DIR] [--time-budget-hours 11] \
    [--latency] [--batch-size 32] [--lr 2e-5] [--calib-examples 512]
python experiments/research/summarize.py study_runs/results.jsonl [--plot] [--out DIR]
```
