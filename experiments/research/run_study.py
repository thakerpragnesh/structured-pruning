"""Run the transformer pruning study: task x seed x variant x percent.

Designed for free, interruptible GPUs (Kaggle, Colab):

- **Resumable.** Every finished run is appended to `<work-dir>/results.jsonl`
  (one JSON object per line, flushed immediately). Re-running the same
  command skips everything already recorded, so a killed session loses at
  most the run in progress. `--resume-from DIR` first copies results and
  cached baselines from a previous session's output (e.g. a Kaggle notebook
  output attached as input).
- **One baseline per (task, model, seed),** fine-tuned once, saved under
  `<work-dir>/baselines/`, and the starting point of every variant -- so
  rows differ only in the pruning criterion.
- **Time budget.** `--time-budget-hours` stops cleanly before the platform's
  session limit instead of being killed mid-run.
- **Comparable across platforms.** Pinned to one GPU unless `--all-gpus`
  (Kaggle's "T4 x2" would otherwise silently double the effective batch
  size through DataParallel), and the GPU, library versions and git commit
  are recorded with every row.

Examples:

    # offline mechanics check, seconds on CPU (what CI runs)
    python experiments/research/run_study.py --smoke

    # first real pass: one task, one seed, the quick preset
    python experiments/research/run_study.py --tasks mrpc --seeds 1 --variants quick

    # the RQ1 FFN study
    python experiments/research/run_study.py --tasks mrpc rte sst2 qnli --seeds 1 2 3 \\
        --variants rq1-ffn --percents 20 40 60 --time-budget-hours 11
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=["mrpc"])
    p.add_argument("--model", default="google-bert/bert-base-uncased")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer checkpoint if different from --model (needed on transformers 5 for "
                        "models shipping only vocab.txt, e.g. prajjwal1/bert-*: use google-bert/bert-base-uncased)")
    p.add_argument("--seeds", nargs="+", type=int, default=[1])
    p.add_argument("--variants", nargs="+", default=["quick"], help="variant names and/or presets")
    p.add_argument("--percents", nargs="+", type=float, default=[40])
    p.add_argument("--epochs", type=float, default=3, help="baseline fine-tuning epochs")
    p.add_argument("--heal-epochs", type=float, default=1, help="fine-tuning after the cut (0 = none)")
    p.add_argument("--heal-examples", type=int, default=None,
                   help="heal on only the first N (seed-shuffled) training examples -- caps cost on SST-2/QNLI")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--calib-examples", type=int, default=512)
    p.add_argument("--train-limit", type=int, default=None, help="subsample training set (debugging)")
    p.add_argument("--eval-limit", type=int, default=None, help="subsample validation set (debugging)")
    p.add_argument("--work-dir", type=Path, default=Path("study_runs"))
    p.add_argument("--resume-from", type=Path, default=None,
                   help="copy results.jsonl and baselines/ from a previous session's work dir")
    p.add_argument("--time-budget-hours", type=float, default=None)
    p.add_argument("--latency", action="store_true", help="also time a forward pass (noisy on shared GPUs)")
    p.add_argument("--latency-batch", type=int, default=32)
    p.add_argument("--all-gpus", action="store_true", help="allow DataParallel across all visible GPUs")
    p.add_argument("--no-save-baselines", action="store_true", help="don't write baseline checkpoints to disk")
    p.add_argument("--list-variants", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny random BERT + synthetic data, no downloads; overrides most options")
    args = p.parse_args(argv)
    if args.smoke:
        # Smoke mode fixes the model/data; seeds, percents and variants stay
        # whatever was passed (defaults: seed 1, 25%).
        args.tasks = ["mrpc"]
        if args.percents == [40]:
            args.percents = [25]
        # lr high enough that the tiny random model actually learns the
        # synthetic rule, so pruning visibly moves the metric.
        args.model, args.epochs, args.heal_epochs, args.lr = "smoke-bert", 8, 1, 1e-3
        args.calib_examples = 64
        if args.work_dir == Path("study_runs"):
            args.work_dir = Path("study_runs_smoke")
    return args


# ------------------------------------------------------------- bookkeeping
def run_key(row) -> tuple:
    return (row["task"], row["model"], row["seed"], row["variant"], float(row["percent"]),
            float(row["heal_epochs"]))


def load_done(results_path: Path) -> set:
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                done.add(run_key(json.loads(line)))
    return done


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def resume(src: Path, work_dir: Path) -> None:
    """Merge a previous session's results and copy its missing baselines."""
    src_results = src / "results.jsonl"
    dst_results = work_dir / "results.jsonl"
    if src_results.exists():
        done = load_done(dst_results)
        added = 0
        for line in src_results.read_text().splitlines():
            if line.strip() and run_key(json.loads(line)) not in done:
                with open(dst_results, "a") as f:
                    f.write(line + "\n")
                added += 1
        print(f"resume: merged {added} result rows from {src_results}")
    src_base = src / "baselines"
    if src_base.is_dir():
        for d in src_base.iterdir():
            target = work_dir / "baselines" / d.name
            if d.is_dir() and not target.exists():
                shutil.copytree(d, target)
                print(f"resume: copied baseline {d.name}")


def environment() -> dict:
    import torch
    import transformers

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                                    text=True, timeout=10).stdout.strip())
    except Exception:
        commit, dirty = None, None
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        where = "kaggle"
    elif os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
        where = "colab"
    else:
        where = "local"
    try:
        import datasets
        datasets_version = datasets.__version__
    except ImportError:
        datasets_version = None
    return {
        "platform": where,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"cpu ({platform.processor()})",
        "gpus_visible": torch.cuda.device_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets_version,
        "git_commit": commit,
        "git_dirty": dirty,
    }


def structure(model) -> dict:
    from prunelib.transformer import resolve_adapter

    adapter = resolve_adapter(model)
    layers = range(adapter.num_layers())
    return {"heads": [adapter.num_attention_heads(i) for i in layers],
            "ffn": [adapter.get_ffn(i)[0].out_features for i in layers]}


def param_counts(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    encoder = sum(p.numel() for p in model.base_model.encoder.parameters())
    return {"params_total": total, "params_encoder": encoder}


def latency_ms(model, batch: int, seq: int = 128, vocab: int = 1000) -> float:
    """Median forward-pass latency. Noisy on shared cloud GPUs -- for a
    paper's latency table, re-run with `--heal-epochs 0 --latency` on one
    fixed machine (baselines are cached, so that pass is cheap)."""
    import torch

    device = next(model.parameters()).device
    ids = torch.randint(0, vocab, (batch, seq), device=device)
    mask = torch.ones_like(ids)
    model.eval()
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    times = []
    with torch.no_grad():
        for i in range(25 if device.type == "cuda" else 8):
            sync()
            t = time.perf_counter()
            model(input_ids=ids, attention_mask=mask)
            sync()
            if i >= (5 if device.type == "cuda" else 2):  # warm-up
                times.append((time.perf_counter() - t) * 1000)
    return float(sorted(times)[len(times) // 2])


# ------------------------------------------------------------------ main
def main(argv=None):
    args = parse_args(argv)
    if not args.all_gpus and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # before torch is imported
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO))

    import torch

    import variants as V
    from glue import GlueHarness, TASKS, set_seeds

    if args.list_variants:
        for name, v in V.VARIANTS.items():
            print(f"{name:<28} [{v.group}] {v.description}")
        print("\npresets:", ", ".join(f"{k} ({len(v)})" for k, v in V.PRESETS.items()))
        return 0

    variant_names = V.resolve(args.variants)
    for t in args.tasks:
        if t not in TASKS:
            raise SystemExit(f"unknown task {t!r}; expected one of {sorted(TASKS)}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_from is not None:
        resume(args.resume_from, args.work_dir)
    results_path = args.work_dir / "results.jsonl"
    errors_path = args.work_dir / "errors.jsonl"
    done = load_done(results_path)

    env = environment()
    with open(args.work_dir / "environment.jsonl", "a") as f:
        f.write(json.dumps({**env, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv[1:]}) + "\n")
    print(f"environment: {env['platform']} | {env['device']} | torch {env['torch']} | "
          f"transformers {env['transformers']} | commit {env['git_commit']}{' (dirty)' if env['git_dirty'] else ''}")

    plan = [(task, seed, v, float(pct)) for task in args.tasks for seed in args.seeds
            for v in variant_names for pct in args.percents]
    pending = [u for u in plan
               if (u[0], args.model, u[1], u[2], u[3], float(args.heal_epochs)) not in done]
    print(f"plan: {len(plan)} runs, {len(plan) - len(pending)} already done, {len(pending)} to go")

    start = time.time()

    def out_of_time():
        return args.time_budget_hours is not None and (time.time() - start) / 3600 > args.time_budget_hours

    current = None  # (task, seed) whose harness/baseline is loaded
    harness = baseline = baseline_meta = None
    finished = 0
    for task, seed, vname, pct in pending:
        if out_of_time():
            print(f"\ntime budget of {args.time_budget_hours}h reached after {finished} runs -- "
                  "stopping cleanly. Re-run the same command (with --resume-from on Kaggle) to continue.")
            return 0

        if current != (task, seed):
            harness = GlueHarness(task, args.model, args.work_dir, seed, epochs=args.epochs,
                                  batch_size=args.batch_size, learning_rate=args.lr,
                                  max_length=args.max_length, train_limit=args.train_limit,
                                  eval_limit=args.eval_limit, smoke=args.smoke,
                                  tokenizer_name=args.tokenizer)
            baseline, baseline_meta = get_baseline(harness, args, task, seed, set_seeds)
            current = (task, seed)

        set_seeds(seed)
        t0 = time.time()
        model = copy.deepcopy(baseline)
        try:
            V.VARIANTS[vname].apply(model, pct, V.Context(harness, args.calib_examples))
            pruned = harness.evaluate(model)
            shape = structure(model)
            counts = param_counts(model)
            lat = latency_ms(model, args.latency_batch) if args.latency else None
            healed = None
            if args.heal_epochs > 0:
                harness.train(model, epochs=args.heal_epochs, train_examples=args.heal_examples)
                healed = harness.evaluate(model)
        except Exception as exc:  # recorded, not fatal: the rest of the plan still runs
            append_jsonl(errors_path, {"task": task, "model": args.model, "seed": seed, "variant": vname,
                                       "percent": pct, "error": f"{type(exc).__name__}: {exc}",
                                       "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            print(f"  ERROR {task} seed={seed} {vname} @{pct:g}%: {type(exc).__name__}: {exc}")
            del model
            continue

        metric = TASKS[task].primary_metric
        row = {
            "task": task, "model": args.model, "tokenizer": args.tokenizer or args.model,
            "seed": seed, "variant": vname,
            "group": V.VARIANTS[vname].group, "percent": pct, "heal_epochs": float(args.heal_epochs),
            "heal_examples": args.heal_examples,
            "primary_metric": metric,
            "baseline": baseline_meta["metrics"], "pruned": pruned, "healed": healed,
            **counts,
            "baseline_params_total": baseline_meta["params_total"],
            "baseline_params_encoder": baseline_meta["params_encoder"],
            "structure": shape,
            "latency_ms": lat, "baseline_latency_ms": baseline_meta.get("latency_ms") if args.latency else None,
            "seconds": round(time.time() - t0, 1),
            "env": {k: env[k] for k in ("platform", "device", "torch", "transformers", "git_commit")},
            "smoke": args.smoke,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_jsonl(results_path, row)
        finished += 1
        b, pr = baseline_meta["metrics"].get(metric), pruned.get(metric)
        hl = healed.get(metric) if healed else None
        saved = 1 - counts["params_encoder"] / baseline_meta["params_encoder"]
        print(f"  [{finished}/{len(pending)}] {task} seed={seed} {vname:<26} @{pct:>4g}%  "
              f"{metric}: base {b:.4f}  pruned {pr:.4f}"
              + (f"  healed {hl:.4f}" if hl is not None else "")
              + f"  enc.params -{saved:.1%}  ({row['seconds']}s)")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\ndone: {finished} new runs written to {results_path}")
    return 0


def get_baseline(harness, args, task, seed, set_seeds):
    """Load the cached fine-tuned baseline for (task, model, seed), or train
    and cache it."""
    import torch
    from transformers import AutoModelForSequenceClassification

    slug = args.model.replace("/", "__")
    bdir = args.work_dir / "baselines" / f"{task}__{slug}__seed{seed}"
    meta_path = bdir / "meta.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if meta_path.exists():
        model = AutoModelForSequenceClassification.from_pretrained(bdir).to(device)
        meta = json.loads(meta_path.read_text())
        print(f"\n=== {task} seed={seed}: loaded cached baseline ({meta['metrics']})")
        if args.latency and "latency_ms" not in meta:
            meta["latency_ms"] = latency_ms(model, args.latency_batch)
        return model, meta

    print(f"\n=== {task} seed={seed}: fine-tuning baseline {args.model} for {args.epochs} epochs")
    set_seeds(seed)
    model = harness.new_model()
    t0 = time.time()
    harness.train(model, progress=True)
    model.to(device)
    meta = {"metrics": harness.evaluate(model), **param_counts(model),
            "train_seconds": round(time.time() - t0, 1), "epochs": args.epochs}
    if args.latency:
        meta["latency_ms"] = latency_ms(model, args.latency_batch)
    print(f"    baseline: {meta['metrics']}  ({meta['train_seconds']}s)")
    if not args.no_save_baselines:
        bdir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(bdir)
        meta_path.write_text(json.dumps(meta, indent=2))
    return model, meta


if __name__ == "__main__":
    sys.exit(main())
