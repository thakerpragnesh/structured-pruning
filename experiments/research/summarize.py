"""Aggregate `results.jsonl` from run_study.py into paper-ready tables.

    python experiments/research/summarize.py study_runs/results.jsonl
    python experiments/research/summarize.py a/results.jsonl b/results.jsonl --plot

Writes, next to the first input (or to --out):

- `summary.csv`  one row per (task, model, variant, percent, heal epochs):
                 mean and std over seeds of the primary metric before and
                 after healing, the drop from the baseline, and the measured
                 encoder-parameter reduction
- `summary.md`   the same as Markdown tables, one per task x research question,
                 sorted so the best criterion at each budget is first
- `plots/*.png`  (with --plot) primary metric vs. percent, one line per variant

Drops are *paired*: every variant at a given seed is cut from that seed's
baseline, so `drop` is computed per seed and then averaged -- seed-to-seed
baseline variance does not inflate the error bars.

Smoke-mode rows are excluded unless --include-smoke.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _mean_std(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _fmt(m, s, pct=False):
    if m is None:
        return "–"
    scale, unit = (100, "") if pct else (1, "")
    body = f"{m * scale:.2f}" if pct else f"{m:.4f}"
    if s is not None:
        body += f" ± {s * scale:.2f}" if pct else f" ± {s:.4f}"
    return body + unit


def load(paths, include_smoke=False):
    rows, seen = [], set()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["task"], r["model"], r["seed"], r["variant"], r["percent"], r["heal_epochs"])
            if key in seen or (r.get("smoke") and not include_smoke):
                continue
            seen.add(key)
            rows.append(r)
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["model"], r.get("group", ""), r["variant"], r["percent"], r["heal_epochs"])].append(r)

    out = []
    for (task, model, group, variant, percent, heal), rs in groups.items():
        metric = rs[0]["primary_metric"]
        base = [r["baseline"].get(metric) for r in rs]
        pruned = [r["pruned"].get(metric) for r in rs]
        healed = [(r["healed"] or {}).get(metric) for r in rs]
        drop_pruned = [b - p for b, p in zip(base, pruned) if b is not None and p is not None]
        drop_healed = [b - h for b, h in zip(base, healed) if b is not None and h is not None]
        reduction = [1 - r["params_encoder"] / r["baseline_params_encoder"] for r in rs]
        lat = [r["latency_ms"] / r["baseline_latency_ms"] for r in rs
               if r.get("latency_ms") and r.get("baseline_latency_ms")]
        agg = {"task": task, "model": model, "group": group, "variant": variant, "percent": percent,
               "heal_epochs": heal, "metric": metric, "n_seeds": len(rs),
               "seeds": " ".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"]))}
        for name, xs in (("baseline", base), ("pruned", pruned), ("healed", healed),
                         ("drop_pruned", drop_pruned), ("drop_healed", drop_healed),
                         ("encoder_param_reduction", reduction), ("latency_ratio", lat)):
            agg[f"{name}_mean"], agg[f"{name}_std"] = _mean_std(xs)
        out.append(agg)
    out.sort(key=lambda a: (a["task"], a["model"], a["group"], a["percent"], -(a["pruned_mean"] or -1e9)))
    return out


def write_csv(aggs, path):
    if not aggs:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggs[0]))
        writer.writeheader()
        writer.writerows(aggs)


def write_markdown(aggs, path, errors_note=""):
    lines = ["# Pruning study summary", "",
             "Mean ± std over seeds. **Pruned** = straight after the cut, before any fine-tuning "
             "(what the criterion knew); **healed** = after the heal epochs. Drop is paired per seed "
             "(baseline − variant, in points). Rows sorted by pruned score within each budget.", ""]
    if errors_note:
        lines += [errors_note, ""]
    by_table = defaultdict(list)
    for a in aggs:
        by_table[(a["task"], a["model"], a["group"])].append(a)
    for (task, model, group), rows in by_table.items():
        metric = rows[0]["metric"]
        widest = max(rows, key=lambda r: r["n_seeds"])  # baseline over the most seeds in this table
        lines += [f"## {task} · {model} · {group or 'ungrouped'}", "",
                  f"Baseline {metric}: {_fmt(widest['baseline_mean'], widest['baseline_std'])} "
                  f"(seeds: {widest['seeds']})", "",
                  f"| % | variant | pruned {metric} | drop (pts) | healed {metric} | drop after heal (pts) "
                  f"| enc. params −% | n |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(
                f"| {r['percent']:g} | {r['variant']} | {_fmt(r['pruned_mean'], r['pruned_std'])} "
                f"| {_fmt(r['drop_pruned_mean'], r['drop_pruned_std'], pct=True)} "
                f"| {_fmt(r['healed_mean'], r['healed_std'])} "
                f"| {_fmt(r['drop_healed_mean'], r['drop_healed_std'], pct=True)} "
                f"| {_fmt(r['encoder_param_reduction_mean'], None, pct=True)} | {r['n_seeds']} |")
        lines.append("")
    Path(path).write_text("\n".join(lines))


def plot(aggs, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping plots")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    by_fig = defaultdict(lambda: defaultdict(list))
    for a in aggs:
        by_fig[(a["task"], a["model"], a["group"], a["heal_epochs"])][a["variant"]].append(a)
    for (task, model, group, heal), variants in by_fig.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for name, pts in sorted(variants.items()):
            pts.sort(key=lambda a: a["percent"])
            x = [p["percent"] for p in pts]
            for ax, key in zip(axes, ("pruned", "healed")):
                y = [p[f"{key}_mean"] for p in pts]
                if all(v is None for v in y):
                    continue
                err = [p[f"{key}_std"] or 0 for p in pts]
                ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=name)
        metric = next(iter(variants.values()))[0]["metric"]
        for ax, title in zip(axes, ("before healing", f"after {heal:g} heal epoch(s)")):
            ax.set_title(title)
            ax.set_xlabel("pruned %")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(metric)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(f"{task} · {model.split('/')[-1]} · {group}")
        fig.tight_layout()
        name = f"{task}__{model.replace('/', '_')}__{group}__heal{heal:g}.png"
        fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"plots written to {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", type=Path, help="one or more results.jsonl files")
    p.add_argument("--out", type=Path, default=None, help="output directory (default: next to first input)")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--include-smoke", action="store_true")
    args = p.parse_args(argv)

    out = args.out or args.results[0].parent
    out.mkdir(parents=True, exist_ok=True)
    rows = load(args.results, include_smoke=args.include_smoke)
    if not rows:
        raise SystemExit("no (non-smoke) result rows found -- pass --include-smoke for smoke runs")

    errors_note = ""
    errors = [r.parent / "errors.jsonl" for r in args.results if (r.parent / "errors.jsonl").exists()]
    n_err = sum(len([l for l in e.read_text().splitlines() if l.strip()]) for e in errors)
    if n_err:
        errors_note = f"⚠ {n_err} run(s) failed -- see errors.jsonl; they are not in these tables."
        print(errors_note)

    aggs = aggregate(rows)
    write_csv(aggs, out / "summary.csv")
    write_markdown(aggs, out / "summary.md", errors_note)
    print(f"{len(rows)} runs -> {len(aggs)} rows: {out / 'summary.csv'}, {out / 'summary.md'}")
    if args.plot:
        plot(aggs, out / "plots")


if __name__ == "__main__":
    main()
