"""
The corrected replacement for `channel_pruning_saliency.py`,
`channel_pruning_distance.py`, `vgg_channel_pruning_saliency.py`, and
`vgg_channel_pruning_dist.py`. One driver, parameterized by `cfg.method`,
instead of four overlapping half-finished scripts.

This implements the two-phase design those original scripts were trying to
build: mask the weakest channels during pruning (weights zeroed, nothing
resized yet, safe to fine-tune against), and only once the whole pruning
schedule is done, build one compressed model and copy the surviving
("unmasked") weights across. See `prunelib/masking.py` for the phase-1/
phase-2 primitives and `LEGACY_PIPELINE_MIGRATION.md` for the full list of
what was wrong with the originals and why.

Why masking during the loop rather than resizing every iteration (an
earlier, simpler version of this file did that, and it's still what
`experiments/01_vgg_cifar10_sweep.py::prune_vgg_layer` does as a
lower-level building block): masking lets fine-tuning happen with the
*exact* eventual channel set already fixed and gradient-dead, across
however many iterations the schedule takes, before you commit to a physical
architecture change. Resizing every iteration works too, and is simpler,
but it's not what was asked for here, and it means every intermediate
model has to be a valid standalone nn.Module (fine for VGG's simple
Sequential, more awkward for architectures with skip connections).

No `layer_number` global tracking "which layer is currently being pruned"
for a `compute_mask` callback to read -- the bug shared by all six original
driver scripts (GITHUB_AUDIT.md section 11). Here there's no such global to
begin with: the loop below passes the layer position directly to
`mask_vgg_layer`.
"""
from __future__ import annotations

import csv
from pathlib import Path

from prunelib import compress_masked_vgg, count_params, mask_vgg_layer
from prunelib.vgg import vgg_conv_bn_positions

from .config import PruningConfig
from .data import build_dataloaders
from .model import freeze_all_but_classifier, get_device, load_model, unfreeze_all  # noqa: F401  (freeze kept for callers who want head-only fine-tuning)
from .train import evaluate, fit_one_cycle


def run_pruning(cfg: PruningConfig):
    """Phase 1 (masking): for up to `cfg.max_iterations`, mask another
    `cfg.prune_step` fraction of each (non-final) conv layer's surviving
    channels, fine-tune the whole model with those masks active, and
    evaluate on the *full* validation set. Stops early if accuracy drop
    exceeds `cfg.accuracy_drop_threshold` -- same schedule as Algorithm 1 in
    the IEEE Access paper.

    Phase 2 (compression): once masking is done, `compress_masked_vgg`
    commits every mask and rebuilds the network with only the surviving
    channels -- a single physically smaller model, evaluated once more to
    confirm the compressed model's accuracy matches what the masked version
    reported (see the equivalence test in `tests/test_masking.py` for why
    this should hold numerically, not just approximately).

    Returns `(results, model)` -- `results` is written to
    `cfg.results_csv_path` and has one row per masking iteration plus a
    final `phase="compressed"` row; `model` is the final, physically
    smaller network.
    """
    cfg.ensure_dirs()
    device = get_device()
    model = load_model(cfg, device)
    train_loader, test_loader = build_dataloaders(cfg)

    baseline = evaluate(model, test_loader, device)
    baseline_acc = baseline["val_acc"]
    n_prunable = len(vgg_conv_bn_positions(model.features)) - 1  # last conv feeds classifier[0]

    print(f"baseline: acc={baseline_acc:.4f}  params={count_params(model):,}")
    results = [{
        "phase": "baseline", "iteration": 0, "params": count_params(model),
        "masked_channels": 0, "val_acc": baseline_acc, "acc_drop": 0.0,
    }]

    total_masked = 0
    for iteration in range(1, cfg.max_iterations + 1):
        for layer_position in range(n_prunable):
            total_masked += mask_vgg_layer(model, layer_position, cfg.prune_step, method=cfg.method)

        unfreeze_all(model)  # fine-tune the whole (masked) model, not just the head
        history = fit_one_cycle(
            model, train_loader, test_loader, device,
            epochs=cfg.fine_tune_epochs_per_iteration,
            log_path=cfg.log_path,
        )
        val_acc = history[-1].val_acc
        acc_drop = baseline_acc - val_acc
        # count_params is *expected* to still equal the baseline here: masking
        # zeroes values via reparametrization, it does not remove parameters.
        # The real reduction only happens at compress_masked_vgg, below.
        params = count_params(model)

        print(
            f"iteration {iteration:>2}  masked_channels={total_masked:>4}  "
            f"params={params:>12,} (unchanged until compression)  "
            f"acc={val_acc:.4f}  drop={acc_drop:.4f}"
        )
        results.append({
            "phase": "masked", "iteration": iteration, "params": params,
            "masked_channels": total_masked, "val_acc": val_acc, "acc_drop": acc_drop,
        })

        if acc_drop > cfg.accuracy_drop_threshold:
            print(f"accuracy drop {acc_drop:.2%} exceeded threshold {cfg.accuracy_drop_threshold:.2%} -- stopping masking phase")
            break

    removed = compress_masked_vgg(model)
    compressed_acc = evaluate(model, test_loader, device)["val_acc"]
    compressed_params = count_params(model)
    print(
        f"compressed: removed {removed} channels across the network  "
        f"params={compressed_params:,} (was {results[0]['params']:,})  "
        f"acc={compressed_acc:.4f}"
    )
    results.append({
        "phase": "compressed", "iteration": results[-1]["iteration"], "params": compressed_params,
        "masked_channels": total_masked, "val_acc": compressed_acc, "acc_drop": baseline_acc - compressed_acc,
    })

    _write_results_csv(results, cfg.results_csv_path)
    return results, model


def _write_results_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "iteration", "params", "masked_channels", "val_acc", "acc_drop"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="max_k", choices=["max_k", "l1", "l2", "random"])
    parser.add_argument("--dataset", default="CIFAR10", choices=["CIFAR10", "ImageFolder", "FakeData"])
    parser.add_argument("--dataset-dir", default="./data")
    parser.add_argument("--prune-step", type=float, default=0.05)
    parser.add_argument("--accuracy-drop-threshold", type=float, default=0.01)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=1)
    parser.add_argument("--run-name", default="vgg16_channel_pruning")
    parser.add_argument("--tiny-check", action="store_true", help="FakeData + random-init weights, no downloads")
    args = parser.parse_args()

    if args.tiny_check:
        cfg = PruningConfig(
            dataset_name="FakeData", load_pretrained=False,
            max_iterations=2, fine_tune_epochs_per_iteration=1,
            method=args.method, run_name=f"tiny_check_{args.method}",
        )
    else:
        cfg = PruningConfig(
            dataset_name=args.dataset, dataset_dir=Path(args.dataset_dir),
            prune_step=args.prune_step, accuracy_drop_threshold=args.accuracy_drop_threshold,
            max_iterations=args.max_iterations, fine_tune_epochs_per_iteration=args.fine_tune_epochs,
            method=args.method, run_name=args.run_name,
        )

    run_pruning(cfg)
