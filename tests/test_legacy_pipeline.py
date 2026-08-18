"""Tests for legacy_pipeline -- the corrected replacement for
channel_pruning_saliency.py / channel_pruning_distance.py / the vgg_* scaffolds."""
import csv

import pytest
import torch

from legacy_pipeline.config import PruningConfig
from legacy_pipeline.data import build_dataloaders
from legacy_pipeline.model import freeze_all_but_classifier, get_device, load_model, unfreeze_all
from legacy_pipeline.pipeline import run_pruning
from legacy_pipeline.train import evaluate, fit_one_cycle
from prunelib import count_params


def test_config_paths_are_pathlib_and_derived_correctly(tmp_path):
    cfg = PruningConfig(output_dir=tmp_path, run_name="myrun")
    assert cfg.run_dir == tmp_path / "myrun"
    assert cfg.log_path == tmp_path / "myrun" / "run.log"
    assert cfg.results_csv_path == tmp_path / "myrun" / "results.csv"


def test_freeze_all_but_classifier_does_not_hardcode_a_parameter_index():
    """The original `freeze()` hardcoded a specific parameter count
    ('if count == 30') that only happened to be correct for one num_classes
    value. This should freeze the backbone and leave the classifier
    trainable regardless of how many output classes there are."""
    cfg = PruningConfig(dataset_name="FakeData", load_pretrained=False, num_classes=37)
    device = get_device()
    model = load_model(cfg, device)

    freeze_all_but_classifier(model)
    assert all(not p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())

    unfreeze_all(model)
    assert all(p.requires_grad for p in model.parameters())


@torch.no_grad()
def test_evaluate_averages_over_the_full_validation_set_not_the_last_batch():
    """D14: the original reassigned `outputs` inside the loop instead of
    accumulating, so it only ever reported the last batch. Build a loader
    where the last batch is trivially perfect and an earlier batch is
    trivially wrong, and confirm the reported accuracy reflects both."""

    class ConstantModel(torch.nn.Module):
        """Always predicts class 0, regardless of input."""

        def forward(self, x):
            batch = x.shape[0]
            logits = torch.full((batch, 2), -10.0)
            logits[:, 0] = 10.0
            return logits

    images = torch.zeros(4, 3, 4, 4)
    labels = torch.tensor([0, 0, 1, 1])  # half right, half wrong for a constant-class-0 predictor
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, labels), batch_size=1
    )

    result = evaluate(ConstantModel(), loader, torch.device("cpu"))
    assert result["val_acc"] == pytest.approx(0.5)  # 2 of 4 correct, not just the last sample's result


def test_fit_one_cycle_evaluates_once_per_epoch_not_once_per_batch():
    """D15: the original called evaluate() inside the per-batch loop. History
    should have exactly one entry per epoch, regardless of how many batches
    are in the training set."""
    cfg = PruningConfig(dataset_name="FakeData", load_pretrained=False, image_size=32, batch_size=4)
    device = get_device()
    model = load_model(cfg, device)
    train_loader, test_loader = build_dataloaders(cfg)
    assert len(train_loader) > 1  # otherwise this test can't distinguish per-batch from per-epoch

    history = fit_one_cycle(model, train_loader, test_loader, device, epochs=2)
    assert len(history) == 2


def test_layer_number_is_not_a_shared_mutable_global():
    """The root-cause systemic bug across all six original driver scripts:
    a global `layer_number` meant to track the current layer, which every
    one of them forgot to update (the line doing so was commented out
    everywhere), so masking was always applied against layer 0's candidate
    list. There is no `layer_number` global in this module at all --
    confirm it structurally doesn't exist to be forgotten."""
    import legacy_pipeline.pipeline as pipeline_module

    assert not hasattr(pipeline_module, "layer_number")


def test_run_pruning_end_to_end_tiny_check(tmp_path):
    """Real torchvision VGG16 (random-init), real FakeData, real masking,
    real fine-tuning, real full-validation-set evaluation, and a real final
    compression pass -- the entire corrected two-phase pipeline, no
    downloads. Slow-ish but this is the test that matters most: every
    original driver script either crashed or silently pruned nothing before
    reaching anything like this.

    Params must stay exactly at baseline through every masked iteration
    (masking zeroes values, it does not remove parameters -- if this ever
    shows a drop before compression, something is calling structural surgery
    too early) and then actually drop at the final `phase="compressed"` row.
    """
    cfg = PruningConfig(
        dataset_name="FakeData", load_pretrained=False,
        image_size=32, batch_size=4,
        max_iterations=1, fine_tune_epochs_per_iteration=1,
        method="max_k", output_dir=tmp_path, run_name="e2e",
    )
    results, model = run_pruning(cfg)

    assert len(results) == 3  # baseline + one masked iteration + final compressed row
    assert results[0]["phase"] == "baseline"
    assert results[1]["phase"] == "masked"
    assert results[2]["phase"] == "compressed"

    assert results[1]["masked_channels"] > 0
    assert results[1]["params"] == results[0]["params"]  # masking must not change param count

    assert results[2]["params"] < results[0]["params"]  # compression must actually shrink it
    assert count_params(model) == results[2]["params"]  # the returned model IS the compressed one

    assert cfg.results_csv_path.exists()
    with open(cfg.results_csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["phase"] == "baseline" and rows[2]["phase"] == "compressed"
