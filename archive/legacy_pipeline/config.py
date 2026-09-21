"""
Configuration for the corrected legacy pipeline.

Replaces `config.ini` plus the ~15 lines of copy-pasted path-building and
global mutable state (`dataset_dir`, `logDir`, `layer_number`, `st`, `en`,
`new_list`, ...) duplicated near-verbatim at the top of every one of the six
original driver scripts. One dataclass, constructed once, passed explicitly
to the functions that need it -- no module-level globals for another
function to forget to update (that's what caused the `layer_number` bug:
see LEGACY_PIPELINE_MIGRATION.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PruningConfig:
    # Dataset
    dataset_dir: Path = Path("./data")
    dataset_name: str = "CIFAR10"          # or "ImageFolder" for a custom on-disk dataset
    train_subdir: str = "train"
    test_subdir: str = "test"
    image_size: int = 224
    batch_size: int = 32
    num_classes: int = 10

    # Model
    load_pretrained: bool = True
    saved_model_path: Path | None = None    # set to load a checkpoint instead of pretrained weights

    # Pruning schedule -- same defaults as Algorithm 1 in the IEEE Access paper
    prune_step: float = 0.05                # fraction of channels pruned per layer per iteration
    accuracy_drop_threshold: float = 0.01
    max_iterations: int = 15
    fine_tune_epochs_per_iteration: int = 1
    method: str = "max_k"                   # "max_k" | "l1" | "l2" | "random"

    # Output
    output_dir: Path = Path("./runs")
    run_name: str = "vgg16_channel_pruning"

    def __post_init__(self):
        self.dataset_dir = Path(self.dataset_dir)
        self.output_dir = Path(self.output_dir)
        if self.saved_model_path is not None:
            self.saved_model_path = Path(self.saved_model_path)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    def ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        return self.run_dir / "run.log"

    @property
    def results_csv_path(self) -> Path:
        return self.run_dir / "results.csv"
