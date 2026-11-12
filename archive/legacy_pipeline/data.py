"""
Dataset loading for the corrected legacy pipeline.

Corrects `load_dataset.py`:
- No module-level mutable state (`dataset_location`, `selected_dataset`,
  `train_directory`, ... were globals mutated by `set_folder_location`,
  read by every other function in the file). Everything here takes and
  returns explicit values.
- The original `data_loader_eval()` applied `transforms.CenterCrop`
  *as a class, not an instance* (`transforms.CenterCrop` instead of
  `transforms.CenterCrop(224)`) for one of its two splits -- a silently
  wrong composed-transform stage rather than an error, since `Compose`
  just calls whatever callable you hand it. Fixed by construction here:
  one `_eval_transform()` builder used everywhere eval-style
  preprocessing is needed, so there's no second hand-copied pipeline to
  get subtly wrong.
"""
from __future__ import annotations

from typing import Callable

import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import PruningConfig

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def _eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


DatasetBuilder = Callable[[PruningConfig], "tuple[Dataset, Dataset]"]

# upper-cased name -> (display name, builder); lookup is case-insensitive.
_DATASETS: dict[str, tuple[str, DatasetBuilder]] = {}


def register_dataset(name: str):
    """Decorator: register a `(cfg) -> (train, test)` builder under `name`
    (matched case-insensitively against `cfg.dataset_name`), so a new
    dataset is added alongside the others rather than by editing
    `build_datasets`."""

    def _register(fn: DatasetBuilder) -> DatasetBuilder:
        _DATASETS[name.upper()] = (name, fn)
        return fn

    return _register


def available_datasets() -> list[str]:
    return [display for display, _ in _DATASETS.values()]


@register_dataset("CIFAR10")
def _cifar10(cfg: PruningConfig) -> tuple[Dataset, Dataset]:
    root = str(cfg.dataset_dir)
    train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=_train_transform(cfg.image_size))
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=_eval_transform(cfg.image_size))
    return train, test


@register_dataset("ImageFolder")
def _image_folder(cfg: PruningConfig) -> tuple[Dataset, Dataset]:
    train_dir = cfg.dataset_dir / cfg.train_subdir
    test_dir = cfg.dataset_dir / cfg.test_subdir
    for d in (train_dir, test_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"expected an ImageFolder-structured directory at {d}")
    train = torchvision.datasets.ImageFolder(str(train_dir), transform=_train_transform(cfg.image_size))
    test = torchvision.datasets.ImageFolder(str(test_dir), transform=_eval_transform(cfg.image_size))
    return train, test


@register_dataset("FakeData")
def _fake_data(cfg: PruningConfig) -> tuple[Dataset, Dataset]:
    # No download, no disk layout required -- used by the --tiny-check
    # path to verify the pipeline mechanically. See
    # LEGACY_PIPELINE_MIGRATION.md for why this exists.
    def split(size: int) -> Dataset:
        return torchvision.datasets.FakeData(
            size=size, image_size=(3, cfg.image_size, cfg.image_size),
            num_classes=cfg.num_classes, transform=transforms.ToTensor(),
        )

    return split(cfg.fakedata_train_size), split(cfg.fakedata_test_size)


def build_datasets(cfg: PruningConfig) -> tuple[Dataset, Dataset]:
    """Returns (train_dataset, test_dataset) per `cfg.dataset_name`."""
    entry = _DATASETS.get(cfg.dataset_name.upper())
    if entry is None:
        raise ValueError(f"unknown dataset_name {cfg.dataset_name!r}, expected one of {available_datasets()}")
    return entry[1](cfg)


def build_dataloaders(cfg: PruningConfig) -> tuple[DataLoader, DataLoader]:
    train_set, test_set = build_datasets(cfg)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader
