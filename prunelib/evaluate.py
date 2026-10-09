"""
Evaluation helpers: parameter counts, *measured* wall-clock latency, and
estimated size at a given bit-width.

Measured latency is what the README's quickstart latency figure
(`experiments/00_demo.py`) comes from — structural surgery removes real
rows/columns from real tensors, so a smaller model is actually faster on
the same hardware, not just "fewer FLOPs on paper" the way masked pruning
is.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_encoder_params(module: nn.Module, encoder_attr: str = "encoder") -> int:
    """Parameter count restricted to a named sub-module (e.g. a Transformer's
    `.encoder`, or `"bert.encoder"` inside a task model that wraps one), for
    reporting that excludes embeddings/heads which pruning doesn't touch.
    Raises `AttributeError` if `module` has no such submodule. It used to
    count the whole model instead, so a typo or a wrapper class reported
    embeddings and heads as encoder parameters without a word."""
    return count_params(module.get_submodule(encoder_attr))


@torch.no_grad()
def measure_latency(
    module: nn.Module,
    example_input: torch.Tensor,
    n_warmup: int = 10,
    n_iters: int = 50,
) -> float:
    """Mean forward-pass latency in milliseconds, CPU wall-clock, measured
    in eval mode. Every submodule's own train/eval mode is restored
    afterwards, mixed modes included (a frozen BatchNorm kept in eval inside
    a training model). It used to be left in eval mode, so a model timed
    mid-training quietly stopped using dropout and updating BatchNorm
    statistics."""
    modes = [(m, m.training) for m in module.modules()]
    module.eval()
    try:
        for _ in range(n_warmup):
            module(example_input)
        start = time.perf_counter()
        for _ in range(n_iters):
            module(example_input)
        elapsed = time.perf_counter() - start
    finally:
        for m, training in modes:
            m.training = training
    return (elapsed / n_iters) * 1000.0


def estimate_size_bytes(module: nn.Module, bits_per_param: float = 32.0) -> float:
    """Estimated model size in bytes at a uniform `bits_per_param` -- e.g.
    16 after `quantization.quantize_model_(model, "float16")`, or 8 for a
    model whose weights have all been packed via
    `quantization.quantize_int8_linear`. This is `count_params(module) *
    bits_per_param / 8`, nothing more -- not a substitute for measuring an
    actually-saved checkpoint's file size (which also includes whatever
    per-tensor scale/zero-point metadata a real INT8 export would store),
    but a quick way to compare precision options without writing one to
    disk for each.
    """
    return count_params(module) * bits_per_param / 8.0
