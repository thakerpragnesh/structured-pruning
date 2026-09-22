"""
Evaluation helpers: parameter counts, *measured* wall-clock latency, and
estimated size at a given bit-width.

Measured latency is what the README's "1.8x latency reduction" figure comes
from — structural surgery removes real rows/columns from real tensors, so a
smaller model is actually faster on the same hardware, not just "fewer FLOPs
on paper" the way masked pruning is.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_encoder_params(module: nn.Module, encoder_attr: str = "encoder") -> int:
    """Parameter count restricted to a named sub-module (e.g. a Transformer's
    `.encoder`), for reporting that excludes embeddings/heads which pruning
    doesn't touch."""
    sub = getattr(module, encoder_attr, module)
    return count_params(sub)


@torch.no_grad()
def measure_latency(
    module: nn.Module,
    example_input: torch.Tensor,
    n_warmup: int = 10,
    n_iters: int = 50,
) -> float:
    """Mean forward-pass latency in milliseconds, CPU wall-clock."""
    module.eval()
    for _ in range(n_warmup):
        module(example_input)
    start = time.perf_counter()
    for _ in range(n_iters):
        module(example_input)
    elapsed = time.perf_counter() - start
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
