"""
Evaluation helpers: parameter counts and *measured* wall-clock latency.

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
