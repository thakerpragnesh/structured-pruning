"""
00_demo.py — the whole pipeline in one place, runs in a few seconds on CPU.

Builds a small conv block, scores its channels with Max-3/L1/L2/random,
physically prunes 50% of channels via structural surgery, and reports the
real parameter count and measured latency before and after.

    python experiments/00_demo.py
"""
import argparse
import time

import torch
import torch.nn as nn

from prunelib import compute_score, count_params, measure_latency, prune_conv_bn, select_prune_indices


class ConvBlock(nn.Module):
    def __init__(self, in_ch=3, mid_ch=64, out_ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.conv2(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = ConvBlock()
    example_input = torch.randn(1, 3, 64, 64)

    print(f"{'method':<8} {'lowest-score channel':>22} {'top-3 sum':>12}")
    for method in ("max_k", "l1", "l2", "random"):
        kwargs = {"k": 3} if method == "max_k" else {}
        scores = compute_score(model.conv1.weight, method=method, **kwargs)
        weakest = select_prune_indices(scores, 1).item()
        print(f"{method:<8} {weakest:>22} {scores[weakest].item():>12.4f}")

    scores = compute_score(model.conv1.weight, method="max_k", k=3)
    n_out = model.conv1.out_channels
    prune_amount = int(n_out * args.prune_fraction)
    prune_idx = set(select_prune_indices(scores, prune_amount).tolist())
    keep_idx = torch.tensor([i for i in range(n_out) if i not in prune_idx])

    params_before = count_params(model)
    latency_before = measure_latency(model, example_input)

    new_conv1, new_bn1, new_conv2 = prune_conv_bn(model.conv1, keep_idx, bn=model.bn1, next_conv=model.conv2)
    pruned_model = ConvBlock(mid_ch=len(keep_idx))
    pruned_model.conv1, pruned_model.bn1, pruned_model.conv2 = new_conv1, new_bn1, new_conv2

    params_after = count_params(pruned_model)
    latency_after = measure_latency(pruned_model, example_input)

    # Correctness check: pruned model must actually run and produce the right
    # output shape, not just report smaller numbers.
    out = pruned_model(example_input)
    assert out.shape == (1, model.conv2.out_channels, 64, 64)

    print(f"\nprune fraction: {args.prune_fraction:.0%}  ({n_out} -> {len(keep_idx)} channels)")
    print(f"params:  {params_before:,} -> {params_after:,}  ({(1 - params_after/params_before):.1%} reduction)")
    print(f"latency: {latency_before:.3f}ms -> {latency_after:.3f}ms  ({latency_before/latency_after:.2f}x)")


if __name__ == "__main__":
    main()
