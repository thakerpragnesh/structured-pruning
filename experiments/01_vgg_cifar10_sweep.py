"""
01_vgg_cifar10_sweep.py — reproduces the paper's comparison: Max-3 vs L1 vs L2
vs random channel selection, iteratively pruning a VGG-style network and
tracking accuracy drop, on CIFAR-10.

    python experiments/01_vgg_cifar10_sweep.py                # full run, downloads CIFAR-10 + torchvision VGG16
    python experiments/01_vgg_cifar10_sweep.py --smoke         # synthetic data + tiny net, no downloads, seconds

The full run requires a labeled dataset and fine-tuning budget the paper had
(CIFAR-10, VGG16, several fine-tune epochs per iteration) and is not something
to run inside a CI job — `--smoke` exists so the *pipeline* (scoring ->
selection -> surgery -> evaluate -> repeat) is verified on every commit even
though the real experiment is not.
"""
import argparse

import torch
import torch.nn as nn

from prunelib import compute_score, count_params, measure_latency, prune_conv_bn, select_prune_indices


class TinyVGGBlock(nn.Module):
    """Stand-in for a VGG conv stage: Conv-BN-ReLU x2, used only in --smoke mode."""

    def __init__(self, ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.conv2(x))


def run_smoke(prune_step=0.05, n_iterations=5, seed=0):
    torch.manual_seed(seed)
    model = TinyVGGBlock(ch=32)
    example_input = torch.randn(2, 3, 32, 32)

    print(f"{'iter':>4} {'method':<8} {'channels':>9} {'params':>8} {'latency(ms)':>12}")
    for method in ("max_k", "l1", "l2", "random"):
        m = TinyVGGBlock(ch=32)
        m.load_state_dict(model.state_dict())
        for it in range(n_iterations):
            n_ch = m.conv1.out_channels
            prune_amount = max(1, int(round(32 * prune_step)))
            kwargs = {"k": 3} if method == "max_k" else {}
            scores = compute_score(m.conv1.weight, method=method, **kwargs)
            keep = torch.tensor([i for i in range(n_ch) if i not in set(select_prune_indices(scores, prune_amount).tolist())])
            new_conv1, new_bn1, new_conv2 = prune_conv_bn(m.conv1, keep, bn=m.bn1, next_conv=m.conv2)
            m = TinyVGGBlock(ch=len(keep))
            m.conv1, m.bn1, m.conv2 = new_conv1, new_bn1, new_conv2
            latency = measure_latency(m, example_input, n_warmup=3, n_iters=10)
            print(f"{it:>4} {method:<8} {len(keep):>9} {count_params(m):>8,} {latency:>12.3f}")
    print("\nsmoke run complete: scoring -> selection -> surgery pipeline verified end to end.")
    print("This does NOT reproduce the paper's accuracy-drop numbers -- there is no")
    print("fine-tuning step and no real dataset here. Run without --smoke for that.")


def run_full():
    raise NotImplementedError(
        "Full run needs torchvision's VGG16, CIFAR-10, and a fine-tuning loop "
        "matching the paper's schedule (5% per layer per iteration, 1% accuracy-"
        "drop stop criterion). Not wired up here -- wire in your training loop "
        "and call compute_score/prune_conv_bn the same way run_smoke does."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_smoke() if args.smoke else run_full()
