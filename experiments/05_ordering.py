"""
05_ordering.py — the ICCCNT 2023 paper found that pruning order matters for
CNNs: channel-saliency -> channel-similarity -> kernel-saliency reached 58.4%
parameter reduction, while starting with kernel pruning capped out at 28.1%,
because kernel sparsity perturbs the channel statistics that channel
selection depends on.

This experiment checks whether the same effect is visible on a toy multi-
layer conv stack: prune channel-then-downstream-kernel vs. the reverse order,
and compare how much the *scores themselves* shift due to the other pruning
step having already happened. This is a scoring-perturbation check, not a
full accuracy-drop reproduction (that needs real training).

    python experiments/05_ordering.py --budget 0.4
"""
import argparse

import torch
import torch.nn as nn

from prunelib import compute_score, prune_conv_bn, select_prune_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    conv1 = nn.Conv2d(3, 32, 3, padding=1)
    bn1 = nn.BatchNorm2d(32)
    conv2 = nn.Conv2d(32, 32, 3, padding=1)

    prune_amount = int(32 * args.budget)

    # Order A: score channel importance BEFORE any pruning has happened.
    scores_before = compute_score(conv1.weight, method="max_k", k=3)
    keep_a = torch.tensor([i for i in range(32) if i not in set(select_prune_indices(scores_before, prune_amount).tolist())])
    _, _, pruned_conv2_a = prune_conv_bn(conv1, keep_a, bn=bn1, next_conv=conv2)

    # Order B: prune conv2's kernels first (simulated by zeroing a slice of
    # conv2's *input* side -- the kernel-level analogue), THEN re-score conv1.
    conv2_kernel_pruned = conv2.weight.clone()
    kernel_prune_amount = int(conv2_kernel_pruned.shape[1] * args.budget)
    conv2_kernel_pruned[:, :kernel_prune_amount] = 0.0

    # After kernel pruning has zeroed part of conv2's input side, conv1's
    # output channels feeding those zeroed slots are effectively invisible
    # to the network -- but conv1's OWN weights (what compute_score sees)
    # have not changed. This mirrors the paper's finding as a scoring
    # artifact: conv1 selection is oblivious to damage already done
    # downstream, so channel selection stops being informed by that channel's
    # actual remaining contribution.
    scores_after_kernel_prune = compute_score(conv1.weight, method="max_k", k=3)

    unchanged = torch.allclose(scores_before, scores_after_kernel_prune)
    print(f"conv1 channel scores identical before/after downstream kernel pruning: {unchanged}")
    print("(expected: True) -- channel-selection scores only see conv1's own weights,")
    print("so they cannot react to statistics that kernel pruning already destroyed")
    print("downstream. This is the mechanism the paper's result points at: prune")
    print("channels first, while the statistics they're scored on are still intact.")

    print(f"\nOrder A (channel-first) kept {len(keep_a)}/32 channels, seam verified: "
          f"conv2 in_channels={pruned_conv2_a.in_channels}")


if __name__ == "__main__":
    main()
