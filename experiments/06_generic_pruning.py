"""
06_generic_pruning.py — proves `prunelib.graph.DependencyGraph` actually
closes the gap the rest of this library leaves open: every other surgery
function needs the caller to already know the seam (`prune_conv_bn`'s
`next_conv=`, `vgg.py` hardcoding that VGG's `.features` is a flat
`Sequential`). This experiment prunes a layer with no seam information
given at all and lets the graph figure out what else has to change.

    python experiments/06_generic_pruning.py --smoke        # synthetic net with a residual add and a concat, no torchvision
    python experiments/06_generic_pruning.py --tiny-check   # real torchvision.models.resnet18, no downloads (weights=None)
"""
import argparse

import torch
import torch.nn as nn

from prunelib import DependencyGraph, count_params


class TinyResidualNet(nn.Module):
    """One residual block (conv-bn-conv-bn + a 1x1 shortcut, summed) feeding
    a concat of two branches, feeding a classifier head through a flatten --
    exercises every coupling DependencyGraph understands (add, cat, flatten)
    in one small, synthetic, torchvision-free model."""

    def __init__(self, channels=8):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Conv2d(channels, channels, 1)
        self.shortcut_bn = nn.BatchNorm2d(channels)

        self.branch_a = nn.Conv2d(channels, 4, 3, padding=1)
        self.branch_b = nn.Conv2d(channels, 6, 3, padding=1)
        self.after_cat = nn.Conv2d(10, 8, 3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(8 * 2 * 2, 10)

    def forward(self, x):
        x = self.relu(self.stem_bn(self.stem(x)))

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = self.shortcut_bn(self.shortcut(x))
        x = self.relu(out + identity)

        x = torch.cat([self.branch_a(x), self.branch_b(x)], dim=1)
        x = self.relu(self.after_cat(x))

        x = self.flatten(self.pool(x))
        return self.fc(x)


def _report(label, model, example_input, keep_idx, dep, target):
    before = count_params(model)
    group = dep.get_pruning_group(target, keep_idx)
    touched = sorted(set(group.output_targets) | set(group.input_targets))
    group.prune()
    after = count_params(model)
    out = model(example_input)

    print(f"\n{label}")
    print(f"  layers coupled by this one prune decision: {touched}")
    print(f"  params: {before:,} -> {after:,}  ({(1 - after / before):.1%} reduction)")
    print(f"  forward pass after surgery: output shape {tuple(out.shape)}")


def run_smoke():
    torch.manual_seed(0)
    model = TinyResidualNet(channels=8)
    x = torch.randn(2, 3, 16, 16)

    dep = DependencyGraph(model, x)
    # Prune conv2's output -- must pull in bn2 and the shortcut conv/bn (the
    # add requires both branches to agree on which channels survive).
    _report("prune conv2 (residual-add coupling)", model, x, torch.tensor([0, 1, 3, 4, 6]), dep, model.conv2)

    # branch_b feeds a concat -- pruning it must offset after_cat's input indices.
    dep = DependencyGraph(model, x)
    _report("prune branch_b (concat-offset coupling)", model, x, torch.tensor([0, 2, 3, 5]), dep, model.branch_b)

    print("\nsmoke run complete: DependencyGraph found and applied both couplings")
    print("with no seam information passed in by hand.")


def run_tiny_check():
    """Exercises the exact same DependencyGraph code path as run_smoke, but
    against a real torchvision.models.resnet18 -- proves the "handle skip
    connections automatically" requirement holds against a real
    architecture's actual BasicBlock/downsample wiring, not just a hand-built
    fixture. weights=None -- no download, no network access needed."""
    import torchvision

    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None)
    model.eval()
    x = torch.randn(1, 3, 64, 64)

    dep = DependencyGraph(model, x)
    # layer2.0 is the first BasicBlock with a downsample/shortcut projection
    # (stride-2, channel count changes 64 -> 128) -- the case that requires
    # DependencyGraph to find and couple that projection conv automatically.
    target = model.layer2[0].conv2
    keep_idx = torch.arange(0, target.out_channels, 2)  # keep every other channel
    _report("prune layer2[0].conv2 on a real ResNet-18", model, x, keep_idx, dep, target)

    print("\ntiny-check complete: DependencyGraph found layer2[0]'s real")
    print("downsample projection conv and pruned it in lockstep with conv2/bn2,")
    print("on the real torchvision.models.resnet18 class -- no seam passed by hand.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="synthetic net with a residual add and a concat, no torchvision")
    group.add_argument("--tiny-check", action="store_true", help="real torchvision.models.resnet18, no downloads")
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
    else:
        run_tiny_check()
