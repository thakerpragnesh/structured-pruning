"""
01_vgg_cifar10_sweep.py — reproduces the paper's comparison: Max-3 vs L1 vs L2
vs random channel selection, iteratively pruning VGG16 and tracking accuracy
drop, on CIFAR-10.

    python experiments/01_vgg_cifar10_sweep.py                 # full run: downloads CIFAR-10 + ImageNet VGG16 weights
    python experiments/01_vgg_cifar10_sweep.py --tiny-check     # real torchvision VGG16 + FakeData, no downloads, seconds
    python experiments/01_vgg_cifar10_sweep.py --smoke          # fully synthetic tiny net, no torchvision needed at all

`build_vgg16` and `prune_vgg_layer` live in `prunelib.vgg` -- shared with
`legacy_pipeline/pipeline.py`, the corrected replacement for the old
`channel_pruning_saliency.py` / `channel_pruning_distance.py` drivers. See
`LEGACY_PIPELINE_MIGRATION.md` for why that pipeline is a rebuild rather than
a line-by-line bug fix of the original.
"""
import argparse

import torch
import torch.nn as nn
import torchvision

from prunelib import build_vgg16, compute_score, count_params, measure_latency, prune_conv_bn, prune_vgg_layer, select_prune_indices
from prunelib.vgg import vgg_conv_bn_positions


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


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def _fine_tune(model, loader, device, epochs, lr=1e-4):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()


def run(dataset_factory, pretrained, prune_step=0.05, accuracy_drop_threshold=0.01,
        max_iterations=15, fine_tune_epochs=1, methods=("max_k", "l1", "l2", "random"), device=None):
    """The actual iterative-pruning experiment: for each scoring method,
    prune every (non-final) conv layer by `prune_step`, fine-tune, evaluate,
    and stop when accuracy drop exceeds `accuracy_drop_threshold` -- the same
    schedule as Algorithm 1 in the IEEE Access paper (5% per iteration, 1%
    accuracy-drop stop criterion, defaults here match that)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set = dataset_factory()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8)

    for method in methods:
        model = build_vgg16(pretrained=pretrained).to(device)
        baseline_acc = _evaluate(model, test_loader, device)
        n_prunable = len(vgg_conv_bn_positions(model.features)) - 1  # last conv excluded, see prune_vgg_layer

        print(f"\n=== method={method}  baseline acc={baseline_acc:.4f} ===")
        for it in range(1, max_iterations + 1):
            for layer_position in range(n_prunable):
                prune_vgg_layer(model, layer_position, prune_step, method=method)
            _fine_tune(model, train_loader, device, epochs=fine_tune_epochs)
            acc = _evaluate(model, test_loader, device)
            drop = baseline_acc - acc
            print(f"  iter {it:>2}  params={count_params(model):>11,}  acc={acc:.4f}  drop={drop:.4f}")
            if drop > accuracy_drop_threshold:
                print(f"  accuracy drop exceeded {accuracy_drop_threshold:.0%} -- stopping {method}")
                break


def run_full():
    """Real run: downloads ImageNet-pretrained VGG16 weights and CIFAR-10.
    Needs network access to download.pytorch.org and the CIFAR-10 host --
    not available in every sandboxed environment, but will work on a normal
    machine with internet access."""
    def cifar10():
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        return train, test

    run(cifar10, pretrained=True)


def run_tiny_check():
    """Exercises the exact same code path as run_full -- the real
    torchvision.models.vgg16 class, the real prune_vgg_layer/prune_conv_bn
    calls -- but with randomly-initialized weights (`pretrained=False`) and
    FakeData instead of downloaded weights and CIFAR-10, so it runs in
    seconds with no network access at all. This is what verifies `run_full`
    is wired correctly without needing internet -- it is not a substitute for
    running `run_full` for a real result.
    """
    def fake_data():
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
        ])
        train = torchvision.datasets.FakeData(size=16, image_size=(3, 224, 224), num_classes=10, transform=transform)
        test = torchvision.datasets.FakeData(size=8, image_size=(3, 224, 224), num_classes=10, transform=transform)
        return train, test

    run(fake_data, pretrained=False, max_iterations=2, fine_tune_epochs=1, methods=("max_k", "l1"))
    print("\ntiny-check complete: the real VGG16 class and real prune_vgg_layer/")
    print("prune_conv_bn calls ran end to end with no shape errors. Numbers above")
    print("are meaningless (random weights, fake data) -- run without any flag,")
    print("on a machine with internet access, for the real result.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="fully synthetic tiny net, no torchvision")
    group.add_argument("--tiny-check", action="store_true", help="real VGG16 class + FakeData, no downloads")
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
    elif args.tiny_check:
        run_tiny_check()
    else:
        run_full()
