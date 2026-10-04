import torch
import torch.nn.utils.prune as prune

from prunelib import build_vgg16, compress_masked_vgg, count_params, mask_vgg_layer, prune_vgg_layer
from prunelib.masking import surviving_channels
from prunelib.vgg import vgg_conv_bn_positions


def _tiny_vgg():
    # random-init, no download: same reason experiments/01 and legacy_pipeline
    # use FakeData/pretrained=False for their fast-path checks.
    return build_vgg16(num_classes=5, pretrained=False)


def test_mask_vgg_layer_does_not_resize_anything():
    model = _tiny_vgg()
    conv_idx, _ = vgg_conv_bn_positions(model.features)[0]
    out_before = model.features[conv_idx].out_channels

    mask_vgg_layer(model, layer_position=0, prune_fraction=0.1, method="max_k")

    assert model.features[conv_idx].out_channels == out_before  # unchanged -- phase 1 only masks
    assert prune.is_pruned(model.features[conv_idx])
    assert count_params(model) == count_params(_tiny_vgg())  # same architecture, same param count


def test_mask_vgg_layer_excludes_already_masked_channels_on_repeat_calls():
    """The correctness fix: without restricting selection to survivors, a
    second call would keep re-selecting channels from the first call (their
    weight is now zero -- the lowest possible score under every criterion),
    and the mask would barely grow. It must grow by roughly prune_fraction
    of the *original* channel count on every call, not shrink toward zero
    newly-masked channels."""
    model = _tiny_vgg()
    conv_idx, _ = vgg_conv_bn_positions(model.features)[0]
    n_out = model.features[conv_idx].out_channels

    first = mask_vgg_layer(model, layer_position=0, prune_fraction=0.1, method="max_k")
    second = mask_vgg_layer(model, layer_position=0, prune_fraction=0.1, method="max_k")
    third = mask_vgg_layer(model, layer_position=0, prune_fraction=0.1, method="max_k")

    expected_per_call = max(1, round(n_out * 0.1))
    assert first == expected_per_call
    assert second == expected_per_call  # not ~0, which is what the un-fixed version would give
    assert third == expected_per_call

    zeroed = n_out - surviving_channels(model.features[conv_idx].weight).numel()
    assert zeroed == first + second + third  # every call's contribution is additive, none wasted on re-selection


def test_mask_vgg_layer_stops_gracefully_when_fully_masked():
    model = _tiny_vgg()
    conv_idx, _ = vgg_conv_bn_positions(model.features)[0]
    n_out = model.features[conv_idx].out_channels

    total = 0
    for _ in range(30):  # enough calls at 10%/call to exhaust every channel
        newly = mask_vgg_layer(model, layer_position=0, prune_fraction=0.1, method="max_k")
        total += newly
        if newly == 0:
            break

    assert total == n_out  # every channel eventually masked, none double-counted
    assert mask_vgg_layer(model, layer_position=0, prune_fraction=0.1, method="max_k") == 0  # nothing left


def test_compress_masked_vgg_matches_masked_accuracy_numerically():
    """The whole point of the two-phase design: a masked (still full-size)
    model and the compressed (physically smaller) model built from it must
    produce identical output, since compression only removes channels that
    were already contributing exactly zero."""
    torch.manual_seed(0)
    model = _tiny_vgg()
    x = torch.randn(1, 3, 64, 64)

    for layer_position in range(3):
        mask_vgg_layer(model, layer_position, prune_fraction=0.2, method="max_k")

    model.eval()
    with torch.no_grad():
        out_masked = model(x)

    removed = compress_masked_vgg(model)
    assert removed > 0

    with torch.no_grad():
        out_compressed = model(x)

    assert torch.allclose(out_masked, out_compressed, atol=1e-4)
    assert count_params(model) < count_params(_tiny_vgg())


def test_compress_masked_vgg_propagates_through_unmasked_layers():
    """A layer that was never masked itself must still have its input
    channel count fixed to match a shrunk *previous* layer's output -- this
    is the 'even a no-op layer still needs the propagation' case described
    in compress_masked_vgg's docstring."""
    model = _tiny_vgg()
    pairs = vgg_conv_bn_positions(model.features)

    mask_vgg_layer(model, layer_position=0, prune_fraction=0.3, method="max_k")
    # deliberately do NOT mask layer_position=1

    compress_masked_vgg(model)

    conv0_idx, _ = pairs[0]
    conv1_idx, _ = pairs[1]
    assert model.features[conv1_idx].in_channels == model.features[conv0_idx].out_channels


def test_prune_vgg_layer_can_prune_the_last_conv_layer():
    """The last conv feeds classifier[0] (a Linear over the flattened 7x7
    avgpool output), not another conv. Pruning it must also drop that
    channel's 49 input columns from classifier[0] -- and give the same output
    as the unpruned model with that channel zeroed."""
    torch.manual_seed(0)
    model = _tiny_vgg().eval()
    pairs = vgg_conv_bn_positions(model.features)
    last = len(pairs) - 1
    conv_idx, _ = pairs[last]
    x = torch.randn(1, 3, 32, 32)

    reference = _tiny_vgg().eval()
    reference.load_state_dict(model.state_dict())

    kept = prune_vgg_layer(model, last, prune_fraction=0.25, method="l1")

    new_conv = model.features[conv_idx]
    assert new_conv.out_channels == kept == 384
    assert model.classifier[0].in_features == kept * 49

    removed = sorted(set(range(512)) - set(_matching_rows(reference.features[conv_idx].weight, new_conv.weight)))
    with torch.no_grad():
        reference.features[conv_idx].weight[removed] = 0
        reference.features[conv_idx].bias[removed] = 0
        assert torch.allclose(model(x), reference(x), atol=1e-4)


def _matching_rows(original: torch.Tensor, pruned: torch.Tensor) -> list[int]:
    """Which rows of `original` survived into `pruned`, by exact value."""
    flat_o = original.reshape(original.shape[0], -1)
    flat_p = pruned.reshape(pruned.shape[0], -1)
    return [int((flat_o == row).all(dim=1).nonzero()[0]) for row in flat_p]


def test_mask_and_compress_the_last_conv_layer_matches_masked_output():
    torch.manual_seed(0)
    model = _tiny_vgg().eval()
    last = len(vgg_conv_bn_positions(model.features)) - 1
    x = torch.randn(1, 3, 32, 32)

    mask_vgg_layer(model, last, prune_fraction=0.5, method="max_k")
    with torch.no_grad():
        out_masked = model(x)

    assert compress_masked_vgg(model) == 256
    assert model.classifier[0].in_features == 256 * 49
    with torch.no_grad():
        assert torch.allclose(model(x), out_masked, atol=1e-4)


def test_kmeans_method_works_through_both_vgg_pruning_paths():
    torch.manual_seed(0)
    model = _tiny_vgg().eval()
    conv_idx, _ = vgg_conv_bn_positions(model.features)[0]

    kept = prune_vgg_layer(model, 0, prune_fraction=0.25, method="kmeans", metric="manhattan")
    assert kept == model.features[conv_idx].out_channels == 48

    first = mask_vgg_layer(model, 1, prune_fraction=0.25, method="kmeans", metric="euclidean")
    second = mask_vgg_layer(model, 1, prune_fraction=0.25, method="kmeans", metric="euclidean")
    assert first == second == 16  # second call clusters survivors only, so both calls count
    compress_masked_vgg(model)
    assert model(torch.randn(1, 3, 32, 32)).shape == (1, 5)


def test_vgg_surgery_does_not_need_torchvision():
    """Only build_vgg16 needs torchvision (an optional dependency); the VGG
    surgery functions are plain torch over any torchvision-shaped VGG, so
    they must import and run without it. Run in a subprocess so blocking
    the import can't leak into other tests."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import sys
        sys.modules["torchvision"] = None  # any `import torchvision` now raises ImportError
        import torch, torch.nn as nn
        import prunelib

        model = nn.Module()
        model.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                                       nn.Conv2d(8, 6, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2)))
        model.classifier = nn.Sequential(nn.Linear(6 * 4, 3))
        assert prunelib.prune_vgg_layer(model, 0, 0.25) == 6
        assert prunelib.prune_vgg_layer(model, 1, 0.5) == 3
        assert model.classifier[0].in_features == 3 * 4
        try:
            prunelib.build_vgg16(pretrained=False)
        except ImportError:
            print("ok")
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
