import torch
import torch.nn.utils.prune as prune

from prunelib import build_vgg16, compress_masked_vgg, count_params, mask_vgg_layer
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

    total_masked = len(vgg_conv_bn_positions(model.features))  # placeholder, real check below
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
