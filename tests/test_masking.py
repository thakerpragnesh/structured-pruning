import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from prunelib.masking import (
    build_channel_mask,
    commit_mask,
    compress_masked_conv_bn,
    mask_channels,
    surviving_channels,
    zeroed_channels,
)
from prunelib.surgery import prune_conv_bn


def test_mask_zeroes_forward_contribution_without_resizing():
    """Masking must not change the module's shape -- only zero the selected
    channels' contribution to the forward pass, weight AND bias."""
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    x = torch.randn(1, 3, 16, 16)
    prune_idx = torch.tensor([1, 4, 6])

    mask_channels(conv, prune_idx)
    assert conv.out_channels == 8  # unchanged -- masking is non-destructive
    assert prune.is_pruned(conv)

    out = conv(x)
    for c in prune_idx.tolist():
        assert torch.allclose(out[:, c], torch.zeros_like(out[:, c]))
    kept = [i for i in range(8) if i not in set(prune_idx.tolist())]
    assert not torch.allclose(out[:, kept], torch.zeros_like(out[:, kept]))


def test_mask_zeroes_bias_too():
    """A masked channel must contribute nothing at all -- if only the weight
    were masked, a nonzero bias would still leak a constant into every
    spatial position of that channel, which wouldn't match what happens once
    the channel is physically removed during compression."""
    conv = nn.Conv2d(3, 4, kernel_size=3)
    conv.bias.data.fill_(5.0)  # deliberately large, easy to spot if it leaks through
    mask_channels(conv, torch.tensor([1, 2]))

    assert torch.allclose(conv.bias[[1, 2]], torch.zeros(2))
    assert torch.allclose(conv.bias[[0, 3]], torch.full((2,), 5.0))


def test_mask_preserves_original_weights_until_commit():
    """Before commit_mask, the original values must still be recoverable --
    this is what makes masking safe to fine-tune/evaluate against before
    committing to anything."""
    conv = nn.Conv2d(3, 4, kernel_size=3)
    original = conv.weight.detach().clone()
    mask_channels(conv, torch.tensor([0, 2]))

    assert torch.allclose(conv.weight_orig, original)  # untouched
    assert torch.allclose(conv.weight[0], torch.zeros_like(conv.weight[0]))  # masked view
    assert torch.allclose(conv.weight[1], original[1])  # unmasked view matches original


def test_commit_mask_bakes_zeros_in_permanently():
    conv = nn.Conv2d(3, 4, kernel_size=3)
    conv.bias.data.fill_(3.0)
    mask_channels(conv, torch.tensor([1, 3]))
    commit_mask(conv)

    assert not prune.is_pruned(conv)
    assert not hasattr(conv, "weight_orig")  # reparametrization fully removed
    assert not hasattr(conv, "bias_orig")
    assert torch.allclose(conv.weight[1], torch.zeros_like(conv.weight[1]))
    assert torch.allclose(conv.weight[3], torch.zeros_like(conv.weight[3]))
    assert torch.allclose(conv.bias[[1, 3]], torch.zeros(2))
    assert torch.allclose(conv.bias[[0, 2]], torch.full((2,), 3.0))


def test_commit_mask_is_a_noop_on_unmasked_module():
    conv = nn.Conv2d(3, 4, kernel_size=3)
    commit_mask(conv)  # must not raise
    assert not prune.is_pruned(conv)


def test_zeroed_and_surviving_channels_are_complements():
    weight = torch.randn(6, 3, 3, 3)
    weight[[1, 4]] = 0.0
    zeroed = zeroed_channels(weight)
    survivors = surviving_channels(weight)

    assert zeroed.tolist() == [1, 4]
    assert survivors.tolist() == [0, 2, 3, 5]


def test_compress_masked_conv_bn_matches_direct_surgery():
    """The whole point of the two-phase design: mask, commit, then compress
    should produce a numerically identical result to calling prune_conv_bn
    directly with the same keep indices. This is what proves 'copy the
    unmasked weights' and 'prune these channels directly' are the same
    operation, just reached via a different (safer, fine-tunable) path.
    """
    torch.manual_seed(0)
    conv = nn.Conv2d(4, 8, kernel_size=3, padding=1)
    bn = nn.BatchNorm2d(8)
    next_conv = nn.Conv2d(8, 5, kernel_size=3, padding=1)
    prune_idx = torch.tensor([2, 5, 7])
    keep_idx_expected = torch.tensor([0, 1, 3, 4, 6])

    # Path A: direct surgery (what prunelib.prune_conv_bn does on its own).
    conv_a = nn.Conv2d(4, 8, kernel_size=3, padding=1)
    conv_a.load_state_dict(conv.state_dict())
    bn_a = nn.BatchNorm2d(8)
    bn_a.load_state_dict(bn.state_dict())
    next_conv_a = nn.Conv2d(8, 5, kernel_size=3, padding=1)
    next_conv_a.load_state_dict(next_conv.state_dict())
    new_conv_a, new_bn_a, new_next_a = prune_conv_bn(conv_a, keep_idx_expected, bn=bn_a, next_conv=next_conv_a)

    # Path B: mask -> commit -> compress.
    mask_channels(conv, prune_idx)
    commit_mask(conv)
    new_conv_b, new_bn_b, new_next_b, keep_idx_b = compress_masked_conv_bn(conv, bn=bn, next_conv=next_conv)

    assert keep_idx_b.tolist() == keep_idx_expected.tolist()
    assert torch.allclose(new_conv_a.weight, new_conv_b.weight)
    assert torch.allclose(new_bn_a.running_mean, new_bn_b.running_mean)
    assert torch.allclose(new_next_a.weight, new_next_b.weight)


def test_compress_raises_if_mask_not_committed():
    conv = nn.Conv2d(4, 8, kernel_size=3)
    mask_channels(conv, torch.tensor([1]))
    try:
        compress_masked_conv_bn(conv)
        assert False, "expected a ValueError for an uncommitted mask"
    except ValueError:
        pass


def test_iterative_masking_composes():
    """A second mask_channels call on an already-masked layer (the normal
    case across pruning iterations) should compose with, not replace, the
    first mask -- this is torch.nn.utils.prune's standard behavior and is
    what makes it safe to call mask_channels once per iteration in a loop."""
    conv = nn.Conv2d(3, 6, kernel_size=3)
    mask_channels(conv, torch.tensor([0, 1]))
    mask_channels(conv, torch.tensor([2]))
    commit_mask(conv)

    zeroed = set(zeroed_channels(conv.weight).tolist())
    assert zeroed == {0, 1, 2}
