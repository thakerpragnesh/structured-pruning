import pytest
import torch
import torch.nn as nn

from prunelib.surgery import prune_conv_bn, prune_ffn_block


def test_d6_conv_values_are_correct_not_just_shape():
    """D6: `deep_model_copy_channelwise` never incremented its destination
    index, so every surviving channel got written to index 0 -- shapes could
    look right while every value except the first was garbage. Verify actual
    values at each kept index match the source, not just the output shape."""
    conv = nn.Conv2d(3, 6, kernel_size=3, bias=True)
    with torch.no_grad():
        for i in range(6):
            conv.weight[i] = float(i)
            conv.bias[i] = float(i) * 10

    keep_idx = torch.tensor([1, 3, 5])
    new_conv, _, _ = prune_conv_bn(conv, keep_idx)

    assert new_conv.out_channels == 3
    for new_i, old_i in enumerate(keep_idx.tolist()):
        assert torch.allclose(new_conv.weight[new_i], conv.weight[old_i])
        assert torch.isclose(new_conv.bias[new_i], conv.bias[old_i])


def test_d6_bn_running_stats_carry_across():
    """The original omitted BatchNorm running-stat migration entirely, so a
    pruned model's forward pass ran but produced silently wrong activations."""
    conv = nn.Conv2d(4, 4, kernel_size=3)
    bn = nn.BatchNorm2d(4)
    with torch.no_grad():
        bn.running_mean.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        bn.running_var.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))

    keep_idx = torch.tensor([0, 2])
    _, new_bn, _ = prune_conv_bn(conv, keep_idx, bn=bn)

    assert torch.allclose(new_bn.running_mean, torch.tensor([1.0, 3.0]))
    assert torch.allclose(new_bn.running_var, torch.tensor([0.1, 0.3]))


def test_conv_chain_shrinks_next_layer_input_to_match():
    conv = nn.Conv2d(3, 8, kernel_size=3)
    next_conv = nn.Conv2d(8, 5, kernel_size=3)
    keep_idx = torch.tensor([0, 2, 4, 6])

    new_conv, _, new_next = prune_conv_bn(conv, keep_idx, next_conv=next_conv)

    assert new_conv.out_channels == 4
    assert new_next.in_channels == 4
    assert new_next.out_channels == 5  # downstream output width is untouched
    assert torch.allclose(new_next.weight, next_conv.weight.index_select(1, keep_idx))


def test_conv_chain_seam_mismatch_raises():
    conv = nn.Conv2d(3, 8, kernel_size=3)
    mismatched_next = nn.Conv2d(999, 5, kernel_size=3)  # wrong in_channels on purpose
    with pytest.raises(ValueError):
        prune_conv_bn(conv, torch.tensor([0, 1]), next_conv=mismatched_next)


def test_d5_ffn_seam_validated_before_surgery():
    """D5: `compute_distance_score_channel` raised a confusing IndexError deep
    inside a loop rather than failing at the actual point of the mistake.
    `prune_ffn_block` should refuse mismatched shapes immediately."""
    fc1 = nn.Linear(16, 64)
    fc2 = nn.Linear(999, 16)  # doesn't match fc1's output width
    with pytest.raises(ValueError):
        prune_ffn_block(fc1, fc2, keep_idx=torch.arange(32))


def test_ffn_block_shrinks_and_preserves_values():
    fc1 = nn.Linear(8, 32)
    fc2 = nn.Linear(32, 8)
    keep_idx = torch.tensor([1, 5, 9, 17, 30])

    new_fc1, new_fc2 = prune_ffn_block(fc1, fc2, keep_idx)

    assert new_fc1.out_features == 5
    assert new_fc2.in_features == 5
    assert torch.allclose(new_fc1.weight, fc1.weight.index_select(0, keep_idx))
    assert torch.allclose(new_fc2.weight, fc2.weight.index_select(1, keep_idx))

    # End-to-end forward pass must actually run (proves the seam is real, not
    # just shape-compatible by coincidence).
    x = torch.randn(2, 8)
    out = new_fc2(torch.relu(new_fc1(x)))
    assert out.shape == (2, 8)
