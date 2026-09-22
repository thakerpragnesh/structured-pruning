import pytest
import torch
import torch.nn as nn

from prunelib.graph import DependencyGraph, LeafTracer, prune_model
from prunelib.surgery import prune_attention_heads, prune_conv_bn


def test_sequential_chain_matches_direct_surgery():
    """A plain conv-bn-conv chain: DependencyGraph should find exactly the
    coupling prune_conv_bn(conv, keep_idx, bn=bn, next_conv=next_conv)
    already handles by hand, and produce numerically identical weights."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(4, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 5, kernel_size=3, padding=1),
    )
    x = torch.randn(1, 4, 8, 8)
    keep_idx = torch.tensor([0, 1, 3, 4, 6])

    conv_a = nn.Conv2d(4, 8, kernel_size=3, padding=1)
    conv_a.load_state_dict(model[0].state_dict())
    bn_a = nn.BatchNorm2d(8)
    bn_a.load_state_dict(model[1].state_dict())
    next_a = nn.Conv2d(8, 5, kernel_size=3, padding=1)
    next_a.load_state_dict(model[2].state_dict())
    new_conv_a, new_bn_a, new_next_a = prune_conv_bn(conv_a, keep_idx, bn=bn_a, next_conv=next_a)

    dep = DependencyGraph(model, x)
    group = dep.get_pruning_group(model[0], keep_idx)
    group.prune()

    assert torch.allclose(model[0].weight, new_conv_a.weight)
    assert torch.allclose(model[1].weight, new_bn_a.weight)
    assert torch.allclose(model[2].weight, new_next_a.weight)

    out = model(x)
    assert out.shape == (1, 5, 8, 8)


class TinyResidualBlock(nn.Module):
    """A shortcut-projection residual block, the ResNet BasicBlock shape."""

    def __init__(self, channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut_conv = nn.Conv2d(channels, out_channels, 1)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = self.shortcut_bn(self.shortcut_conv(x))
        return self.relu(out + identity)


def test_residual_add_couples_shortcut_conv_to_main_path():
    torch.manual_seed(0)
    model = TinyResidualBlock(channels=4, out_channels=8)
    x = torch.randn(2, 4, 6, 6)
    dep = DependencyGraph(model, x)
    keep_idx = torch.tensor([0, 2, 3, 5, 6])  # keep 5 of 8

    group = dep.get_pruning_group(model.conv2, keep_idx)
    assert set(group.output_targets) == {"conv2", "bn2", "shortcut_conv", "shortcut_bn"}
    idx_sets = [torch.sort(v).values for v in group.output_targets.values()]
    assert all(torch.equal(idx_sets[0], s) for s in idx_sets[1:])  # every branch agrees on which channels survive

    group.prune()
    out = model(x)
    assert out.shape == (2, 5, 6, 6)


class IdentitySkipBlock(nn.Module):
    """A bare-identity skip (no shortcut conv): the add's other operand is
    literally the block's own input, produced further back than the
    immediate block."""

    def __init__(self, channels):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.stem(x)
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


def test_identity_skip_traces_back_to_the_actual_producing_conv():
    torch.manual_seed(0)
    model = IdentitySkipBlock(channels=6)
    x = torch.randn(1, 3, 8, 8)
    dep = DependencyGraph(model, x)
    keep_idx = torch.tensor([0, 1, 2, 4])  # keep 4 of 6, pruning conv2's output

    group = dep.get_pruning_group(model.conv2, keep_idx)
    assert set(group.output_targets) == {"conv2", "bn2", "stem"}
    idx_sets = [torch.sort(v).values for v in group.output_targets.values()]
    assert all(torch.equal(idx_sets[0], s) for s in idx_sets[1:])
    # stem's output also feeds conv1 directly -- its input must shrink to match stem's new output.
    assert "conv1" in group.input_targets
    assert torch.equal(torch.sort(group.input_targets["conv1"]).values, idx_sets[0])

    group.prune()
    out = model(x)
    assert out.shape == (1, 4, 8, 8)


def test_two_branches_disagreeing_on_a_shared_layer_raises():
    """If two different index sets would both be assigned to the same shared
    layer, that's a real conflict -- DependencyGraph must raise, not silently
    keep whichever arrived first."""
    torch.manual_seed(0)
    model = TinyResidualBlock(channels=4, out_channels=8)
    x = torch.randn(1, 4, 6, 6)
    dep = DependencyGraph(model, x)
    group = dep.get_pruning_group(model.conv2, torch.tensor([0, 1, 2, 3, 4]))
    with pytest.raises(ValueError, match="disagree"):
        # Same shortcut_conv, but a different (conflicting) index set this time.
        dep._record(group.output_targets, "shortcut_conv", torch.tensor([0, 1, 2, 3, 5]))


class ConcatBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Conv2d(3, 4, 3, padding=1)
        self.branch_b = nn.Conv2d(3, 6, 3, padding=1)
        self.after = nn.Conv2d(10, 5, 3, padding=1)  # 4 (a) + 6 (b) = 10

    def forward(self, x):
        merged = torch.cat([self.branch_a(x), self.branch_b(x)], dim=1)
        return self.after(merged)


def test_cat_offsets_downstream_input_indices_for_the_second_branch():
    torch.manual_seed(0)
    model = ConcatBranches()
    x = torch.randn(1, 3, 8, 8)
    dep = DependencyGraph(model, x)
    keep_idx = torch.tensor([0, 2, 3, 5])  # keep 4 of branch_b's 6 channels
    prune_idx = torch.tensor([1, 4])  # complement within branch_b's own 6
    expected = prune_idx + 4  # branch_a contributes the concat's first 4 channels

    group = dep.get_pruning_group(model.branch_b, keep_idx)
    assert torch.equal(torch.sort(group.input_targets["after"]).values, torch.sort(expected).values)

    group.prune()
    assert model.after.in_channels == 8  # 4 (branch_a, untouched) + 4 (branch_b, kept)
    out = model(x)
    assert out.shape == (1, 5, 8, 8)


class ConvThenClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(4 * 2 * 2, 10)

    def forward(self, x):
        x = self.flatten(self.pool(self.conv(x)))
        return self.fc(x)


def test_flatten_boundary_expands_channel_indices_to_flat_indices():
    torch.manual_seed(0)
    model = ConvThenClassifier()
    x = torch.randn(1, 3, 8, 8)
    dep = DependencyGraph(model, x)
    keep_idx = torch.tensor([0, 2])  # keep 2 of conv's 4 output channels
    prune_idx = torch.tensor([1, 3])
    spatial = 2 * 2
    expected_flat = torch.cat([torch.arange(c * spatial, (c + 1) * spatial) for c in prune_idx.tolist()])

    group = dep.get_pruning_group(model.conv, keep_idx)
    assert torch.equal(torch.sort(group.input_targets["fc"]).values, torch.sort(expected_flat).values)

    group.prune()
    assert model.fc.in_features == 2 * spatial
    out = model(x)
    assert out.shape == (1, 10)


class DepthwiseSandwich(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.entry = nn.Conv2d(3, channels, 3, padding=1)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.exit = nn.Conv2d(channels, 5, 3, padding=1)

    def forward(self, x):
        return self.exit(self.dw(self.entry(x)))


def test_depthwise_conv_passes_indices_through_in_both_directions():
    torch.manual_seed(0)
    model = DepthwiseSandwich(channels=6)
    x = torch.randn(1, 3, 8, 8)
    dep = DependencyGraph(model, x)
    keep_idx = torch.tensor([0, 1, 3, 5])  # keep 4 of 6

    group = dep.get_pruning_group(model.entry, keep_idx)
    assert "dw" in group.output_targets
    assert torch.equal(torch.sort(group.output_targets["dw"]).values, torch.sort(group.output_targets["entry"]).values)
    assert "exit" in group.input_targets

    group.prune()
    assert model.dw.groups == 4
    assert model.dw.in_channels == 4 and model.dw.out_channels == 4
    out = model(x)
    assert out.shape == (1, 5, 8, 8)


class GroupedConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(4, 8, 3, padding=1)
        self.grouped = nn.Conv2d(8, 8, 3, padding=1, groups=2)  # groups>1, not depthwise

    def forward(self, x):
        return self.grouped(self.entry(x))


def test_non_depthwise_grouped_conv_raises_not_implemented():
    model = GroupedConvBlock()
    x = torch.randn(1, 4, 8, 8)
    dep = DependencyGraph(model, x)
    with pytest.raises(NotImplementedError):
        dep.get_pruning_group(model.entry, torch.tensor([0, 2, 4, 6]))


class UnknownDownstream(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.gru_cell = nn.GRUCell(4, 7)  # a width DependencyGraph has no reason to expect

    def forward(self, x):
        x = self.flatten(self.pool(self.conv(x)))
        return self.gru_cell(x)


def test_unrecognized_module_type_raises_instead_of_silently_passing_through():
    model = UnknownDownstream()
    x = torch.randn(1, 3, 8, 8)
    dep = DependencyGraph(model, x)
    with pytest.raises(NotImplementedError):
        dep.get_pruning_group(model.conv, torch.tensor([0, 2]))


def test_output_prune_values_are_correct_not_just_shape():
    """Mirrors test_d6_conv_values_are_correct_not_just_shape's reasoning for
    the generic path: verify actual kept values, not just resulting shapes."""
    conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)
    with torch.no_grad():
        for i in range(6):
            conv.weight[i] = float(i)
            conv.bias[i] = float(i) * 10
    next_conv = nn.Conv2d(6, 4, kernel_size=3, padding=1)
    model = nn.Sequential(conv, next_conv)
    x = torch.randn(1, 3, 8, 8)
    dep = DependencyGraph(model, x)
    keep_idx = torch.tensor([1, 3, 5])

    group = dep.get_pruning_group(model[0], keep_idx)
    group.prune()

    assert model[0].out_channels == 3
    for new_i, old_i in enumerate(keep_idx.tolist()):
        assert torch.allclose(model[0].weight[new_i], torch.full((3, 3, 3), float(old_i)))
        assert torch.isclose(model[0].bias[new_i], torch.tensor(float(old_i) * 10))


def test_prune_model_scores_and_prunes_in_one_call():
    """`prune_model` is the generic, any-architecture counterpart to
    `vgg.py::prune_vgg_layer` -- score, select, and prune a layer plus
    everything it's coupled to, in one call, with no seam passed by hand."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(4, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 5, kernel_size=3, padding=1),
    )
    x = torch.randn(1, 4, 8, 8)

    group = prune_model(model, x, model[0], prune_fraction=0.5, method="l1")

    assert model[0].out_channels == 4
    assert model[1].num_features == 4
    assert model[2].in_channels == 4
    assert set(group.output_targets) | set(group.input_targets) == {"0", "1", "2"}
    out = model(x)
    assert out.shape == (1, 5, 8, 8)


def test_prune_model_rejects_out_of_range_fraction():
    model = nn.Sequential(nn.Conv2d(4, 8, kernel_size=3, padding=1))
    x = torch.randn(1, 4, 8, 8)
    with pytest.raises(ValueError, match="prune_fraction"):
        prune_model(model, x, model[0], prune_fraction=1.0)


def test_mask_then_commit_and_compress_matches_direct_prune():
    """`PruningGroup.mask()` + `.commit_and_compress()` must produce exactly
    what calling `.prune()` directly would -- mirrors
    test_masking.py::test_compress_masked_conv_bn_matches_direct_surgery's
    reasoning, one level up (a whole dependency group, not one layer)."""

    def build():
        torch.manual_seed(1)
        return nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 5, kernel_size=3, padding=1),
        )

    model_direct = build()
    model_masked = build()
    x = torch.randn(1, 4, 8, 8)
    keep_idx = torch.tensor([0, 1, 3, 4, 6])

    dep_direct = DependencyGraph(model_direct, x)
    dep_direct.get_pruning_group(model_direct[0], keep_idx).prune()

    dep_masked = DependencyGraph(model_masked, x)
    group = dep_masked.get_pruning_group(model_masked[0], keep_idx)
    group.mask()
    # Shapes are unchanged while masked -- the model still runs at full width.
    out_masked = model_masked(x)
    assert out_masked.shape == (1, 5, 8, 8)
    assert model_masked[0].out_channels == 8
    group.commit_and_compress()

    assert model_masked[0].out_channels == 5  # keep_idx has 5 of 8 original channels
    assert torch.allclose(model_direct[0].weight, model_masked[0].weight)
    assert torch.allclose(model_direct[1].weight, model_masked[1].weight)
    assert torch.allclose(model_direct[2].weight, model_masked[2].weight)
    out = model_masked(x)
    assert out.shape == (1, 5, 8, 8)


class TinyAttentionBlock(nn.Module):
    """Not a realistic attention computation -- just four Linears wired
    together the way `surgery.prune_attention_heads` expects (Q/K/V output
    rows, output-projection input columns, all partitioned by head), enough
    to exercise the `special_handlers` hook without needing `transformers`
    installed."""

    def __init__(self, hidden: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        return self.out(q + k + v)


class AttentionWrapper(nn.Module):
    def __init__(self, hidden: int, num_heads: int):
        super().__init__()
        self.attn = TinyAttentionBlock(hidden, num_heads)

    def forward(self, x):
        return self.attn(x)


def _prune_tiny_attention_block(module: TinyAttentionBlock, keep_heads: torch.Tensor) -> TinyAttentionBlock:
    new_q, new_k, new_v, new_out = prune_attention_heads(
        module.query, module.key, module.value, module.out,
        keep_heads=keep_heads, num_heads=module.num_heads,
    )
    module.query, module.key, module.value, module.out = new_q, new_k, new_v, new_out
    module.num_heads = keep_heads.numel()
    return module


def test_special_handler_prunes_attention_block_via_prune_attention_heads():
    """`special_handlers` is the extension point KT.md calls out for
    attention-block pruning: DependencyGraph can't work out Q/K/V/output
    rewiring on its own (that's the "much harder problem" its module
    docstring disclaims), but a registered handler can act on it as the
    *starting* layer of a `get_pruning_group` call."""
    torch.manual_seed(0)
    model = AttentionWrapper(hidden=8, num_heads=4)
    x = torch.randn(2, 8)

    dep = DependencyGraph(
        model, x,
        tracer=LeafTracer([TinyAttentionBlock]),
        special_handlers={TinyAttentionBlock: _prune_tiny_attention_block},
    )
    group = dep.get_pruning_group(model.attn, torch.tensor([0, 1, 3]))  # keep 3 of 4 heads
    group.prune()

    assert model.attn.num_heads == 3
    assert model.attn.query.out_features == 6  # 3 heads * head_dim(2)
    assert model.attn.out.in_features == 6
    out = model(x)
    assert out.shape == (2, 8)  # hidden size at the block's boundary is unchanged by head pruning
