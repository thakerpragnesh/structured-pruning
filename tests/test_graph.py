import copy

import pytest
import torch
import torch.nn as nn

from prunelib import DependencyGraph, LeafTracer, prune_count, prune_model
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
        group.add_output_target("shortcut_conv", torch.tensor([0, 1, 2, 3, 5]))


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


def test_reused_graph_offsets_a_cat_by_each_branchs_current_width():
    """One graph is meant to cover a pruning session, but it kept the shapes
    of its first trace. After `branch_a` shrank from 4 channels to 2, a prune
    of `branch_b`'s channel 0 was offset by `branch_a`'s old width (column 4
    of `after`, i.e. `branch_b`'s channel 2) instead of its current one
    (column 2), and nothing failed."""
    torch.manual_seed(0)
    model = ConcatBranches()
    x = torch.randn(1, 3, 8, 8)
    dep = DependencyGraph(model, x)
    dep.get_pruning_group(model.branch_a, torch.tensor([0, 1])).prune()  # after: 2 (a) + 6 (b) = 8 inputs
    after = model.after.weight.detach().clone()

    group = dep.get_pruning_group(model.branch_b, torch.tensor([1, 2, 3, 4, 5]))
    assert group.input_targets["after"].tolist() == [2]
    group.prune()
    assert torch.equal(model.after.weight, after[:, [0, 1, 3, 4, 5, 6, 7]])


def test_reused_graph_accepts_a_module_an_earlier_prune_swapped_in():
    """The graph mapped module objects to names when it was built, so the
    module a prune put in a layer's place -- the one `model[2]` returns from
    then on -- was "not a submodule of the traced model"."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 6, 3, padding=1), nn.ReLU(), nn.Conv2d(6, 2, 1))
    x = torch.randn(1, 3, 6, 6)
    dep = DependencyGraph(model, x)
    prune_model(model, x, model[0], prune_fraction=0.25, dependency_graph=dep)
    prune_model(model, x, model[2], prune_fraction=0.5, dependency_graph=dep)  # model[2] was rebuilt by the first prune

    assert (model[0].out_channels, model[2].in_channels, model[2].out_channels, model[4].in_channels) == (6, 6, 3, 3)
    with pytest.raises(ValueError, match="not a submodule"):
        dep.get_pruning_group(nn.Conv2d(3, 8, 3), torch.tensor([0]))


def test_tracing_leaves_batchnorm_statistics_and_train_modes_alone():
    """ShapeProp's forward pass ran in the model's own mode, so tracing a
    model in training mode nudged every BatchNorm's running statistics --
    and a graph now re-traces after each prune that resizes something."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(), nn.Conv2d(4, 2, 1), nn.BatchNorm2d(2))
    model[4].eval()  # a frozen BatchNorm inside a training model: mixed modes come back as they were
    stats = {name: t.clone() for name, t in model.named_buffers()}

    dep = DependencyGraph(model, torch.randn(2, 3, 6, 6))
    dep.get_pruning_group("0", torch.tensor([0, 1, 3])).prune()
    dep.get_pruning_group("3", torch.tensor([1]))  # resized since the first trace: re-traces

    assert model.training and model[1].training and not model[4].training
    assert torch.equal(model[1].running_mean, stats["1.running_mean"][[0, 1, 3]])
    assert all(torch.equal(t, stats[name]) for name, t in model.named_buffers() if not name.startswith("1."))


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


class _FlattenInto(nn.Module):
    def __init__(self, flatten):
        super().__init__()
        self.flatten = flatten
        self.a, self.pool, self.fc = nn.Conv2d(3, 4, 3, padding=1), nn.AdaptiveAvgPool2d(2), nn.Linear(16, 5)

    def forward(self, x):
        return self.fc(self.flatten(self.pool(self.a(x))))


@pytest.mark.parametrize("flatten", [lambda y: y.view(y.size(0), -1), lambda y: y.reshape(y.shape[0], -1)])
def test_classifier_flatten_written_with_the_batch_size_is_recognized(flatten):
    """`x.view(x.size(0), -1)` is the usual way to flatten into a classifier
    head, but the walk reached the `size` call (a user of the pruned tensor)
    and refused it. Reading a shape carries no channels, so it is skipped."""
    torch.manual_seed(0)
    model = _FlattenInto(flatten).eval()
    x = torch.randn(2, 3, 8, 8)
    group = DependencyGraph(model, x).get_pruning_group("a", torch.tensor([0, 2]))
    assert group.input_targets["fc"].tolist() == [4, 5, 6, 7, 12, 13, 14, 15]
    group.prune()
    assert model(x).shape == (2, 5)


def test_a_reshape_that_isnt_the_classifier_flatten_is_refused_at_the_reshape():
    """Every view/reshape/flatten of a 4D tensor was taken for the
    classifier-head flatten. `x.flatten(2)` keeps the channel dimension,
    yet its indices were expanded by `H * W`, and the walk failed later at
    whatever came next, naming that layer instead of the reshape."""
    model = _FlattenInto(lambda y: torch.flatten(y, 2).mean(-1))
    model.fc = nn.Linear(4, 5)
    with pytest.raises(NotImplementedError, match=r"reshaping \(1, 4, 2, 2\) to \(1, 4, 4\) isn't"):
        DependencyGraph(model, torch.randn(1, 3, 8, 8)).get_pruning_group("a", torch.tensor([0, 2]))


class _ActivationForms(nn.Module):
    def __init__(self, form):
        super().__init__()
        self.form = form
        self.a, self.b = nn.Conv2d(3, 6, 3, padding=1), nn.Conv2d(6, 2, 1)
        self.act, self.pool = nn.ReLU(), nn.MaxPool2d(2)

    def forward(self, x):
        y = self.a(x)
        if self.form == "module":
            y = self.pool(self.act(y))
        elif self.form == "function":
            y = nn.functional.max_pool2d(nn.functional.relu(y), 2)
        else:
            y = nn.functional.max_pool2d(y.relu(), 2)
        return self.b(y)


@pytest.mark.parametrize("form", ["function", "method"])
def test_functional_activations_prune_like_their_modules(form):
    """Liskov: `nn.ReLU()` and `nn.MaxPool2d` passed a prune through, but
    the same ops written `F.relu(y)`, `y.relu()` or `F.max_pool2d(y, 2)`
    raised, so whether a model could be pruned depended on how its
    activations were spelled. Both forms now prune identically."""
    x = torch.randn(1, 3, 8, 8)
    keep = torch.tensor([0, 2, 3, 5])
    torch.manual_seed(0)
    reference = _ActivationForms("module")
    torch.manual_seed(0)
    model = _ActivationForms(form)
    for m in (reference, model):
        DependencyGraph(m, x).get_pruning_group("a", keep).prune()
    assert torch.equal(model.a.weight, reference.a.weight) and torch.equal(model.b.weight, reference.b.weight)
    assert torch.equal(model(x), reference(x))


class _ScaledAndGated(nn.Module):
    """`a`'s output scaled by constants, then gated by a 1-channel spatial
    map that broadcasts across every channel."""

    def __init__(self):
        super().__init__()
        self.a, self.gate, self.b = nn.Conv2d(3, 6, 1), nn.Conv2d(3, 1, 1), nn.Conv2d(6, 2, 1)

    def forward(self, x):
        return self.b((self.a(x) * 0.5 + 1) * torch.sigmoid(self.gate(x)))


def test_scalars_and_channel_broadcast_operands_share_no_channels():
    """`x * 0.5` and `x + 1` were refused (`*` had no propagator; `+` wanted
    a second tensor), and `+` with a `[N, 1, H, W]` map coupled the map's
    1-channel conv as if it had `a`'s channels. Neither shares channels
    with `a`, so the prune passes them by."""
    torch.manual_seed(0)
    model = _ScaledAndGated().eval()
    x = torch.randn(2, 3, 5, 5)
    reference = copy.deepcopy(model)
    with torch.no_grad():
        reference.b.weight[:, [1, 4]] = 0

    group = DependencyGraph(model, x).get_pruning_group("a", torch.tensor([0, 2, 3, 5]))
    assert set(group.output_targets) == {"a"} and set(group.input_targets) == {"b"}
    group.prune()
    assert model.gate.out_channels == 1 and torch.allclose(model(x), reference(x), atol=1e-6)


class _SqueezeExcite(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 6, 3, padding=1)
        self.fc1, self.fc2 = nn.Conv2d(6, 3, 1), nn.Conv2d(3, 6, 1)
        self.b = nn.Conv2d(6, 2, 1)

    def forward(self, x):
        y = torch.relu(self.a(x))
        scale = torch.sigmoid(self.fc2(torch.relu(self.fc1(nn.functional.adaptive_avg_pool2d(y, 1)))))
        return self.b(y * scale)


def test_squeeze_excitation_block_prunes_to_the_same_function():
    """`y * scale` was refused, so no squeeze-and-excitation block
    (MobileNetV3's, EfficientNet's) could be pruned through. Pruning `a`
    now also removes those channels from `fc1`'s input and `fc2`'s output,
    and the result computes exactly what the original does once the
    channels are cut off downstream."""
    torch.manual_seed(0)
    model = _SqueezeExcite().eval()
    x = torch.randn(2, 3, 6, 6)
    reference = copy.deepcopy(model)
    with torch.no_grad():
        reference.b.weight[:, [1, 4]] = 0
        reference.fc1.weight[:, [1, 4]] = 0

    group = DependencyGraph(model, x).get_pruning_group("a", torch.tensor([0, 2, 3, 5]))
    assert set(group.output_targets) == {"a", "fc2"} and set(group.input_targets) == {"fc1", "b"}
    group.prune()
    assert torch.allclose(model(x), reference(x), atol=1e-6)


def test_mobilenet_v3_prunes_through_its_squeeze_excitation_blocks():
    """The same on a real architecture: each MobileNetV3 block gates its
    depthwise conv's output with `scale * input`, written as a plain `*`.
    Pruning a block's expansion conv reaches the depthwise conv, the
    squeeze-and-excitation convs and the projection, and the model runs."""
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    model = torchvision.models.mobilenet_v3_small(weights=None).eval()
    x = torch.randn(1, 3, 64, 64)

    group = prune_model(model, x, "features.4.block.0.0", prune_fraction=0.5)
    block = "features.4.block."
    assert set(group.output_targets) == {block + n for n in ("0.0", "0.1", "1.0", "1.1", "2.fc2")}
    assert set(group.input_targets) == {block + "2.fc1", block + "3.0"}
    assert model(x).shape == (1, 1000)


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


def test_prune_model_always_keeps_at_least_one_channel():
    """A budget just under 1 used to round up to every channel, leaving a
    zero-width Linear that still ran -- the next layer just saw no input."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    prune_model(model, torch.randn(2, 4), model[0], prune_fraction=0.95)
    assert model[0].out_features == 1 and model[2].in_features == 1


def test_prune_count_rounds_one_way_and_each_caller_sets_its_bounds():
    assert prune_count(8, 0.95, min_prune=0) == 7  # prune_model: may prune none, always keeps one
    assert prune_count(8, 0.0, min_prune=0) == 0
    assert prune_count(8, 0.01) == 1  # prune_vgg_layer: always prunes one...
    assert prune_count(8, 1.0) == 7  # ...and always keeps one
    assert prune_count(64, 0.1, available=3, min_keep=0) == 3  # mask_vgg_layer: budget from the original width, capped by survivors
    assert prune_count(1, 0.5) == 0  # nothing prunable is 0, never negative


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


def test_mask_refuses_a_non_affine_batchnorm_and_prune_still_works():
    """`.mask()` then `.commit_and_compress()` is meant to equal `.prune()`,
    but on a `BatchNorm2d(affine=False)` `.mask()` died inside
    `torch.ones_like` (no weight to mask) while `.prune()` worked. Such a
    BatchNorm can't be masked at all -- its running mean turns a zeroed
    channel into a non-zero constant -- so its rule now refuses, naming the
    layer, and `.prune()` still matches a direct prune even though the conv
    before it was masked before the refusal."""

    def build():
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8, affine=False), nn.Conv2d(8, 4, 3, padding=1)
        ).eval()
        model[1].running_mean.uniform_(-1, 1)
        return model

    x = torch.randn(1, 3, 8, 8)
    keep = torch.tensor([0, 2, 4, 6])
    direct = build()
    DependencyGraph(direct, x).get_pruning_group(direct[0], keep).prune()

    model = build()
    group = DependencyGraph(model, x).get_pruning_group(model[0], keep)
    with pytest.raises(NotImplementedError, match=r"'1'.*affine=False"):
        group.mask()
    group.prune()
    assert torch.allclose(model(x), direct(x))


class _MovedSkip(nn.Module):
    """A residual whose skip branch moves channels around before the add:
    built with `torch.cat`, or split off a wider conv with `torch.chunk`."""

    def __init__(self, chunk: bool):
        super().__init__()
        self.chunk = chunk
        self.main = nn.Conv2d(3, 6, 1)
        self.a, self.b = nn.Conv2d(3, 2, 1), nn.Conv2d(3, 4, 1)
        self.wide = nn.Conv2d(3, 12, 1)
        self.head = nn.Conv2d(6, 2, 1)

    def forward(self, x):
        skip = torch.chunk(self.wide(x), 2, 1)[0] if self.chunk else torch.cat([self.a(x), self.b(x)], 1)
        return self.head(self.main(x) + skip)


@pytest.mark.parametrize("chunk, op", [(False, "cat"), (True, "getitem")])
def test_skip_branch_that_moves_channels_is_refused_at_that_op(chunk, op):
    """The walk back to a skip branch's producer stepped through any
    function call as if channel indices carried over unchanged. Through a
    `cat` it pinned `main`'s indices on the last conv concatenated, then
    failed with a misleading "two branches disagree". It now refuses at the
    op that moves the channels."""
    model = _MovedSkip(chunk).eval()
    dep = DependencyGraph(model, torch.randn(1, 3, 4, 4))
    with pytest.raises(NotImplementedError, match=f"skip connection back through '{op}'"):
        dep.get_pruning_group("main", torch.tensor([0, 1, 2, 3]))


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


def test_special_handler_group_refuses_to_mask():
    """A handler rebuilds its block in one step, so `.mask()` used to skip
    it silently: the caller fine-tuned believing the pruned heads were
    masked when nothing had happened to them."""
    model = AttentionWrapper(hidden=8, num_heads=4)
    x = torch.randn(2, 8)
    dep = DependencyGraph(
        model, x,
        tracer=LeafTracer([TinyAttentionBlock]),
        special_handlers={TinyAttentionBlock: _prune_tiny_attention_block},
    )
    group = dep.get_pruning_group(model.attn, torch.tensor([0, 1, 3]))

    with pytest.raises(NotImplementedError, match="special handler"):
        group.mask()
    assert not hasattr(model.attn.query, "weight_orig")  # nothing was half-masked
    group.prune()  # the one-step path still works
    assert model.attn.num_heads == 3


class TinyAttentionBlockSubclass(TinyAttentionBlock):
    pass


def test_special_handler_also_matches_subclasses_of_the_registered_type():
    """Liskov: a subclass of a registered block type must be handled like
    its parent. `special_handlers` used to be keyed on the exact `type()`,
    while `LeafTracer` already used `isinstance` -- so a subclass traced as a
    leaf, then fell through to "expected Conv2d or Linear"."""
    torch.manual_seed(0)
    model = AttentionWrapper(hidden=8, num_heads=4)
    model.attn = TinyAttentionBlockSubclass(hidden=8, num_heads=4)
    x = torch.randn(2, 8)

    dep = DependencyGraph(
        model, x,
        tracer=LeafTracer([TinyAttentionBlock]),
        special_handlers={TinyAttentionBlock: _prune_tiny_attention_block},
    )
    dep.get_pruning_group(model.attn, torch.tensor([1, 2])).prune()

    assert model.attn.num_heads == 2
    assert model(x).shape == (2, 8)


def test_depthwise_conv_cannot_be_the_starting_layer():
    """A depthwise conv only carries the channels its producer emits, so
    pruning it alone left the producer's output wider than the depthwise
    conv's new input -- a model that failed on its next forward pass. Start
    from the producer instead (see the depthwise sandwich test above)."""
    model = DepthwiseSandwich(channels=6)
    dep = DependencyGraph(model, torch.randn(1, 3, 8, 8))
    with pytest.raises(TypeError, match="start from the layer that feeds it"):
        dep.get_pruning_group(model.dw, torch.tensor([0, 1, 3, 5]))
