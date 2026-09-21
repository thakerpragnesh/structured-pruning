import pytest
import torch
import torch.nn as nn

from prunelib.surgery import prune_attention_heads, prune_conv_bn, prune_ffn_block


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


def _qkvo(hidden=8, bias=True):
    torch.manual_seed(0)
    query = nn.Linear(hidden, hidden, bias=bias)
    key = nn.Linear(hidden, hidden, bias=bias)
    value = nn.Linear(hidden, hidden, bias=bias)
    output = nn.Linear(hidden, hidden, bias=bias)
    return query, key, value, output


def _mha_forward(query, key, value, output, num_heads, x):
    """Minimal real multi-head self-attention forward, used to prove
    `prune_attention_heads`' output actually composes -- not just that shapes
    are compatible."""
    head_dim = query.out_features // num_heads

    def split(t):
        return t.view(*t.shape[:-1], num_heads, head_dim).transpose(-3, -2)

    q, k, v = split(query(x)), split(key(x)), split(value(x))
    scores = q @ k.transpose(-1, -2) / head_dim ** 0.5
    ctx = (torch.softmax(scores, dim=-1) @ v).transpose(-3, -2)
    ctx = ctx.reshape(*x.shape[:-1], num_heads * head_dim)
    return output(ctx)


def test_attention_heads_values_are_correct_not_just_shape():
    """Mirrors test_d6_conv_values_are_correct_not_just_shape's reasoning:
    verify the surviving head's actual rows/columns, not just output shape."""
    query, key, value, output = _qkvo(hidden=8)  # 2 heads of 4

    new_q, new_k, new_v, new_o = prune_attention_heads(
        query, key, value, output, keep_heads=torch.tensor([1]), num_heads=2
    )

    assert new_q.out_features == 4 and new_o.in_features == 4
    assert torch.equal(new_q.weight, query.weight[4:8])
    assert torch.equal(new_k.weight, key.weight[4:8])
    assert torch.equal(new_v.weight, value.weight[4:8])
    assert torch.equal(new_q.bias, query.bias[4:8])
    assert torch.equal(new_o.weight, output.weight[:, 4:8])
    # output's bias is over hidden size, not the head layout -- untouched.
    assert torch.equal(new_o.bias, output.bias)


def test_dropping_a_middle_head_of_four_keeps_the_others_in_head_order():
    query, key, value, output = _qkvo(hidden=16)  # 4 heads of 4
    new_q, _, _, new_o = prune_attention_heads(
        query, key, value, output, keep_heads=torch.tensor([0, 2, 3]), num_heads=4
    )
    expected_rows = torch.cat([torch.arange(0, 4), torch.arange(8, 16)])
    assert torch.equal(new_q.weight, query.weight[expected_rows])
    assert torch.equal(new_o.weight, output.weight[:, expected_rows])


def test_attention_forward_pass_holds_after_surgery():
    """End-to-end proof the seam is real: a forward pass through the pruned
    Q/K/V/output quartet, with the surviving heads' true head count, must
    actually run and produce the right shape."""
    query, key, value, output = _qkvo(hidden=8)
    x = torch.randn(3, 5, 8)

    keep_heads = torch.tensor([1])
    new_q, new_k, new_v, new_o = prune_attention_heads(
        query, key, value, output, keep_heads=keep_heads, num_heads=2
    )
    out = _mha_forward(new_q, new_k, new_v, new_o, num_heads=keep_heads.numel(), x=x)
    assert out.shape == (3, 5, 8)

    before = _mha_forward(query, key, value, output, num_heads=2, x=x)
    assert not torch.allclose(before, out, atol=1e-4)  # a head was actually removed


def test_attention_biasless_projections_survive_surgery():
    query, key, value, output = _qkvo(hidden=8, bias=False)
    new_q, new_k, new_v, new_o = prune_attention_heads(
        query, key, value, output, keep_heads=torch.tensor([0]), num_heads=2
    )
    assert new_q.bias is None and new_k.bias is None and new_v.bias is None and new_o.bias is None


def test_attention_qkv_seam_mismatch_is_rejected():
    query, key, value, output = _qkvo(hidden=8)
    mismatched_key = nn.Linear(8, 999)
    with pytest.raises(ValueError, match="seam mismatch"):
        prune_attention_heads(query, mismatched_key, value, output, keep_heads=torch.tensor([0]), num_heads=2)


def test_attention_output_seam_mismatch_is_rejected():
    query, key, value, _ = _qkvo(hidden=8)
    mismatched_output = nn.Linear(999, 8)
    with pytest.raises(ValueError, match="seam mismatch"):
        prune_attention_heads(query, key, value, mismatched_output, keep_heads=torch.tensor([0]), num_heads=2)


def test_attention_indivisible_num_heads_is_rejected():
    query, key, value, output = _qkvo(hidden=8)
    with pytest.raises(ValueError, match="not divisible"):
        prune_attention_heads(query, key, value, output, keep_heads=torch.tensor([0]), num_heads=3)


def test_attention_out_of_range_and_empty_keep_heads_are_rejected():
    query, key, value, output = _qkvo(hidden=16)  # 4 heads of 4
    with pytest.raises(ValueError, match="out-of-range"):
        prune_attention_heads(query, key, value, output, keep_heads=torch.tensor([0, 99]), num_heads=4)
    with pytest.raises(ValueError, match="collapse"):
        prune_attention_heads(query, key, value, output, keep_heads=torch.tensor([], dtype=torch.long), num_heads=4)


def test_prune_attention_heads_against_a_real_hf_bert_model():
    """Verified against a real HF BERT forward pass, same bar `prune_ffn_block`
    is held to (see README's status table)."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64)
    model = transformers.BertModel(config)
    layer = model.encoder.layer[0].attention

    new_q, new_k, new_v, new_o = prune_attention_heads(
        layer.self.query, layer.self.key, layer.self.value, layer.output.dense,
        keep_heads=torch.tensor([0, 1, 3]), num_heads=config.num_attention_heads,
    )
    layer.self.query, layer.self.key, layer.self.value = new_q, new_k, new_v
    layer.output.dense = new_o
    # BertSelfAttention reshapes Q/K/V using its own num_attention_heads /
    # attention_head_size / all_head_size, not config.num_attention_heads --
    # both must be updated for the forward pass below to reshape correctly.
    layer.self.num_attention_heads = 3
    layer.self.attention_head_size = new_q.out_features // 3
    layer.self.all_head_size = new_q.out_features

    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    out = model(input_ids)  # must not raise -- proves the seam is real inside the real module
    assert out.last_hidden_state.shape == (2, 6, config.hidden_size)
