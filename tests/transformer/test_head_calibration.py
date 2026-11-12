import pytest
import torch

from conftest import StubBert
from prunelib.transformer import BertLayerAdapter, HeadCalibrator, HeadStats


def wide_model():
    """4 heads of size 4, so per-head arithmetic is not trivially 2x2."""
    return StubBert(hidden=16, intermediate=16, num_layers=2, heads=4)


def record_contexts(model, batches, layer_idx):
    """The output projection's input on every real token: `(tokens, all_head)`."""
    output = BertLayerAdapter(model).get_attention_heads(layer_idx)[3]
    rows = []

    def hook(_module, args):
        rows.append(args[0].detach())

    handle = output.register_forward_pre_hook(hook)
    with torch.no_grad():
        for batch in batches:
            model(**batch)
    handle.remove()
    flat = []
    for batch, ctx in zip(batches, rows):
        mask = batch["attention_mask"].reshape(-1).bool()
        flat.append(ctx.reshape(-1, ctx.shape[-1])[mask])
    return torch.cat(flat), output.weight.detach()


def test_head_stats_match_an_explicit_per_token_computation(batches):
    model = wide_model()
    stats = HeadCalibrator(model).collect(batches, layer_idx=0)

    ctx, w_o = record_contexts(model, batches, 0)
    heads, d = 4, 4
    assert stats.tokens == ctx.shape[0] == 9  # 3 + 2 + 4 real tokens; padding excluded
    per_head = ctx.reshape(-1, heads, d)
    for h in range(heads):
        contribution = per_head[:, h] @ w_o[:, h * d:(h + 1) * d].t()   # (tokens, hidden)
        rms = contribution.square().sum(dim=1).mean().sqrt()
        centered = contribution - contribution.mean(dim=0)
        std = centered.square().sum(dim=1).mean().sqrt()
        assert stats.contribution_rms[h] == pytest.approx(rms.item(), rel=1e-4)
        assert stats.contribution_std[h] == pytest.approx(std.item(), rel=1e-4)
        assert torch.allclose(stats.context_mean[h], per_head[:, h].mean(dim=0), atol=1e-6)


def test_std_never_exceeds_rms(batches):
    stats = HeadCalibrator(wide_model()).collect(batches, layer_idx=1)
    assert (stats.contribution_std <= stats.contribution_rms + 1e-6).all()


def test_a_head_with_zero_output_columns_contributes_nothing(batches):
    model = wide_model()
    output = BertLayerAdapter(model).get_attention_heads(0)[3]
    with torch.no_grad():
        output.weight[:, 4:8] = 0.0  # head 1
    stats = HeadCalibrator(model).collect(batches, layer_idx=0)
    assert stats.contribution_rms[1] == 0.0
    assert stats.contribution_rms[0] > 0.0


def test_collect_many_matches_collect_per_layer(batches):
    model = wide_model()
    calibrator = HeadCalibrator(model)
    many = calibrator.collect_many(batches)
    assert sorted(many) == [0, 1]
    for idx in (0, 1):
        single = calibrator.collect(batches, layer_idx=idx)
        assert torch.allclose(many[idx].contribution_rms, single.contribution_rms)
        assert torch.allclose(many[idx].context_mean, single.context_mean)


def test_select_and_to_keep_the_chosen_heads(batches):
    stats = HeadCalibrator(wide_model()).collect(batches, layer_idx=0)
    narrowed = stats.select([0, 3])
    assert narrowed.num_heads == 2 and narrowed.head_dim == 4
    assert torch.equal(narrowed.contribution_rms, stats.contribution_rms[[0, 3]])
    assert stats.to(torch.float64).contribution_rms.dtype == torch.float64


def test_calibration_leaves_no_hooks_and_restores_train_mode(batches):
    model = wide_model()
    model.train()
    calibrator = HeadCalibrator(model)
    calibrator.collect(batches, layer_idx=0)
    calibrator.collect_gradient_importance(batches, loss_fn=lambda out, _: out.square().mean())
    assert model.training
    for module in model.modules():
        assert not module._forward_pre_hooks and not module._forward_hooks


def test_empty_batches_are_rejected():
    with pytest.raises(ValueError, match="no tokens"):
        HeadCalibrator(wide_model()).collect([], layer_idx=0)
    with pytest.raises(ValueError, match="no examples"):
        HeadCalibrator(wide_model()).collect_gradient_importance(
            [], loss_fn=lambda out, _: out.sum()
        )


# --- gradient importance (Michel et al., 2019) -----------------------------

def single_examples():
    return [
        {"input_ids": torch.tensor([[3, 4, 5, 6]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)},
        {"input_ids": torch.tensor([[7, 8, 9, 10]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)},
        {"input_ids": torch.tensor([[11, 2, 12, 13]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)},
    ]


def loss_fn(outputs, _batch):
    return outputs.square().mean()


def test_gradient_importance_matches_finite_differences():
    """Scaling head h's columns of W_O by (1 + eps) is exactly scaling its
    gate xi_h, so a central difference over that is an independent check
    of the hook-based gradient."""
    model = wide_model().double()
    examples = single_examples()
    importance = HeadCalibrator(model).collect_gradient_importance(examples, loss_fn=loss_fn)

    adapter, eps, d = BertLayerAdapter(model), 1e-5, 4
    for layer in (0, 1):
        output = adapter.get_attention_heads(layer)[3]
        for h in range(4):
            cols = slice(h * d, (h + 1) * d)
            total = 0.0
            for batch in examples:
                original = output.weight[:, cols].clone()
                losses = []
                for sign in (1, -1):
                    with torch.no_grad():
                        output.weight[:, cols] = original * (1 + sign * eps)
                        losses.append(loss_fn(model(**batch), batch).item())
                with torch.no_grad():
                    output.weight[:, cols] = original
                total += abs(losses[0] - losses[1]) / (2 * eps)
            assert importance[layer][h].item() == pytest.approx(total / len(examples), rel=1e-4)


def test_gradient_importance_is_per_example_regardless_of_batching():
    """A mean-reduced loss scales each example's gradient by 1/batch_size;
    the calibrator undoes that, so one batch of two equals two of one."""
    model = wide_model().double()
    a, b = single_examples()[:2]
    joint = [{
        "input_ids": torch.cat([a["input_ids"], b["input_ids"]]),
        "attention_mask": torch.cat([a["attention_mask"], b["attention_mask"]]),
    }]
    calibrator = HeadCalibrator(model)
    separate = calibrator.collect_gradient_importance([a, b], loss_fn=loss_fn)
    together = calibrator.collect_gradient_importance(joint, loss_fn=loss_fn)
    for layer in (0, 1):
        assert torch.allclose(separate[layer], together[layer], rtol=1e-6)


def test_gradient_importance_does_not_touch_parameter_grads():
    model = wide_model()
    HeadCalibrator(model).collect_gradient_importance(single_examples(), loss_fn=loss_fn)
    assert all(p.grad is None for p in model.parameters())


def test_a_head_the_loss_cannot_see_has_zero_gradient_importance():
    model = wide_model()
    output = BertLayerAdapter(model).get_attention_heads(1)[3]
    with torch.no_grad():
        output.weight[:, 8:12] = 0.0  # head 2 of the last layer writes nothing
    importance = HeadCalibrator(model).collect_gradient_importance(single_examples(), loss_fn=loss_fn)
    assert importance[1][2] == 0.0
    assert (importance[1][[0, 1, 3]] > 0).all()


def test_gradient_importance_without_a_loss_explains_itself():
    with pytest.raises(ValueError, match="labels"):
        HeadCalibrator(wide_model()).collect_gradient_importance(single_examples())


def test_head_stats_dataclass_reports_its_shape():
    stats = HeadStats(
        context_mean=torch.zeros(3, 5), contribution_rms=torch.ones(3),
        contribution_std=torch.ones(3), tokens=7,
    )
    assert stats.num_heads == 3 and stats.head_dim == 5
