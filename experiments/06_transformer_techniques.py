"""
06_transformer_techniques.py — the calibration-aware transformer pipeline
from `prunelib.transformer`, end to end on a real HuggingFace BERT class.

    1. heads:  calibrate (activation moments + Michel et al. gradient
               importance), rank every head in the model globally, cut,
               and fold each removed head's mean output into the bias
    2. FFN:    re-calibrate on the *head-pruned* model (the head cut shifts
               every downstream FFN's input), rank neurons globally by
               ||W_out[:, i]|| * E|a_i|, cut with bias compensation
    3. twins:  record firing patterns, find co-activation twins, and merge
               (not just drop) them in one layer

    python experiments/06_transformer_techniques.py --smoke

`--smoke` uses a random-init tiny BertForSequenceClassification and random
token batches -- no downloads -- so it verifies the mechanics (shapes, seams,
hooks, bias compensation) and nothing about accuracy. For accuracy, run the
same calls against a fine-tuned checkpoint and a real calibration set.
"""
import argparse

import torch

from prunelib import count_params
from prunelib.transformer import (
    ActivationAwareSelector,
    ActivationRecorder,
    FFNCalibrator,
    GradientHeadSelector,
    HeadCalibrator,
    JaccardTwinFinder,
    TwinRedundancySelector,
    prune_ffn_layer,
    prune_model_attention,
    prune_model_ffn,
)


def _batches(vocab_size, n_batches=4, batch=8, seq=16, num_labels=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batches):
        mask = torch.ones(batch, seq, dtype=torch.long)
        mask[: batch // 2, seq * 3 // 4:] = 0  # real padding, so masking is exercised
        out.append({
            "input_ids": torch.randint(0, vocab_size, (batch, seq), generator=g),
            "attention_mask": mask,
            "labels": torch.randint(0, num_labels, (batch,), generator=g),
        })
    return out


def _heads(model):
    return [layer.attention.self.num_attention_heads for layer in model.bert.encoder.layer]


def _ffn_widths(model):
    return [layer.intermediate.dense.out_features for layer in model.bert.encoder.layer]


def run_smoke(head_percent=25, ffn_percent=40, seed=0):
    torch.manual_seed(seed)
    try:
        from transformers import BertConfig, BertForSequenceClassification
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install transformers to run this experiment") from exc

    config = BertConfig(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=8,
        intermediate_size=128, vocab_size=500, num_labels=2,
        attn_implementation="eager",
    )
    model = BertForSequenceClassification(config).eval()
    batches = _batches(config.vocab_size)
    params_before = count_params(model)

    # 1. Attention heads -- global ranking by gradient importance, mean-ablation.
    calibrator = HeadCalibrator(model)
    head_stats = calibrator.collect_many(batches)
    importance = calibrator.collect_gradient_importance(batches)
    print(f"heads per layer before: {_heads(model)}")
    prune_model_attention(
        model, GradientHeadSelector(prune_percent=head_percent), allocation="global",
        normalize="l2", stats_by_layer=head_stats,
        gradient_importance_by_layer=importance, compensate_bias=True,
    )
    print(f"heads per layer after:  {_heads(model)}  (global: layers give up different amounts)")

    # 2. FFN -- calibrate on the head-pruned model, then global activation-aware cut.
    ffn_stats = FFNCalibrator(model).collect_many(batches)
    print(f"\nFFN width before: {_ffn_widths(model)}")
    prune_model_ffn(
        model, ActivationAwareSelector(prune_percent=ffn_percent), allocation="global",
        stats_by_layer=ffn_stats, compensate_bias=True,
    )
    print(f"FFN width after:  {_ffn_widths(model)}")

    # 3. Twins in layer 0 -- behavioural redundancy, merged rather than dropped.
    # Random init has no real twins, so plant one (as experiment 04 does):
    # neuron 1 becomes an exact copy of neuron 0, so it fires identically.
    dense = model.bert.encoder.layer[0].intermediate.dense
    with torch.no_grad():
        dense.weight[1] = dense.weight[0]
        dense.bias[1] = dense.bias[0]
    fires =ActivationRecorder(model, tokenizer=None).record_batches(batches, layer_idx=0)
    twins, dead = JaccardTwinFinder().find(fires, threshold=0.9)
    print(f"\nlayer 0: {len(twins)} twin pairs at Jaccard >= 0.9, {len(dead)} dead neurons")
    if twins:
        stats0 = FFNCalibrator(model).collect(batches, layer_idx=0)
        kept = prune_ffn_layer(
            model, 0, TwinRedundancySelector([(i, j) for i, j, _ in twins], merge=True),
            stats=stats0, compensate_bias=True,
        )
        print(f"layer 0 FFN after merging twins: {kept} neurons")

    params_after = count_params(model)
    with torch.no_grad():
        logits = model(**{k: v for k, v in batches[0].items() if k != "labels"}).logits
    assert logits.shape == (8, config.num_labels)

    print(f"\nmodel params: {params_before:,} -> {params_after:,} "
          f"({1 - params_after / params_before:.1%} reduction)")
    print(f"forward pass output shape: {tuple(logits.shape)} (seams hold inside the real HF modules)")
    print("\nsmoke run complete. Random weights and random tokens: this checks the")
    print("mechanics only. Re-run the same calls on a fine-tuned checkpoint for accuracy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="random-init tiny BERT, no downloads")
    parser.add_argument("--head-percent", type=float, default=25)
    parser.add_argument("--ffn-percent", type=float, default=40)
    args = parser.parse_args()
    if not args.smoke:
        raise SystemExit(
            "Only --smoke is wired here. For a real run, load a fine-tuned checkpoint with "
            "AutoModelForSequenceClassification.from_pretrained(...) and a tokenized "
            "calibration set, then make the same calls run_smoke() makes."
        )
    run_smoke(args.head_percent, args.ffn_percent)
