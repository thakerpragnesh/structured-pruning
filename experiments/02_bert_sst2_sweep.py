"""
02_bert_sst2_sweep.py — extends the CNN Max-k criterion to Transformer FFN
blocks: score each FFN's intermediate neurons, prune the weakest, physically
shrink both Linear layers, and (in full mode) fine-tune and report SST-2 F1.

    python experiments/02_bert_sst2_sweep.py                          # downloads bert-base + SST-2, fine-tunes
    python experiments/02_bert_sst2_sweep.py --smoke                  # random-init tiny BERT config, no downloads
"""
import argparse

import torch

from prunelib import compute_score, count_params, prune_ffn_block, select_prune_indices


def _score_ffn_neurons(fc1_weight: torch.Tensor, method: str = "max_k") -> torch.Tensor:
    """Max-k/L1/L2 saliency doesn't require a 4D conv tensor to be meaningful --
    an FFN's `fc1` row is exactly analogous to a conv output channel's flattened
    kernel. Reuse the same scoring code by reshaping to a fake [out, in, 1, 1]
    conv weight rather than duplicating the math."""
    reshaped = fc1_weight.unsqueeze(-1).unsqueeze(-1)  # [hidden, in_features, 1, 1]
    kwargs = {"k": 3} if method == "max_k" else {}
    return compute_score(reshaped, method=method, **kwargs)


def run_smoke(prune_fraction=0.3, seed=0):
    torch.manual_seed(seed)
    try:
        from transformers import BertConfig, BertForSequenceClassification
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install transformers to run this experiment") from exc

    config = BertConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, vocab_size=1000, num_labels=2,
    )
    model = BertForSequenceClassification(config)

    layer = model.bert.encoder.layer[0]
    fc1, fc2 = layer.intermediate.dense, layer.output.dense
    params_before = count_params(model)

    scores = _score_ffn_neurons(fc1.weight, method="max_k")
    prune_amount = int(config.intermediate_size * prune_fraction)
    keep_idx = torch.tensor(
        [i for i in range(config.intermediate_size)
         if i not in set(select_prune_indices(scores, prune_amount).tolist())]
    )

    new_fc1, new_fc2 = prune_ffn_block(fc1, fc2, keep_idx)
    layer.intermediate.dense, layer.output.dense = new_fc1, new_fc2

    params_after = count_params(model)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    out = model(input_ids)  # must not raise -- proves the seam holds inside the real module

    print(f"FFN intermediate size: {config.intermediate_size} -> {len(keep_idx)}")
    print(f"model params: {params_before:,} -> {params_after:,} ({(1 - params_after/params_before):.1%} reduction)")
    print(f"forward pass output shape: {tuple(out.logits.shape)} (no shape errors -> seam is real)")
    print("\nsmoke run complete. This does NOT reproduce SST-2 F1 -- no real weights,")
    print("no fine-tuning, no dataset. Run without --smoke for that.")


def run_full():
    raise NotImplementedError(
        "Full run needs a pretrained bert-base checkpoint, the SST-2 dataset "
        "via `datasets`, and a fine-tune loop after pruning each layer. Wire "
        "in AutoModelForSequenceClassification.from_pretrained('bert-base-uncased') "
        "and reuse _score_ffn_neurons / prune_ffn_block exactly as run_smoke does."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_smoke() if args.smoke else run_full()
