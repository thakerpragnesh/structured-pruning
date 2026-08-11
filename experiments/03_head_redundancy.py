"""
03_head_redundancy.py — extends the paper's Manhattan/Euclidean/Cosine K-Means
comparison from CNN channels to Transformer attention heads: flatten each
head's Q/K/V weight slice into a vector, compute pairwise distance under all
three metrics, and report which pairs are closest (candidates for redundancy).

    python experiments/03_head_redundancy.py             # real BERT-base, all layers
    python experiments/03_head_redundancy.py --smoke     # random-init tiny BERT config
"""
import argparse

import torch

from prunelib import pairwise_distance_matrix


def _head_vectors(query_weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """query_weight: [hidden, hidden] -> [num_heads, head_dim * hidden] flattened per head."""
    hidden = query_weight.shape[0]
    head_dim = hidden // num_heads
    return query_weight.reshape(num_heads, head_dim, hidden).reshape(num_heads, -1)


def run_smoke(seed=0):
    torch.manual_seed(seed)
    try:
        from transformers import BertConfig, BertModel
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install transformers to run this experiment") from exc

    config = BertConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64)
    model = BertModel(config)

    for layer_idx, layer in enumerate(model.encoder.layer):
        q_weight = layer.attention.self.query.weight
        vectors = _head_vectors(q_weight, config.num_attention_heads)

        print(f"\nlayer {layer_idx}: {config.num_attention_heads} heads")
        for metric in ("manhattan", "euclidean", "cosine"):
            dist = pairwise_distance_matrix(vectors, metric=metric)
            off_diag = dist + torch.eye(dist.shape[0]) * dist.max()
            min_val = off_diag.min().item()
            i, j = (off_diag == off_diag.min()).nonzero()[0].tolist()
            print(f"  {metric:<10} closest pair: heads ({i}, {j})  distance={min_val:.4f}")

    print("\nsmoke run complete. Random-init weights carry no real redundancy signal --")
    print("this only verifies the scan runs correctly against a real HF model's shapes.")
    print("Run without --smoke against fine-tuned bert-base-uncased for a real result.")


def run_full():
    raise NotImplementedError(
        "Full run needs a fine-tuned bert-base-uncased checkpoint (redundancy "
        "only appears after training). Load it with BertModel.from_pretrained "
        "and reuse _head_vectors / pairwise_distance_matrix exactly as run_smoke does."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_smoke() if args.smoke else run_full()
