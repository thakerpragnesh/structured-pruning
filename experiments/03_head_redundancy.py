"""
03_head_redundancy.py — extends the paper's Manhattan/Euclidean/Cosine K-Means
comparison from CNN channels to Transformer attention heads: flatten each
head's Q/K/V weight slice into a vector, compute pairwise distance under all
three metrics, and report which pairs are closest (candidates for redundancy).
Then act on it: cluster each layer's heads with K-Means (Manhattan) on their
concatenated Q/K/V slices, prune the lowest-L1 head(s) within each cluster via
`prunelib.prune_attention_heads`, and verify the pruned model still runs.

    python experiments/03_head_redundancy.py             # real BERT-base, all layers
    python experiments/03_head_redundancy.py --smoke     # random-init tiny BERT config
"""
import argparse

import torch

from prunelib import complement_indices, kmeans_prune_indices, pairwise_distance_matrix, prune_attention_heads


def _head_vectors(query_weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """query_weight: [hidden, hidden] -> [num_heads, head_dim * hidden] flattened per head."""
    hidden = query_weight.shape[0]
    head_dim = hidden // num_heads
    return query_weight.reshape(num_heads, head_dim, hidden).reshape(num_heads, -1)


def _qkv_head_vectors(self_attn, num_heads: int) -> torch.Tensor:
    """[num_heads, 3 * head_dim * hidden]: each head's Q, K and V rows
    flattened together, so clustering (and the within-cluster L1 tiebreak)
    sees everything the head owns, not just its query projection."""
    return torch.cat(
        [_head_vectors(proj.weight.detach(), num_heads) for proj in (self_attn.query, self_attn.key, self_attn.value)],
        dim=1,
    )


def prune_redundant_heads(attention, n_prune: int, metric: str = "manhattan", seed: int = 0) -> torch.Tensor:
    """Cluster one HF `BertAttention` block's heads and prune the `n_prune`
    lowest-L1 heads within their clusters (`kmeans_prune_indices`' default
    `n_clusters = num_heads - n_prune`: one surviving head per cluster).
    Rewires the block in place and returns the pruned head indices."""
    self_attn = attention.self
    num_heads = self_attn.num_attention_heads
    pruned = kmeans_prune_indices(_qkv_head_vectors(self_attn, num_heads), n_prune, metric=metric, seed=seed)
    keep = complement_indices(num_heads, pruned)

    q, k, v, o = prune_attention_heads(
        self_attn.query, self_attn.key, self_attn.value, attention.output.dense, keep_heads=keep, num_heads=num_heads,
    )
    self_attn.query, self_attn.key, self_attn.value, attention.output.dense = q, k, v, o
    # BertSelfAttention reshapes with its own attributes, not the config's --
    # see test_surgery.py::test_prune_attention_heads_against_a_real_hf_bert_model.
    self_attn.num_attention_heads = keep.numel()
    self_attn.attention_head_size = q.out_features // keep.numel()
    self_attn.all_head_size = q.out_features
    return pruned


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

    print("\npruning 1 head per layer (K-Means/Manhattan, lowest L1 within cluster):")
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    for layer_idx, layer in enumerate(model.encoder.layer):
        pruned = prune_redundant_heads(layer.attention, n_prune=1)
        print(f"  layer {layer_idx}: pruned heads {pruned.tolist()} -> {layer.attention.self.num_attention_heads} heads left")
    out = model(input_ids).last_hidden_state
    assert out.shape == (2, 8, config.hidden_size)
    print(f"  pruned model forward pass OK, output {tuple(out.shape)}")

    print("\nsmoke run complete. Random-init weights carry no real redundancy signal --")
    print("this only verifies the scan runs correctly against a real HF model's shapes.")
    print("Run without --smoke against fine-tuned bert-base-uncased for a real result.")


def run_full():
    raise NotImplementedError(
        "Full run needs a fine-tuned bert-base-uncased checkpoint (redundancy "
        "only appears after training). Load it with BertModel.from_pretrained "
        "and reuse _head_vectors / pairwise_distance_matrix / prune_redundant_heads "
        "exactly as run_smoke does."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_smoke() if args.smoke else run_full()
