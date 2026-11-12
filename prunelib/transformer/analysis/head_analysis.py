"""Attention-head redundancy analysis.

Two families of signal, both computed from weights alone:

- **Query-weight similarity / distance** -- cosine similarity or
  Lp-distance between heads' flattened query weight vectors, optionally on
  unit-normalized vectors to isolate directional (as opposed to magnitude)
  redundancy. One `normalize` switch replaces what used to be two
  nearly-identical classes (`SimilarityScanner`, `DistributionScanner`).
  This is the thesis-era signal, and a weak one: `W_Q` only decides *where*
  a head looks, so two heads with similar queries but different values do
  different jobs.
- **OV-circuit norms / similarity** -- each head's `W_O[:, h] @ W_V[h]`, the
  `(hidden, hidden)` map from what the head attends to onto what it writes
  into the residual stream. Its Frobenius norm bounds how much the head can
  change the block output at all; the cosine between two heads' OV circuits
  says whether they write the same thing. Both are what
  `head_selectors.OVNormHeadSelector` / `RedundantHeadSelector` rank by.

The OV quantities are computed through `(head_dim, head_dim)` Gram blocks
rather than by materialising every head's `(hidden, hidden)` product: for
`A_h = W_O[:, h]` and `B_h = W_V[h]`,
`<A_h B_h, A_g B_g>_F = sum((A_h^T A_g) * (B_h B_g^T))` -- `O(heads^2 *
head_dim^2)` memory instead of `O(heads * hidden^2)`.
"""
import torch

from ...scanners import pairwise_distance_matrix
from ..models.adapters import resolve_adapter


def flat_query_heads(query_weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """`(num_heads, head_dim * in_features)`: each head's query rows, flattened.

    The head split is read off `query_weight.shape[0]` rather than a
    model-wide `hidden_size`, which keeps this correct after
    `attention_surgery` has shrunk this layer's heads independently of the
    others.
    """
    w_q = query_weight.detach()
    out_features, in_features = w_q.shape
    if out_features % num_heads:
        raise ValueError(
            f"query output size {out_features} is not divisible by num_attention_heads "
            f"{num_heads}; the adapter is reporting a head layout this layer does not have."
        )
    head_dim = out_features // num_heads
    # The head partition runs down the output rows, so reshaping dim 0
    # into (heads, head_dim) is what groups each head's rows together.
    return w_q.reshape(num_heads, head_dim * in_features).float()


def ov_gram(value_weight: torch.Tensor, output_weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """`(num_heads, num_heads)` Frobenius inner products between heads' OV circuits.

    `value_weight` is `(all_head_size, hidden_in)` -- head `h` owns rows
    `[h*d, (h+1)*d)`; `output_weight` is `(hidden_out, all_head_size)` --
    head `h` owns the same range of *columns*. Entry `[h, g]` is
    `<W_O[:, h] W_V[h], W_O[:, g] W_V[g]>_F`; the diagonal is each head's
    squared OV norm.
    """
    rows = value_weight.shape[0]
    if rows % num_heads or output_weight.shape[1] != rows:
        raise ValueError(
            f"value rows {rows} / output columns {output_weight.shape[1]} do not split into "
            f"{num_heads} equal heads."
        )
    head_dim = rows // num_heads
    v = value_weight.detach().float().reshape(num_heads, head_dim, -1)               # B_h: (d, in)
    o = output_weight.detach().float().reshape(output_weight.shape[0], num_heads, head_dim)
    o = o.permute(1, 0, 2)                                                           # A_h: (out, d)
    out_gram = torch.einsum("hod,goe->hgde", o, o)     # A_h^T A_g
    value_gram = torch.einsum("hdi,gei->hgde", v, v)   # B_h B_g^T
    return (out_gram * value_gram).sum(dim=(-1, -2))


def ov_norms(value_weight: torch.Tensor, output_weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """`(num_heads,)` Frobenius norm of each head's OV circuit `W_O[:, h] W_V[h]`."""
    return ov_gram(value_weight, output_weight, num_heads).diagonal().clamp_min(0).sqrt()


def ov_similarity(value_weight: torch.Tensor, output_weight: torch.Tensor, num_heads: int,
                  eps: float = 1e-12) -> torch.Tensor:
    """`(num_heads, num_heads)` cosine similarity between heads' OV circuits.

    Signed: two heads whose OV circuits point in opposite directions
    (cosine near -1) cancel rather than duplicate each other, so they are
    not redundant and must not look it.
    """
    gram = ov_gram(value_weight, output_weight, num_heads)
    norms = gram.diagonal().clamp_min(0).sqrt()
    return gram / (norms.unsqueeze(0) * norms.unsqueeze(1)).clamp_min(eps)


class AttentionHeadAnalyzer:
    def __init__(self, model=None, adapter=None):
        if adapter is None and model is None:
            raise ValueError("AttentionHeadAnalyzer requires either `model` or `adapter`.")
        self.adapter = resolve_adapter(model, adapter)

    def get_flat_heads(self, layer_idx: int) -> torch.Tensor:
        return flat_query_heads(
            self.adapter.get_query_weight(layer_idx), self.adapter.num_attention_heads(layer_idx)
        )

    def compute_similarity(self, layer_idx: int):
        heads = torch.nn.functional.normalize(self.get_flat_heads(layer_idx), p=2, dim=1)
        return torch.mm(heads, heads.t()).cpu().numpy()

    def compute_distance_matrix(self, layer_idx: int, metric: str = "manhattan", normalize: bool = False):
        """Pairwise head distance under any `prunelib.pairwise_distance_matrix`
        metric ("manhattan", "euclidean", "cosine", or one added with
        `prunelib.register_metric`) -- the same registry the CNN channel
        comparison uses, so the two can't drift apart."""
        heads = self.get_flat_heads(layer_idx)
        if normalize:
            heads = torch.nn.functional.normalize(heads, p=2, dim=1)
        dist = pairwise_distance_matrix(heads, metric=metric)
        dist.fill_diagonal_(0.0)
        # .cpu() before .numpy(): on a CUDA model this raised
        # "can't convert cuda tensor to numpy" -- compute_similarity
        # already did this, compute_distance_matrix did not.
        return dist.cpu().numpy()

    def _value_output(self, layer_idx: int):
        _, _, value, output = self.adapter.get_attention_heads(layer_idx)
        return value.weight, output.weight, self.adapter.num_attention_heads(layer_idx)

    def compute_ov_similarity(self, layer_idx: int):
        """Cosine similarity between heads' OV circuits -- see `ov_similarity`."""
        return ov_similarity(*self._value_output(layer_idx)).cpu().numpy()

    def compute_ov_norms(self, layer_idx: int):
        """Frobenius norm of each head's OV circuit -- see `ov_norms`."""
        return ov_norms(*self._value_output(layer_idx)).cpu().numpy()
