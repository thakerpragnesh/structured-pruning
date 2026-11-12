"""Physically removes attention heads from a transformer layer's Q/K/V and
output projections.

`head_analysis.AttentionHeadAnalyzer` finds which heads look redundant;
this is the other half -- turning a list of head indices into a smaller
model. Q, K, and V each partition their *output* rows into `num_heads`
contiguous blocks of `head_dim` rows (HuggingFace's layout: head `h` owns
rows `[h*head_dim, (h+1)*head_dim)`); the attention output projection
partitions its *input* columns the same way, since its input is the heads'
concatenated context vectors. Removing head `h` therefore means dropping
those rows from Q/K/V (weight and bias) and those columns from the output
projection -- never partial rows or columns, or the surviving heads' math
would be corrupted.

**Bias compensation** works exactly as it does for `ffn_surgery.FFNSurgeon`.
Removing head `h` deletes `W_O[:, h] @ ctx_h` from the block output. That
term depends on the input, but so does an FFN neuron's `W_out[:, i] * a_i`,
and in both cases compensation restores only the *expectation*:
`W_O[:, h] @ E[ctx_h]` is a constant, and a constant is exactly what the
output bias can carry. That turns zero-ablation of the head into mean-ablation,
the standard and much gentler way interpretability work removes heads. An
earlier revision of this module said heads could not be compensated "because
a head's contribution is a function of the input". That reasoning would rule
out FFN compensation too. It was wrong, and the option now exists
(`compensate_bias=True`, with `head_calibration.HeadStats`).

There is still **no merge** option. A dropped FFN twin can be folded into a
survivor because both neurons are scalars along fixed output columns. A
head is a rank-`head_dim` map behind its own attention pattern: two heads
that attend alike can still read and write different subspaces, and no
surviving head's `head_dim`-wide value space can absorb another's in general.

Updating a model's head-count bookkeeping (`num_attention_heads` /
`attention_head_size` / `all_head_size`) after this resize is the caller's
job, via the adapter -- see `adapters.BertLayerAdapter.set_attention_heads`
and `workflow.prune_attention_heads`.
"""
import torch
import torch.nn as nn

from .linear_ops import rebuild_linear


class AttentionSurgeon:
    def resize(self, query: nn.Linear, key: nn.Linear, value: nn.Linear, output: nn.Linear,
               num_heads: int, head_indices, stats=None, compensate_bias: bool = False):
        """Return `(query, key, value, output, num_heads)` with the given
        heads removed.

        `query`/`key`/`value` are `(hidden, hidden)` Linear modules whose
        output rows partition into `num_heads` head-sized blocks. `output`
        is the attention block's output projection, whose *input* columns
        partition the same way. `head_indices` are the heads to drop.

        With `compensate_bias=True`, each dropped head's mean contribution
        `W_O[:, h] @ stats.context_mean[h]` is added to the output bias,
        which preserves the block's mean output exactly on the calibration
        distribution. `stats` is this layer's `head_calibration.HeadStats`.
        """
        out_features = query.weight.shape[0]
        if out_features % num_heads:
            raise ValueError(
                f"query output size {out_features} is not divisible by num_heads {num_heads}; "
                "num_heads does not match this layer's actual Q/K/V layout."
            )
        if output.weight.shape[1] != out_features:
            raise ValueError(
                f"Attention block is inconsistent: query/key/value produce {out_features} "
                f"features but the output projection expects {output.weight.shape[1]} inputs. "
                "Check the adapter's get_attention_heads() is returning a matching quartet."
            )
        head_dim = out_features // num_heads

        drop = {int(h) for h in head_indices}
        out_of_range = sorted(h for h in drop if h < 0 or h >= num_heads)
        if out_of_range:
            raise ValueError(
                f"head_indices contains out-of-range head indices {out_of_range[:5]} "
                f"for a layer with {num_heads} heads."
            )
        keep_heads = [h for h in range(num_heads) if h not in drop]
        if not keep_heads:
            raise ValueError(
                "Selection would remove every head, which collapses the attention block. "
                "Keep at least one head."
            )

        device = query.weight.device
        row_keep = torch.cat([
            torch.arange(h * head_dim, (h + 1) * head_dim, device=device) for h in keep_heads
        ])

        bias_delta = None
        if compensate_bias:
            bias_delta = self._bias_compensation(output, stats, num_heads, head_dim, sorted(drop))

        new_query = self._slice_out_features(query, row_keep)
        new_key = self._slice_out_features(key, row_keep)
        new_value = self._slice_out_features(value, row_keep)
        new_output = self._slice_in_features(output, row_keep, bias_delta=bias_delta)
        return new_query, new_key, new_value, new_output, len(keep_heads)

    @staticmethod
    def _bias_compensation(output: nn.Linear, stats, num_heads: int, head_dim: int, dropped):
        """`sum_h W_O[:, h] @ E[ctx_h]` over the dropped heads: the mean
        output contribution the cut removes."""
        if output.bias is None:
            raise ValueError(
                "compensate_bias=True but the attention output projection has no bias to fold "
                "the removed heads' mean contribution into. Pass compensate_bias=False."
            )
        if stats is None:
            raise ValueError(
                "compensate_bias=True requires head calibration stats. Collect them with "
                "`head_calibration.HeadCalibrator(model).collect(batches, layer_idx)`."
            )
        if stats.num_heads != num_heads or stats.head_dim != head_dim:
            raise ValueError(
                f"Head stats cover {stats.num_heads} heads of size {stats.head_dim} but the layer "
                f"has {num_heads} of size {head_dim}; they were collected against a different "
                "(or already pruned) layer."
            )
        if not dropped:
            return None
        w = output.weight.data.to(torch.float32)
        cols = torch.cat([
            torch.arange(h * head_dim, (h + 1) * head_dim, device=w.device) for h in dropped
        ])
        means = stats.context_mean.to(w.device, torch.float32)[dropped].reshape(-1)
        return w[:, cols] @ means

    @staticmethod
    def _slice_out_features(linear: nn.Linear, keep: torch.Tensor) -> nn.Linear:
        """Keep only rows `keep` of weight and bias -- shrinks the layer's
        output dimension (Q/K/V's per-head rows).
        """
        bias = None if linear.bias is None else linear.bias.data[keep]
        return rebuild_linear(linear, linear.weight.data[keep], bias)

    @staticmethod
    def _slice_in_features(linear: nn.Linear, keep: torch.Tensor, bias_delta=None) -> nn.Linear:
        """Keep only columns `keep` of weight -- shrinks the layer's input
        dimension (the output projection's per-head columns). Its bias is
        over the model's hidden size, not the head layout, so it survives
        whole, plus `bias_delta` when compensating.
        """
        bias = None
        if linear.bias is not None:
            bias = linear.bias.data.clone()
            if bias_delta is not None:
                bias += bias_delta.to(bias.dtype)
        return rebuild_linear(linear, linear.weight.data[:, keep], bias)
