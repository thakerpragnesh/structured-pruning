"""
Index-set helpers shared by selection, surgery, and dependency tracing.

Each of these used to be re-derived inline wherever it was needed:
"indices not in this set" in `graph._keep_complement`, `graph.
get_pruning_group`, `vgg.prune_vgg_layer` and `saliency.keep_indices`; "each
channel owns a contiguous block of `block_size` columns" in `graph.
_handle_flatten` (spatial positions after a flatten), `vgg.
_shrink_classifier_input` (the same thing, VGG-specific) and `surgery.
prune_attention_heads` (`head_dim` rows per head). Both are vectorized --
a boolean mask + `nonzero()`, and broadcast arithmetic -- never a Python
`set()` or per-index loop (KT.md section 3's last design-decision row).
"""
from __future__ import annotations

import torch


def complement_indices(n: int, idx: torch.Tensor | None) -> torch.Tensor:
    """Indices `0..n-1` not present in `idx`, ascending. `idx=None` means
    nothing is excluded (returns every index)."""
    mask = torch.ones(n, dtype=torch.bool)
    if idx is not None:
        mask[idx.to(torch.long)] = False
    return mask.nonzero(as_tuple=True)[0]


def expand_blocks(idx: torch.Tensor, block_size: int) -> torch.Tensor:
    """Expand each index `i` to the contiguous block `[i * block_size, (i + 1)
    * block_size)`, preserving the order of `idx`. E.g. channel indices ->
    the flat feature indices they own after an NCHW flatten
    (`block_size = H * W`), or head indices -> the Q/K/V rows they own
    (`block_size = head_dim`)."""
    idx = idx.to(torch.long)
    return (idx.unsqueeze(1) * block_size + torch.arange(block_size)).reshape(-1)
