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

Index sets live on the CPU, whatever device the weights they index are on:
every function in `prunelib` that returns channel indices returns a CPU
`long` tensor, and every function that indexes a tensor with them moves
them to that tensor's device first (`on_device_of`). They used to land on
whichever device produced them -- a CUDA weight's scores gave CUDA indices,
`complement_indices` built a CPU mask -- so pruning a model on a GPU failed
at the first `mask[cuda_idx]` or `cuda_weight.index_select(0, cpu_idx)`.

Every factory call in `prunelib` that builds an index set, a mask over one,
or a draw from a CPU generator says `device="cpu"` explicitly. Left implicit,
they followed torch's default device, so under `torch.set_default_device(
"cuda")` or `with torch.device("cuda"):` (common in LLM code)
`complement_indices` returned CUDA indices, `expand_blocks` mixed devices,
and `"random"`/`"kmeans"` selection failed (a CPU generator asked for a CUDA
tensor).
"""
from __future__ import annotations

import torch


def as_index(idx: torch.Tensor) -> torch.Tensor:
    """`idx` as the CPU `long` tensor every index set in `prunelib` is."""
    return idx.to(device="cpu", dtype=torch.long)


def on_device_of(idx: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """`idx` as a `long` tensor on `tensor`'s device, ready to index it."""
    return idx.to(device=tensor.device, dtype=torch.long)


def complement_indices(n: int, idx: torch.Tensor | None) -> torch.Tensor:
    """Indices `0..n-1` not present in `idx`, ascending. `idx=None` means
    nothing is excluded (returns every index)."""
    mask = torch.ones(n, dtype=torch.bool, device="cpu")
    if idx is not None:
        mask[as_index(idx)] = False
    return mask.nonzero(as_tuple=True)[0]


def expand_blocks(idx: torch.Tensor, block_size: int) -> torch.Tensor:
    """Expand each index `i` to the contiguous block `[i * block_size, (i + 1)
    * block_size)`, preserving the order of `idx`. E.g. channel indices ->
    the flat feature indices they own after an NCHW flatten
    (`block_size = H * W`), or head indices -> the Q/K/V rows they own
    (`block_size = head_dim`)."""
    return (as_index(idx).unsqueeze(1) * block_size + torch.arange(block_size, device="cpu")).reshape(-1)
