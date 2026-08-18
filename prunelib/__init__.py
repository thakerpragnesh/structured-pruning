from .saliency import (
    compute_score,
    keep_indices,
    l1_saliency,
    l2_saliency,
    max_k_saliency,
    random_saliency,
    select_prune_indices,
)
from .surgery import prune_conv_bn, prune_ffn_block
from .scanners import CoActivationScanner, pairwise_distance_matrix
from .evaluate import count_encoder_params, count_params, measure_latency
from .masking import (
    build_channel_mask,
    commit_mask,
    compress_masked_conv_bn,
    mask_channels,
    surviving_channels,
    zeroed_channels,
)

__all__ = [
    "compute_score",
    "keep_indices",
    "l1_saliency",
    "l2_saliency",
    "max_k_saliency",
    "random_saliency",
    "select_prune_indices",
    "prune_conv_bn",
    "prune_ffn_block",
    "CoActivationScanner",
    "pairwise_distance_matrix",
    "count_encoder_params",
    "count_params",
    "measure_latency",
    "build_channel_mask",
    "commit_mask",
    "compress_masked_conv_bn",
    "mask_channels",
    "surviving_channels",
    "zeroed_channels",
]

# vgg.py imports torchvision, which is an optional dependency (`pip install
# structured-pruning[vision-experiments]`) -- don't make the whole package
# fail to import for someone who only wants the core, framework-agnostic
# saliency/surgery/scanner functions.
try:
    from .vgg import build_vgg16, compress_masked_vgg, mask_vgg_layer, prune_vgg_layer, vgg_conv_bn_positions

    __all__ += ["build_vgg16", "compress_masked_vgg", "mask_vgg_layer", "prune_vgg_layer", "vgg_conv_bn_positions"]
except ImportError:
    pass

__version__ = "0.1.0"
