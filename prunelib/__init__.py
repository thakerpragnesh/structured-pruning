from .saliency import (
    available_methods,
    compute_score,
    keep_indices,
    l1_saliency,
    l2_saliency,
    max_k_saliency,
    prune_count,
    random_saliency,
    register_saliency,
    select_prune_indices,
)
from .surgery import prune_attention_heads, prune_conv_bn, prune_ffn_block
from .scanners import CoActivationScanner, pairwise_distance_matrix, register_metric
# vgg.py imports torchvision lazily (inside build_vgg16 only), so this import
# is safe without the optional `vision-experiments` extra installed.
from .vgg import build_vgg16, compress_masked_vgg, mask_vgg_layer, prune_vgg_layer, vgg_conv_bn_positions
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
    "available_methods",
    "compute_score",
    "keep_indices",
    "l1_saliency",
    "l2_saliency",
    "max_k_saliency",
    "prune_count",
    "random_saliency",
    "register_saliency",
    "select_prune_indices",
    "prune_conv_bn",
    "prune_ffn_block",
    "prune_attention_heads",
    "CoActivationScanner",
    "pairwise_distance_matrix",
    "register_metric",
    "count_encoder_params",
    "count_params",
    "measure_latency",
    "build_channel_mask",
    "commit_mask",
    "compress_masked_conv_bn",
    "mask_channels",
    "surviving_channels",
    "zeroed_channels",
    "build_vgg16",
    "compress_masked_vgg",
    "mask_vgg_layer",
    "prune_vgg_layer",
    "vgg_conv_bn_positions",
]

__version__ = "0.1.0"
