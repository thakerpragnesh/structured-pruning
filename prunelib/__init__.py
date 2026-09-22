from .saliency import (
    compute_score,
    keep_indices,
    l1_saliency,
    l2_saliency,
    max_k_saliency,
    random_saliency,
    select_prune_indices,
)
from .surgery import prune_attention_heads, prune_conv_bn, prune_ffn_block
from .scanners import CoActivationScanner, pairwise_distance_matrix
from .evaluate import count_encoder_params, count_params, estimate_size_bytes, measure_latency
from .masking import (
    build_channel_mask,
    commit_mask,
    compress_masked_conv_bn,
    mask_channels,
    surviving_channels,
    zeroed_channels,
)
from .graph import DependencyGraph, LeafTracer, PruningGroup, prune_model
from .quantization import (
    Int8Tensor,
    dequantize_float16,
    dequantize_int8_linear,
    quantize_fixed_point32,
    quantize_float16,
    quantize_int8_linear,
    quantize_model_,
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
    "prune_attention_heads",
    "CoActivationScanner",
    "pairwise_distance_matrix",
    "count_encoder_params",
    "count_params",
    "estimate_size_bytes",
    "measure_latency",
    "build_channel_mask",
    "commit_mask",
    "compress_masked_conv_bn",
    "mask_channels",
    "surviving_channels",
    "zeroed_channels",
    "DependencyGraph",
    "PruningGroup",
    "LeafTracer",
    "prune_model",
    "Int8Tensor",
    "quantize_float16",
    "dequantize_float16",
    "quantize_int8_linear",
    "dequantize_int8_linear",
    "quantize_fixed_point32",
    "quantize_model_",
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
