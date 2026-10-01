from .saliency import (
    SALIENCY_METHODS,
    compute_score,
    keep_indices,
    l1_saliency,
    l2_saliency,
    max_k_saliency,
    random_saliency,
    register_saliency_method,
    select_prune_indices,
)
from .indices import complement_indices, expand_blocks
from .surgery import (
    is_depthwise_conv,
    prune_attention_heads,
    prune_conv_bn,
    prune_ffn_block,
    slice_batchnorm,
    slice_conv2d,
    slice_depthwise_conv2d,
    slice_linear,
)
from .distance import DISTANCE_METRICS, DistanceMetric, pairwise_distance, register_distance_metric
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
from .selection import SELECTION_METHODS, register_selection_method, select_prune_indices_by_method
from .clustering import kmeans, kmeans_prune_indices  # also registers the "kmeans" selection rule
from .module_rules import MODULE_RULES, ChannelRole, ModuleRule, find_module_rule, register_module_rule
from .graph import (
    OP_PROPAGATORS,
    DependencyGraph,
    LeafTracer,
    Propagation,
    PruningGroup,
    propagate_add,
    propagate_cat,
    propagate_flatten,
    prune_model,
    register_op_propagator,
)
from .quantization import (
    QUANTIZATION_METHODS,
    Int8Tensor,
    dequantize_float16,
    dequantize_int8_linear,
    quantize_fixed_point32,
    quantize_float16,
    quantize_int8_linear,
    quantize_model_,
    register_quantization_method,
)
# vgg.py's surgery functions are plain torch; only `build_vgg16` needs the
# optional torchvision dependency, and it imports it on call.
from .vgg import build_vgg16, compress_masked_vgg, mask_vgg_layer, prune_vgg_layer, vgg_conv_bn_positions

__all__ = [
    # scoring and selection
    "compute_score",
    "keep_indices",
    "l1_saliency",
    "l2_saliency",
    "max_k_saliency",
    "random_saliency",
    "select_prune_indices",
    "select_prune_indices_by_method",
    "kmeans",
    "kmeans_prune_indices",
    # structural surgery
    "prune_conv_bn",
    "prune_ffn_block",
    "prune_attention_heads",
    "slice_conv2d",
    "slice_depthwise_conv2d",
    "slice_linear",
    "slice_batchnorm",
    "is_depthwise_conv",
    "complement_indices",
    "expand_blocks",
    # redundancy scanning
    "CoActivationScanner",
    "pairwise_distance_matrix",
    "pairwise_distance",
    "DistanceMetric",
    # evaluation
    "count_encoder_params",
    "count_params",
    "estimate_size_bytes",
    "measure_latency",
    # two-phase masking
    "build_channel_mask",
    "commit_mask",
    "compress_masked_conv_bn",
    "mask_channels",
    "surviving_channels",
    "zeroed_channels",
    # dependency-graph pruning
    "DependencyGraph",
    "PruningGroup",
    "LeafTracer",
    "prune_model",
    "ModuleRule",
    "ChannelRole",
    "find_module_rule",
    "Propagation",
    "propagate_add",
    "propagate_cat",
    "propagate_flatten",
    # quantization
    "Int8Tensor",
    "quantize_float16",
    "dequantize_float16",
    "quantize_int8_linear",
    "dequantize_int8_linear",
    "quantize_fixed_point32",
    "quantize_model_",
    # VGG wiring
    "build_vgg16",
    "compress_masked_vgg",
    "mask_vgg_layer",
    "prune_vgg_layer",
    "vgg_conv_bn_positions",
    # extension points: add a method/metric/layer type/op without editing prunelib
    "SALIENCY_METHODS",
    "register_saliency_method",
    "SELECTION_METHODS",
    "register_selection_method",
    "DISTANCE_METRICS",
    "register_distance_metric",
    "MODULE_RULES",
    "register_module_rule",
    "OP_PROPAGATORS",
    "register_op_propagator",
    "QUANTIZATION_METHODS",
    "register_quantization_method",
]

__version__ = "0.1.0"
