"""Structured FFN-neuron and attention-head pruning for transformer models.

Merged in from the standalone `transformer_pruning` project. Import
everything from here (`from prunelib.transformer import X`): this namespace
is the public API, and the subpackages are its internal layout.

Relationship to core `prunelib`: core is the lowest layer -- this package
may import it (the CSD score and pairwise head distances are shared with the
CNN side, and `RegisteredScorer` exposes every core saliency method here),
but core never imports this package, so CNN-only users don't load it.
Core's `prunelib.prune_attention_heads(q, k, v, o, keep_heads, num_heads)`
is the stateless tensor-level primitive; `prunelib.transformer.prune_attention_heads(
model, layer_index, head_indices)` is the model-level entry point here,
which takes heads to DROP and also updates the layer's head bookkeeping.

The package is grouped by role, and each subpackage may only depend on the
ones listed before it. `tests/test_package_layout.py` enforces that order.

    models/        how to reach into a model: adapters, layer discovery
    measurement/   running data through it: hooks, calibration, recording
    analysis/      signals from weights / firings: scores, OV math, twins
    selection/     criteria: contexts, budgets, FFN and head selectors
    surgery/       resizing modules: FFN and attention surgeons
    pipeline/      orchestration: allocation strategies, prune_* functions
"""
from .models.layers import LayerHandle, LayerKind, discover_layers
from .models.adapters import (
    LayerStackAdapter,
    FFNLayerAdapter,
    AttentionLayerAdapter,
    TransformerLayerAdapter,
    BertLayerAdapter,
    register_adapter,
    resolve_adapter,
)
from .selection.budget import PruneBudgetMixin, keep_count
from .pipeline.allocation import (
    AllocationStrategy,
    GlobalAllocation,
    PrunableStructure,
    SCORE_NORMALIZERS,
    UniformAllocation,
)
from .analysis.scoring import (
    SaliencyScorer,
    TopKMagnitudeScorer,
    Max3SaliencyScorer,
    LpNormScorer,
    CSDScorer,
    RegisteredScorer,
    lowest_scoring,
)
from .selection.context import FFNContext, HeadContext, Selection
from .measurement.ffn_calibration import CalibrationStats, FFNCalibrator
from .measurement.head_calibration import HeadStats, HeadCalibrator
from .analysis.clustering import kmeans_assign
from .analysis.network_scanner import NetworkSaliencyScanner
from .analysis.head_analysis import AttentionHeadAnalyzer, ov_norms, ov_similarity
from .surgery.ffn_surgery import FFNSurgeon
from .surgery.attention_surgery import AttentionSurgeon
from .selection.ffn_selectors import (
    NeuronSelector,
    ImportanceSelector,
    SaliencySelector,
    InOutNormSelector,
    ActivationAwareSelector,
    TwinRedundancySelector,
    WeightClusterRedundancySelector,
)
from .selection.head_selectors import (
    HeadSelector,
    HeadImportanceSelector,
    OVNormHeadSelector,
    ActivationAwareHeadSelector,
    GradientHeadSelector,
    RedundantHeadSelector,
    HEAD_SIMILARITIES,
)
from .pipeline.workflow import (
    prune_ffn_layer,
    prune_model_ffn,
    prune_attention_heads,
    prune_attention_layer,
    prune_model_attention,
)
from .measurement.activation_recording import ActivationRecorder
from .analysis.redundancy import JaccardTwinFinder

__all__ = [
    "LayerHandle",
    "LayerKind",
    "discover_layers",
    "LayerStackAdapter",
    "FFNLayerAdapter",
    "AttentionLayerAdapter",
    "TransformerLayerAdapter",
    "BertLayerAdapter",
    "register_adapter",
    "resolve_adapter",
    "PruneBudgetMixin",
    "keep_count",
    "AllocationStrategy",
    "GlobalAllocation",
    "PrunableStructure",
    "SCORE_NORMALIZERS",
    "UniformAllocation",
    "SaliencyScorer",
    "TopKMagnitudeScorer",
    "Max3SaliencyScorer",
    "LpNormScorer",
    "CSDScorer",
    "RegisteredScorer",
    "lowest_scoring",
    "FFNContext",
    "HeadContext",
    "Selection",
    "CalibrationStats",
    "FFNCalibrator",
    "HeadStats",
    "HeadCalibrator",
    "kmeans_assign",
    "NetworkSaliencyScanner",
    "AttentionHeadAnalyzer",
    "ov_norms",
    "ov_similarity",
    "FFNSurgeon",
    "AttentionSurgeon",
    "NeuronSelector",
    "ImportanceSelector",
    "SaliencySelector",
    "InOutNormSelector",
    "ActivationAwareSelector",
    "TwinRedundancySelector",
    "WeightClusterRedundancySelector",
    "HeadSelector",
    "HeadImportanceSelector",
    "OVNormHeadSelector",
    "ActivationAwareHeadSelector",
    "GradientHeadSelector",
    "RedundantHeadSelector",
    "HEAD_SIMILARITIES",
    "prune_ffn_layer",
    "prune_model_ffn",
    "prune_attention_heads",
    "prune_attention_layer",
    "prune_model_attention",
    "ActivationRecorder",
    "JaccardTwinFinder",
]
