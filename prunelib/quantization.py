"""
Post-training quantization: precision-reduction methods applied to a
model's weights, as a separate compression stage from pruning. Reference
implementation of thesis Ch. 6.6 (Pragnesh Thaker's Ph.D. thesis, "Pruning
and Quantization Techniques for Deep Neural Network Acceleration", NITK
Surathkal, July 2025 -- see KT.md section 10.2, which is where this gap was
first written down). None of this existed anywhere in `prunelib` before
this module; `prunelib` covered pruning only.

Three methods, in the order the thesis found them best-to-worst by accuracy
retention on a pruned VGG16 (thesis Ch. 6.6.4):

- `quantize_float16` -- direct fp32->fp16 cast. Best tradeoff the thesis
  measured (halves memory for a 1.34% accuracy drop); widely supported
  natively by accelerators and mobile processors.
- `quantize_int8_linear` / `dequantize_int8_linear` -- the standard affine
  scale/zero-point mapping (thesis Eq. 6.7-6.10). The thesis includes this
  as a reference technique rather than one it ran end-to-end.
- `quantize_fixed_point32` -- a fixed 1-sign/3-integer/28-fractional bit
  format. Thesis found this markedly worse (3.26% accuracy drop) than
  Float16, because a fixed exponent can't adapt to each layer's actual
  weight distribution the way floating point does. Kept here to reproduce
  the thesis's own comparison, not as a recommended technique.

Float16 and fixed-point both round-trip a tensor and hand back something the
same shape, ready to keep computing with. INT8 is different -- realizing its
actual 4x memory reduction means storing `int8` values plus a separate
`scale`/`zero_point`, not a tensor you can substitute in place of the
original and keep doing float arithmetic against -- so it returns a small
`Int8Tensor` struct instead, with an explicit `dequantize_int8_linear` to
get back to float32 for computation.

Every function here operates on a `torch.Tensor` and returns a *new* one
(or a new struct), matching the rest of `prunelib`'s convention of never
mutating what's passed in (`surgery.py`, `graph.py`). `quantize_model_` is
the one deliberate, name-flagged exception -- there's no "new, smaller
module" to construct and return here the way surgery has one, since
quantization doesn't change any tensor's shape, only its precision -- see
its own docstring, which follows the same precedent `masking.commit_mask`
already set for an explicitly in-place operation in this library.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

_QUANTIZABLE_MODULE_TYPES = (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d)


def quantize_float16(tensor: torch.Tensor) -> torch.Tensor:
    """Direct fp32 (or any float dtype) -> fp16 cast."""
    return tensor.to(torch.float16)


def dequantize_float16(tensor: torch.Tensor) -> torch.Tensor:
    """Cast a Float16 tensor back to float32."""
    return tensor.to(torch.float32)


@dataclass
class Int8Tensor:
    """The result of `quantize_int8_linear`: `values` (int8) plus the
    `scale`/`zero_point` needed to recover approximate float32 values via
    `dequantize_int8_linear`. `original_shape` is stored for convenience --
    `values.shape` is already the same shape, but this makes the struct
    self-documenting without needing the original tensor in scope."""

    values: torch.Tensor
    scale: float
    zero_point: int
    original_shape: torch.Size


def quantize_int8_linear(tensor: torch.Tensor) -> Int8Tensor:
    """Linear (affine) INT8 quantization -- thesis Eq. 6.7-6.9:

        scale = (r_max - r_min) / (q_max - q_min)
        zero_point = round(q_min - r_min / scale)
        x_int8 = round(x_fp32 / scale) + zero_point

    with `[q_min, q_max] = [-128, 127]` for signed INT8. Values are clamped
    to that range after rounding (a tensor with outliers far from its bulk
    would otherwise overflow `int8`, which silently wraps rather than
    clips -- clamping first is what keeps this a lossy-but-bounded
    approximation instead of a wildly wrong one for a few entries).

    A constant tensor (`r_max == r_min`) would divide by zero in the general
    formula above; handled separately by setting `scale` to the constant's
    own magnitude (or `1.0` for an all-zero tensor) and `zero_point` to `0`
    -- `x_int8` is then just its sign, and `scale * sign` reproduces the
    original constant exactly regardless of magnitude, which the general
    formula's fixed `[-128, 127]` grid can't generally do for an arbitrary
    non-integer value.
    """
    r_min = tensor.min().item()
    r_max = tensor.max().item()
    q_min, q_max = -128, 127

    if r_max == r_min:
        scale = abs(r_min) if r_min != 0.0 else 1.0
        zero_point = 0
    else:
        scale = (r_max - r_min) / (q_max - q_min)
        zero_point = int(round(q_min - r_min / scale))
        zero_point = max(q_min, min(q_max, zero_point))

    x_int8 = torch.clamp(torch.round(tensor / scale) + zero_point, q_min, q_max).to(torch.int8)
    return Int8Tensor(values=x_int8, scale=scale, zero_point=zero_point, original_shape=tensor.shape)


def dequantize_int8_linear(quantized: Int8Tensor) -> torch.Tensor:
    """Recover an approximate float32 tensor: `scale * (x_int8 - zero_point)`
    (thesis Eq. 6.10)."""
    return (quantized.values.to(torch.float32) - quantized.zero_point) * quantized.scale


def quantize_fixed_point32(tensor: torch.Tensor, integer_bits: int = 3, fractional_bits: int = 28) -> torch.Tensor:
    """Round-trip `tensor` through a signed fixed-point format (1 sign bit +
    `integer_bits` + `fractional_bits`, thesis defaults 3/28 -- still 32
    bits total, so this is a precision experiment, not a memory-saving one,
    unlike Float16/INT8 above) and back to float32.

    Values outside the representable range `[-2^(integer_bits+fractional_bits),
    2^(integer_bits+fractional_bits) - 1]` are *clamped*, never wrapped --
    the thesis's own finding (Ch. 6.6.2) is that a fixed exponent can't
    rescale to fit a layer's actual weight distribution the way floating
    point can, so values at the tails are either wasted precision or lost;
    clamping is what "lost" means here, made explicit rather than silently
    wrapping to a wildly wrong value.
    """
    scale = 2**fractional_bits
    max_val = 2 ** (integer_bits + fractional_bits) - 1
    min_val = -(2 ** (integer_bits + fractional_bits))
    scaled = torch.round(tensor * scale)
    clamped = torch.clamp(scaled, min_val, max_val)
    return clamped / scale


_MODEL_METHODS = {
    "float16": quantize_float16,
    "fixed_point32": quantize_fixed_point32,
}


@torch.no_grad()
def quantize_model_(model: nn.Module, method: str = "float16", **kwargs) -> nn.Module:
    """Apply `method`'s quantization error to every Linear/Conv2d/BatchNorm
    weight (and bias, if present) in `model`, **in place**, then cast the
    result back to that parameter's original dtype. "float16" or
    "fixed_point32" only -- INT8 isn't included here because
    `quantize_int8_linear` returns a separate `Int8Tensor` struct, not a
    same-shape, same-dtype replacement this function's uniform copy-back
    loop can use; call `quantize_int8_linear`/`dequantize_int8_linear`
    directly per-tensor if you need INT8's actual memory savings.

    Casting back to float32 (rather than leaving the model in `float16`)
    is deliberate: this function's job is to measure a precision option's
    effect on the model's *output*, the same way the thesis evaluates
    accuracy drop after quantization -- not to actually shrink the model's
    storage. For real fp16 storage savings, call `model.half()` directly;
    nothing here does that better than PyTorch's own built-in.

    Mutates `model` in place and returns it for convenience -- unlike the
    rest of `prunelib`, which never mutates what's passed in, because
    there's no new, smaller module to construct and return instead (shapes
    are unchanged; only numeric precision is). This follows the same
    explicit, name-flagged exception `masking.commit_mask` already set.
    """
    if method not in _MODEL_METHODS:
        raise ValueError(f"unknown method {method!r}, expected one of {list(_MODEL_METHODS)}")
    fn = _MODEL_METHODS[method]

    for module in model.modules():
        if not isinstance(module, _QUANTIZABLE_MODULE_TYPES):
            continue
        if getattr(module, "weight", None) is not None:
            module.weight.copy_(fn(module.weight, **kwargs).to(module.weight.dtype))
        if getattr(module, "bias", None) is not None:
            module.bias.copy_(fn(module.bias, **kwargs).to(module.bias.dtype))
    return model
