import torch
import torch.nn as nn
import pytest

from prunelib.quantization import (
    dequantize_float16,
    dequantize_int8_linear,
    quantize_fixed_point32,
    quantize_float16,
    quantize_int8_linear,
    quantize_model_,
)


def test_quantize_float16_actually_reduces_precision():
    x = torch.tensor([1.0 / 3.0, 100.123456, -7.0])
    q = quantize_float16(x)
    assert q.dtype == torch.float16
    assert not torch.equal(q.to(torch.float32), x)  # precision was actually lost
    back = dequantize_float16(q)
    assert back.dtype == torch.float32
    assert torch.allclose(back, x, atol=1e-1)  # fp16 has ~3 decimal digits


def test_int8_quantize_matches_the_documented_formula():
    """Independent brute-force computation of scale/zero_point/x_int8 from
    the thesis's own Eq. 6.7-6.9, compared against the function's output --
    same pattern as saliency.py's brute-force comparisons."""
    torch.manual_seed(0)
    x = torch.randn(5, 5) * 3
    q = quantize_int8_linear(x)

    r_min, r_max = x.min().item(), x.max().item()
    expected_scale = (r_max - r_min) / (127 - (-128))
    expected_zero_point = max(-128, min(127, round(-128 - r_min / expected_scale)))
    expected_int8 = torch.clamp(torch.round(x / expected_scale) + expected_zero_point, -128, 127).to(torch.int8)

    assert q.scale == pytest.approx(expected_scale)
    assert q.zero_point == expected_zero_point
    assert torch.equal(q.values, expected_int8)
    assert q.original_shape == x.shape


def test_int8_roundtrip_error_is_bounded_by_scale():
    torch.manual_seed(1)
    x = torch.randn(20, 20) * 5
    q = quantize_int8_linear(x)
    back = dequantize_int8_linear(q)
    assert back.dtype == torch.float32
    # Rounding to the nearest quantization level can't be off by more than
    # half a step (with a little slack for float rounding at the boundary).
    assert (back - x).abs().max().item() <= q.scale / 2 + 1e-6


def test_int8_constant_tensor_does_not_divide_by_zero():
    x = torch.full((3, 3), 2.5)
    q = quantize_int8_linear(x)
    back = dequantize_int8_linear(q)
    assert torch.allclose(back, x)  # constant round-trips exactly, whatever its magnitude


def test_int8_all_zero_tensor_does_not_divide_by_zero():
    x = torch.zeros(3, 3)
    q = quantize_int8_linear(x)
    assert q.scale == 1.0
    back = dequantize_int8_linear(q)
    assert torch.allclose(back, x)


def test_fixed_point32_matches_bruteforce_scale_round_for_in_range_values():
    x = torch.tensor([0.1, -0.25, 1.5, -3.999])
    result = quantize_fixed_point32(x, integer_bits=3, fractional_bits=28)
    scale = 2**28
    expected = torch.round(x * scale) / scale
    assert torch.allclose(result, expected)


def test_fixed_point32_clips_out_of_range_values_instead_of_wrapping():
    """The thesis's own point (Ch. 6.6.2): a value outside the representable
    range must clip to the boundary, not silently wrap around to a wildly
    different (possibly sign-flipped) value."""
    integer_bits, fractional_bits = 3, 4  # small format so "out of range" is easy to construct
    max_representable = (2 ** (integer_bits + fractional_bits) - 1) / 2**fractional_bits
    min_representable = -(2 ** (integer_bits + fractional_bits)) / 2**fractional_bits

    huge = torch.tensor([1000.0, -1000.0])
    result = quantize_fixed_point32(huge, integer_bits=integer_bits, fractional_bits=fractional_bits)

    assert result[0].item() == pytest.approx(max_representable)
    assert result[1].item() == pytest.approx(min_representable)


def test_quantize_model_in_place_applies_float16_precision_loss_but_keeps_fp32_dtype():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    original_weight = model.weight.clone()
    original_bias = model.bias.clone()

    returned = quantize_model_(model, method="float16")

    assert returned is model
    assert model.weight.dtype == torch.float32  # cast back, per quantize_model_'s docstring
    assert model.bias.dtype == torch.float32
    assert torch.allclose(model.weight, original_weight.half().float())
    assert torch.allclose(model.bias, original_bias.half().float())
    assert not torch.equal(model.weight, original_weight)  # precision was actually lost


def test_quantize_model_fixed_point32_touches_conv_and_batchnorm_too():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))
    with torch.no_grad():
        model[1].weight.fill_(0.33333)  # BN's default (1.0) round-trips exactly at any bit-width; this doesn't
    original_conv_weight = model[0].weight.clone()
    original_bn_weight = model[1].weight.clone()

    quantize_model_(model, method="fixed_point32", integer_bits=3, fractional_bits=4)

    assert model[0].weight.dtype == torch.float32
    assert not torch.equal(model[0].weight, original_conv_weight)
    assert not torch.equal(model[1].weight, original_bn_weight)


def test_quantize_model_unknown_method_raises():
    model = nn.Linear(4, 3)
    with pytest.raises(ValueError, match="unknown method"):
        quantize_model_(model, method="int8")
