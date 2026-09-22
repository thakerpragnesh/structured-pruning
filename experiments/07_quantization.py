"""
07_quantization.py — the pruning + quantization pipeline in one place, runs
in a few seconds on CPU.

Prunes a small conv net with `prune_model` (generic, `DependencyGraph`-based
pruning -- see `06_generic_pruning.py`), then applies each of the three
post-training quantization methods from thesis Ch. 6.6 (`prunelib.quantization`,
which closes the gap KT.md section 10.2 used to describe: quantization
wasn't implemented anywhere in this codebase) to the pruned model, and
reports estimated size and the resulting change in output relative to the
unquantized pruned model as a cheap proxy for "how much did precision loss
change what the model computes" -- not a real accuracy-drop measurement,
since that needs real data and labels this experiment doesn't have.

    python experiments/07_quantization.py
"""
import copy

import torch
import torch.nn as nn

from prunelib import count_params, estimate_size_bytes, prune_model
from prunelib.quantization import quantize_fixed_point32, quantize_int8_linear, quantize_model_


class ConvNet(nn.Module):
    def __init__(self, in_ch=3, mid_ch=32, out_ch=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.bn1(self.conv1(x))))


def _output_drift(reference: torch.Tensor, other: torch.Tensor) -> float:
    return (reference - other).abs().mean().item()


def main():
    torch.manual_seed(0)
    model = ConvNet()
    example_input = torch.randn(2, 3, 32, 32)

    params_before = count_params(model)
    prune_model(model, example_input, model.conv1, prune_fraction=0.5, method="max_k", k=3)
    params_after = count_params(model)
    print(f"pruning: {params_before:,} -> {params_after:,} params "
          f"({(1 - params_after / params_before):.1%} reduction)")

    model.eval()
    with torch.no_grad():
        reference_out = model(example_input)
    size_fp32 = estimate_size_bytes(model, bits_per_param=32.0)

    print(f"\n{'method':<16} {'est. size (bytes)':>18} {'vs fp32':>10} {'mean |delta| vs fp32 output':>30}")
    print(f"{'float32 (base)':<16} {size_fp32:>18,.0f} {'1.00x':>10} {'--':>30}")

    fp16_model = copy.deepcopy(model)
    quantize_model_(fp16_model, method="float16")
    with torch.no_grad():
        fp16_out = fp16_model(example_input)
    size_fp16 = estimate_size_bytes(fp16_model, bits_per_param=16.0)
    print(f"{'float16':<16} {size_fp16:>18,.0f} {size_fp16/size_fp32:>9.2f}x "
          f"{_output_drift(reference_out, fp16_out):>30.6f}")

    fixed_model = copy.deepcopy(model)
    quantize_model_(fixed_model, method="fixed_point32", integer_bits=3, fractional_bits=28)
    with torch.no_grad():
        fixed_out = fixed_model(example_input)
    size_fixed = estimate_size_bytes(fixed_model, bits_per_param=32.0)  # still 32 bits total, precision-only
    print(f"{'fixed_point32':<16} {size_fixed:>18,.0f} {size_fixed/size_fp32:>9.2f}x "
          f"{_output_drift(reference_out, fixed_out):>30.6f}")

    # INT8 realizes real compression, but returns a scale/zero-point struct
    # per tensor rather than a drop-in module -- quantize just conv1's
    # weight here to show the round trip, rather than rewiring a whole model.
    q = quantize_int8_linear(model.conv1.weight)
    size_conv1_fp32 = estimate_size_bytes(model.conv1, bits_per_param=32.0)
    size_conv1_int8 = estimate_size_bytes(model.conv1, bits_per_param=8.0)
    print(f"{'int8 (conv1 only)':<16} {size_conv1_int8:>18,.0f} {size_conv1_int8/size_conv1_fp32:>9.2f}x "
          f"{'(vs conv1 fp32; see prunelib.quantization)':>30}")
    print(f"  conv1 weight: scale={q.scale:.6f}, zero_point={q.zero_point}")


if __name__ == "__main__":
    main()
