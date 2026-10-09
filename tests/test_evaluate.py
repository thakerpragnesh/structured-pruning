import torch
import torch.nn as nn

from prunelib.evaluate import count_params, estimate_size_bytes, measure_latency


def test_measure_latency_restores_every_submodule_mode():
    """It switched the model to eval mode to time it and left it there, so a
    model timed mid-training stopped using dropout and updating BatchNorm
    statistics. Mixed modes (a frozen layer kept in eval) come back too."""
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5)).train()
    model[1].eval()
    measure_latency(model, torch.randn(2, 4), n_warmup=1, n_iters=1)
    assert [m.training for m in model.modules()] == [True, True, False, True]


def test_estimate_size_bytes_scales_with_bit_width():
    model = nn.Linear(10, 5)  # 50 weights + 5 bias = 55 params
    params = count_params(model)
    assert params == 55

    fp32 = estimate_size_bytes(model, bits_per_param=32.0)
    fp16 = estimate_size_bytes(model, bits_per_param=16.0)
    int8 = estimate_size_bytes(model, bits_per_param=8.0)

    assert fp32 == params * 4
    assert fp16 == params * 2
    assert int8 == params * 1
    assert fp16 == fp32 / 2
    assert int8 == fp32 / 4


def test_estimate_size_bytes_defaults_to_float32():
    model = nn.Linear(10, 5)
    assert estimate_size_bytes(model) == estimate_size_bytes(model, bits_per_param=32.0)
