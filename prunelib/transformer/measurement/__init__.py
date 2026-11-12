"""Running real data through a model and keeping what the pruning criteria
need from it.

- `hooks` -- `run_hooked` / `flatten_tokens` / `model_device`: the
  hook-and-forward loop every measurement below shares.
- `ffn_calibration` -- `FFNCalibrator` -> `CalibrationStats` (per-neuron
  activation moments), plus pairwise cross moments for merge scales.
- `head_calibration` -- `HeadCalibrator` -> `HeadStats` (per-head
  contribution moments) and Michel et al. gradient importance.
- `activation_recording` -- `ActivationRecorder`: the full per-token firing
  pattern, for twin detection.

Depends on: `models`.
"""
