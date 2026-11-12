"""Running a model over batches with forward hooks attached -- the one
mechanism every measurement in this package is built on.

`ffn_calibration.FFNCalibrator`, `head_calibration.HeadCalibrator` and
`activation_recording.ActivationRecorder` all do the same four things:
hook some modules, move each batch to the model's device, run it, and
clean up -- and they differ only in what their hooks compute. That shared
part lived first in three copies (one per measurement, two of them in
`ffn_calibration.py`), and then as a private helper that `head_calibration`
had to import across a module boundary. Here it is a module of its own,
so a measurement is only its hook (Single Responsibility), and a fix to
the loop -- e.g. which keys to strip, or how padding is masked -- lands in
every measurement at once.
"""
import torch


def model_device(model) -> torch.device:
    """The device to move batches to: `model.device` if the model has one
    (HuggingFace models do), else its first parameter's."""
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def flatten_tokens(tensor: torch.Tensor, mask) -> torch.Tensor:
    """Flatten a `(batch, seq, features)` tensor to `(valid_tokens, features)`,
    dropping padding positions per `mask` (a `(batch, seq)` attention mask, or
    `None` to keep everything). Detached; dtype unchanged; possibly empty.

    Every hook masks through this one function, so no two measurements can
    disagree about which tokens count. Padding is an artifact of batching:
    counting it would drag every mean toward whatever the model emits on
    `[PAD]`, and make every neuron look co-active with every other.
    """
    rows = tensor.detach().reshape(-1, tensor.shape[-1])
    if mask is not None:
        rows = rows[mask.reshape(-1).to(torch.bool)]
    return rows


def run_hooked(model, batches, registrations, state, max_batches=None, step=None):
    """Run `batches` through `model` in eval mode with hooks attached.

    `registrations` is a list of `(module, hook, pre)`: `pre=True`
    registers a forward *pre*-hook (sees the module's input), otherwise an
    ordinary forward hook (sees its output). Before each batch,
    `state["mask"]` is set to that batch's `attention_mask` (or `None`),
    which is how hooks know which token positions are padding.

    `step(batch)` replaces the default `torch.no_grad()` forward, for a
    caller that needs gradients (see
    `head_calibration.HeadCalibrator.collect_gradient_importance`); the
    default also strips `labels`, so a classification model is never asked
    for a loss nobody reads.

    Every hook is removed and the model's train/eval mode restored even if
    a batch raises -- a hook left behind would keep firing on every later
    forward pass, long after the measurement ended.
    """
    device = model_device(model)
    was_training = model.training
    model.eval()
    handles = [
        module.register_forward_pre_hook(hook) if pre else module.register_forward_hook(hook)
        for module, hook, pre in registrations
    ]
    try:
        for i, batch in enumerate(batches):
            if max_batches is not None and i >= max_batches:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            state["mask"] = batch.get("attention_mask")
            if step is None:
                batch.pop("labels", None)
                with torch.no_grad():
                    model(**batch)
            else:
                step(batch)
    finally:
        for handle in handles:
            handle.remove()
        state["mask"] = None
        if was_training:
            model.train()
