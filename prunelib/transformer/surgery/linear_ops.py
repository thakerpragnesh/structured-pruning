"""Building a resized `nn.Linear` that is otherwise faithful to the one it
replaces -- the single mechanical step both surgeons end in.

`FFNSurgeon` and `AttentionSurgeon` each used to construct their new
Linears by hand, and that hand-written step is where real bugs lived: a
bare `nn.Linear(...)` defaults to float32 (CHANGELOG 0.2.0, bug 3 -- it
silently converted a half-precision model's FFN) and to
`requires_grad=True` (a frozen layer would quietly start training during a
"heal"). Three copies of those guarantees became one function.
"""
import torch
import torch.nn as nn


def rebuild_linear(source: nn.Linear, weight: torch.Tensor, bias=None) -> nn.Linear:
    """A new `nn.Linear` holding `weight` (`(out, in)`) and `bias`, on
    `source`'s device and in its dtype, with `source`'s `requires_grad`
    flags.

    `bias` must be given exactly when `source` has one: surgery never adds
    or removes a bias, it only reshapes or shifts it.
    """
    if (bias is None) != (source.bias is None):
        raise ValueError(
            "rebuild_linear: `bias` must be given exactly when the source Linear has one "
            f"(source has {'a' if source.bias is not None else 'no'} bias)."
        )
    new = nn.Linear(
        weight.shape[1], weight.shape[0], bias=bias is not None,
        device=source.weight.device, dtype=source.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(weight)
        if bias is not None:
            new.bias.copy_(bias)
    new.weight.requires_grad_(source.weight.requires_grad)
    if bias is not None:
        new.bias.requires_grad_(source.bias.requires_grad)
    return new
