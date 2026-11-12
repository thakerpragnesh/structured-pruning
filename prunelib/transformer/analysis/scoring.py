"""Saliency scoring strategies.

Each strategy scores a weight tensor's output units (conv filters or
linear neurons) behind one shared interface. Any consumer (a network
scanner, a neuron selector) accepts a `SaliencyScorer` and never needs to
know which concrete scoring rule it got (Liskov substitution) or be
edited when a new one is added (Open/Closed).

Scorers here see only *one* weight tensor, so they can only express
weight-magnitude criteria. Criteria that also need the FFN's *output*
projection or calibration activations (see `ffn_calibration.py`) are
expressed as `NeuronSelector`s instead, which receive an `FFNContext` --
see `ffn_selectors.py`.
"""
from abc import ABC, abstractmethod

import torch

from ...saliency import compute_score, csd_saliency


class SaliencyScorer(ABC):
    @abstractmethod
    def score(self, weight: torch.Tensor) -> torch.Tensor:
        """Return one score per output unit (dim 0 of `weight`)."""


def _as_groups(weight: torch.Tensor) -> torch.Tensor:
    """Reshape a weight to `(out_units, group, position)`.

    A 2D linear weight `(out, in)` becomes `(out, 1, in)`: every incoming
    connection is one "position" in a single group. A 4D conv weight
    `(out, in_ch, kh, kw)` becomes `(out, in_ch, kh * kw)`: each input
    channel is a group, each kernel tap a position. Ranks in between and
    above follow the same rule (dim 1 is the group, the rest flattens),
    which is what lets one code path cover conv and linear.
    """
    if weight.dim() == 2:
        return weight.unsqueeze(1)
    if weight.dim() >= 3:
        return weight.reshape(weight.shape[0], weight.shape[1], -1)
    raise ValueError(f"Expected a weight of rank >= 2, got {weight.dim()}D")


class TopKMagnitudeScorer(SaliencyScorer):
    """Per output unit, sum its `k` largest-magnitude incoming weights.

    For a conv filter the top-`k` is taken per input channel across
    kernel positions and then summed over channels; for a linear neuron
    it is simply the `k` strongest incoming connections. `k=3` is the
    thesis's Max-3 rule (`Max3SaliencyScorer`).
    """

    def __init__(self, k: int = 3):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k

    def score(self, weight: torch.Tensor) -> torch.Tensor:
        groups = _as_groups(weight.detach()).abs()
        positions = groups.shape[-1]
        k = min(self.k, positions)
        # When k covers the whole axis, top-k *is* the axis -- skip the sort.
        top = groups if k == positions else torch.topk(groups, k=k, dim=-1, sorted=False).values
        return top.sum(dim=(1, 2))


class Max3SaliencyScorer(TopKMagnitudeScorer):
    """The thesis's Max-3 criterion: `TopKMagnitudeScorer(k=3)`."""

    def __init__(self):
        super().__init__(k=3)


class LpNormScorer(SaliencyScorer):
    """Per output unit, the Lp norm of all its incoming weights.

    The standard magnitude baseline the Max-3 rule is meant to beat --
    kept here so an experiment can report both from the same scanner
    rather than hand-rolling the comparison.
    """

    def __init__(self, p: float = 2.0):
        self.p = p

    def score(self, weight: torch.Tensor) -> torch.Tensor:
        flat = weight.detach().reshape(weight.shape[0], -1)
        return torch.linalg.vector_norm(flat, ord=self.p, dim=1)


class CSDScorer(SaliencyScorer):
    """Per output unit, the L1 dispersion of its incoming weights from
    their own mean: `sum(|w - mean(w)|)`.

    Adapted from the "Custom Standard Deviation" (CSD) in Thaker & Mohan,
    "Enhancing Deep Compression of CNNs" (IEEE Access, 2024), which used
    dispersion as a training-time regularization target -- channels with
    low dispersion were pushed toward zero by an `L1Norm/CSD` loss added
    to the fine-tuning objective, then removed once shrunk. Used directly
    as a scorer here instead: a unit whose incoming weights are nearly
    uniform contributes a roughly constant signal regardless of the
    input -- indistinguishable from a bias term -- while high dispersion
    means the unit actually discriminates between inputs. No retraining
    pass is needed to compute it, unlike the source paper's regularizer.
    """

    def score(self, weight: torch.Tensor) -> torch.Tensor:
        # One implementation shared with the CNN side (`prunelib.compute_score(..., "csd")`).
        return csd_saliency(weight)


class RegisteredScorer(SaliencyScorer):
    """Adapter from the core `prunelib` saliency registry to this interface:
    any method name `prunelib.compute_score` accepts -- the built-ins or one
    added with `prunelib.register_saliency` -- becomes a `SaliencyScorer`
    usable by `SaliencySelector` and `NetworkSaliencyScanner`.

    Note the core `"max_k"` treats a Linear weight as a 1x1 conv (top-k
    over the *kernel* of each input feature, which has one tap), so on an
    FFN `W_in` it reduces to the L1 norm. For "the k strongest incoming
    connections" of a Linear neuron, use `TopKMagnitudeScorer` instead.
    """

    def __init__(self, method: str, **kwargs):
        self.method = method
        self.kwargs = kwargs

    def score(self, weight: torch.Tensor) -> torch.Tensor:
        return compute_score(weight, method=self.method, **self.kwargs)


def lowest_scoring(scorer: SaliencyScorer, weight: torch.Tensor, amount: int):
    """Convenience for one-off lookups: the `amount` lowest-scoring units
    for a single weight tensor, as `[index, score]` pairs.
    """
    scores = scorer.score(weight)
    amount = max(0, min(amount, scores.shape[0]))
    values, indices = torch.topk(scores, k=amount, largest=False)
    # One host sync for the whole batch, not one per element.
    return [[i, v] for i, v in zip(indices.tolist(), values.tolist())]
