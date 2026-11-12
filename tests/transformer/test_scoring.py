import pytest
import torch

from prunelib.transformer import CSDScorer, LpNormScorer, Max3SaliencyScorer, TopKMagnitudeScorer, lowest_scoring


def test_max3_linear_matches_reference():
    w = torch.tensor([[1.0, -5.0, 2.0, 0.5], [0.0, 0.0, 0.0, 9.0]])
    # Top-3 magnitudes: row 0 -> 5+2+1, row 1 -> 9+0+0
    assert torch.allclose(Max3SaliencyScorer().score(w), torch.tensor([8.0, 9.0]))


def test_max3_conv_sums_topk_per_input_channel():
    w = torch.zeros(2, 3, 2, 2)
    w[0, :, 0, 0] = 4.0  # one strong tap per input channel
    w[1] = 1.0           # uniformly weak
    scores = Max3SaliencyScorer().score(w)
    # Unit 0: 3 channels x (4 + 0 + 0). Unit 1: 3 channels x (1 + 1 + 1).
    assert torch.allclose(scores, torch.tensor([12.0, 9.0]))


def test_topk_degenerates_to_full_sum_when_k_exceeds_width():
    w = torch.randn(4, 2)
    assert torch.allclose(TopKMagnitudeScorer(k=99).score(w), w.abs().sum(dim=1))


def test_max3_is_topk_three():
    w = torch.randn(6, 10)
    assert torch.allclose(Max3SaliencyScorer().score(w), TopKMagnitudeScorer(k=3).score(w))


def test_scorer_rejects_bad_k_and_rank():
    with pytest.raises(ValueError, match="k must be"):
        TopKMagnitudeScorer(k=0)
    with pytest.raises(ValueError, match="rank >= 2"):
        Max3SaliencyScorer().score(torch.randn(5))


def test_lp_norm_scorer():
    w = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    assert torch.allclose(LpNormScorer(p=2).score(w), torch.tensor([5.0, 1.0]))
    assert torch.allclose(LpNormScorer(p=1).score(w), torch.tensor([7.0, 1.0]))


def test_scoring_does_not_require_grad_or_mutate():
    w = torch.randn(4, 5, requires_grad=True)
    before = w.detach().clone()
    scores = Max3SaliencyScorer().score(w)
    assert not scores.requires_grad
    assert torch.equal(w.detach(), before)


def test_csd_scorer_rewards_dispersion_not_magnitude():
    # Row 0 is perfectly uniform (zero dispersion) despite nonzero weights;
    # row 1 has the same L1 magnitude but is spread around its mean.
    w = torch.tensor([
        [5.0, 5.0, 5.0, 5.0],
        [10.0, 0.0, 10.0, 0.0],
    ])
    scores = CSDScorer().score(w)
    assert scores[0].item() == pytest.approx(0.0)
    assert scores[1].item() == pytest.approx(20.0)


def test_csd_scorer_does_not_require_grad_or_mutate():
    w = torch.randn(4, 5, requires_grad=True)
    before = w.detach().clone()
    scores = CSDScorer().score(w)
    assert not scores.requires_grad
    assert torch.equal(w.detach(), before)


def test_lowest_scoring_returns_weakest_first():
    w = torch.tensor([[9.0, 9.0], [0.1, 0.1], [5.0, 5.0]])
    result = lowest_scoring(Max3SaliencyScorer(), w, amount=2)
    assert [idx for idx, _ in result] == [1, 2]
    assert all(isinstance(idx, int) and isinstance(val, float) for idx, val in result)


def test_lowest_scoring_clamps_amount():
    w = torch.randn(3, 4)
    assert len(lowest_scoring(Max3SaliencyScorer(), w, amount=100)) == 3
    assert lowest_scoring(Max3SaliencyScorer(), w, amount=-5) == []
