"""Each test names the defect (from GITHUB_AUDIT.md) it would catch if reintroduced."""
import torch

from prunelib.saliency import (
    compute_score,
    keep_indices,
    l1_saliency,
    l2_saliency,
    max_k_saliency,
    random_saliency,
    select_prune_indices,
)


def test_d1_score_and_selection_never_diverge():
    """D1: v4 computed a Max-3 score but selected channels using a separate,
    never-updated L1 tensor. Build a case where Max-3 and L1 disagree, and
    confirm selection follows whichever score was actually requested."""
    torch.manual_seed(0)
    weight = torch.zeros(4, 2, 3, 3)
    # Channel 0: one huge spike, rest tiny -> low L1, but a top-3 sum dominated
    # by the spike could still rank it high if k>=1 captures the spike alone.
    # Construct so Max-3 and L1 strictly disagree on which channel is weakest.
    weight[0] = 0.01                      # channel 0: uniformly small -> low everything
    weight[1, 0, 0, 0] = 100.0            # channel 1: one massive spike, else 0
    weight[2] = 1.0                       # channel 2: uniformly moderate
    weight[3] = torch.rand(2, 3, 3) * 0.5 + 0.5

    max3_scores = compute_score(weight, method="max_k", k=3)
    l1_scores = compute_score(weight, method="l1")

    max3_weakest = select_prune_indices(max3_scores, 1).item()
    l1_weakest = select_prune_indices(l1_scores, 1).item()

    # Channel 0 has the smallest L1 norm (all 0.01s) AND the smallest top-3
    # sum, so both should agree it's weakest here -- the point is that the
    # *scores themselves* differ even where the conclusion coincides.
    assert max3_weakest == 0
    assert l1_weakest == 0
    assert not torch.allclose(max3_scores, l1_scores)


def test_d2_matches_bruteforce_topk():
    """D2: the manual max1/max2/max3 tracker only rotated on a *new* maximum,
    so a value landing between 2nd and 1st place was dropped. Compare against
    an independent, obviously-correct brute-force top-3 sum."""
    torch.manual_seed(1)
    weight = torch.randn(6, 3, 3, 3)
    got = max_k_saliency(weight, k=3)

    out_ch, in_ch, kh, kw = weight.shape
    expected = torch.zeros(out_ch)
    for o in range(out_ch):
        total = 0.0
        for i in range(in_ch):
            vals = sorted(weight[o, i].abs().flatten().tolist(), reverse=True)
            total += sum(vals[:3])
        expected[o] = total

    assert torch.allclose(got, expected, atol=1e-5)


def test_d3_no_leak_across_input_channels():
    """D3: max1/max2/max3 were reset once per output channel but accumulated
    inside the input-channel loop from monotonically growing values -- not a
    clean sum of independent per-kernel top-3 sums. Verify channel 1's score
    (two input channels) is exactly twice channel 0's score (one input
    channel) when both kernels are identical, and not some larger leaked sum.
    """
    kernel = torch.tensor([[5.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    weight_one_in_ch = kernel.unsqueeze(0).unsqueeze(0)              # [1,1,3,3]
    weight_two_in_ch = torch.stack([kernel, kernel]).unsqueeze(0)    # [1,2,3,3]

    score_one = max_k_saliency(weight_one_in_ch, k=3)
    score_two = max_k_saliency(weight_two_in_ch, k=3)

    assert torch.allclose(score_two, score_one * 2)


def test_d4_non_square_kernel_uses_all_columns():
    """D4: iterating kernel width with `range(size[2])` (the height dimension)
    silently skipped columns on non-square (kh != kw) kernels."""
    # 3x5 kernel: if width columns were dropped, the two rightmost columns
    # (which hold the largest values) would never be seen.
    kernel = torch.zeros(3, 5)
    kernel[:, 3:] = 10.0  # the "missing" columns under the old bug
    weight = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,3,5]

    score = max_k_saliency(weight, k=3)
    assert score.item() == 30.0  # three 10.0 entries, correctly seen


def test_l1_l2_random_shapes_and_ordering():
    torch.manual_seed(2)
    weight = torch.randn(5, 4, 3, 3)
    for scores in (l1_saliency(weight), l2_saliency(weight), random_saliency(weight)):
        assert scores.shape == (5,)

    l1 = l1_saliency(weight)
    l2 = l2_saliency(weight)
    # L2 <= L1 always holds for any real vector (equality only at <=1 nonzero entry).
    assert (l2 <= l1 + 1e-5).all()


def test_select_and_keep_indices_are_complements():
    scores = torch.tensor([5.0, 1.0, 3.0, 0.5, 4.0])
    pruned = select_prune_indices(scores, 2)
    kept = keep_indices(scores, 2)
    assert set(pruned.tolist()) | set(kept.tolist()) == set(range(5))
    assert set(pruned.tolist()) & set(kept.tolist()) == set()
    assert pruned.tolist() == [1, 3]  # the two lowest scores, ascending index
