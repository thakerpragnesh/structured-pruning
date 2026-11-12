import pytest
import torch

from prunelib.transformer import kmeans_assign


def test_kmeans_separates_two_well_separated_blobs():
    x = torch.cat([
        torch.zeros(4, 3) + torch.tensor([1.0, 0.0, 0.0]),
        torch.zeros(4, 3) + torch.tensor([0.0, 0.0, 1.0]),
    ])
    assignment = kmeans_assign(x, k=2, seed=0)
    first_group, second_group = assignment[:4], assignment[4:]
    assert len(set(first_group.tolist())) == 1
    assert len(set(second_group.tolist())) == 1
    assert first_group[0] != second_group[0]


def test_kmeans_k_equal_n_gives_each_point_its_own_cluster():
    x = torch.randn(5, 3)
    assignment = kmeans_assign(x, k=5, seed=1)
    assert len(set(assignment.tolist())) == 5


def test_kmeans_rejects_out_of_range_k():
    x = torch.randn(3, 2)
    with pytest.raises(ValueError, match="k must be in"):
        kmeans_assign(x, k=0)
    with pytest.raises(ValueError, match="k must be in"):
        kmeans_assign(x, k=4)


def test_kmeans_is_reproducible_for_a_fixed_seed():
    x = torch.randn(20, 4)
    a = kmeans_assign(x, k=4, seed=7)
    b = kmeans_assign(x, k=4, seed=7)
    assert torch.equal(a, b)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_kmeans_runs_on_cuda():
    """The convergence check compares this iteration's assignment with the
    last one; both must live on the input's device, or `torch.equal` raises
    on the first iteration."""
    x = torch.randn(12, 3, device="cuda")
    assignment = kmeans_assign(x, k=3)
    assert assignment.device == x.device
