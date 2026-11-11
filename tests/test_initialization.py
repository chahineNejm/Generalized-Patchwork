import pytest
import torch

from pwgp.initialization import hierarchical_supports, observed_supports


@pytest.mark.parametrize("dimensions", [1, 2, 5])
def test_global_and_patch_supports_are_distinct_observations(dimensions):
    generator = torch.Generator().manual_seed(17)
    x = torch.rand(200, dimensions, generator=generator, dtype=torch.float64)
    global_z = observed_supports(x, 12, seed=31)
    patch_z = hierarchical_supports(x, 3, 4, seed=31)
    for z in (global_z, patch_z.reshape(-1, dimensions)):
        assert bool((z[:, None] == x).all(-1).any(-1).all())
        assert len(torch.unique(z, dim=0)) == len(z)
        assert z.dtype == x.dtype and z.device == x.device
    torch.testing.assert_close(global_z, observed_supports(x, 12, seed=31))
    torch.testing.assert_close(patch_z, hierarchical_supports(x, 3, 4, seed=31))


def test_initialization_does_not_repeat_duplicate_rows_or_cap_budget():
    x = torch.tensor([[0.2], [0.2], [0.8]], dtype=torch.float64)
    assert len(torch.unique(observed_supports(x, 2), dim=0)) == 2
    with pytest.raises(ValueError, match="3 supports from 2 distinct"):
        observed_supports(x, 3)
    with pytest.raises(ValueError, match="distinct observed locations"):
        hierarchical_supports(x, 1, 3)


def test_boundary_observations_are_not_moved_to_initialize_logit_parameters():
    x = torch.tensor([[0.0], [0.2], [0.8], [1.0]], dtype=torch.float64)
    torch.testing.assert_close(observed_supports(x, 2).sort(dim=0).values, x[1:3])
    with pytest.raises(ValueError, match="3 supports from 2 distinct"):
        observed_supports(x, 3)
