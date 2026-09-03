import pytest
import torch

from titans_pytorch.neural_memory import softclamp_grad_norm


@pytest.mark.parametrize('dtype', (torch.float32, torch.float16, torch.bfloat16))
def test_softclamp_zero_gradient_stays_finite(dtype):
    gradients = torch.zeros((2, 3, 4), dtype = dtype)

    clipped = softclamp_grad_norm(gradients, max_value = 1.)

    assert clipped.shape == gradients.shape
    assert torch.isfinite(clipped).all()
    assert torch.equal(clipped, gradients)


def test_softclamp_preserves_nonzero_direction():
    gradients = torch.tensor([[[3., 4., 0., 0.]]])

    clipped = softclamp_grad_norm(gradients, max_value = 2.)

    assert torch.isfinite(clipped).all()
    assert torch.allclose(clipped / clipped.norm(dim = -1, keepdim = True), gradients / gradients.norm(dim = -1, keepdim = True))
    assert clipped.norm(dim = -1).item() <= 2.
