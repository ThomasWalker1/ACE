import pytest
import torch

from ace import bpp_proxy


def test_proxy_returns_one_finite_value_per_image() -> None:
    image = torch.rand(3, 3, 17, 19)
    rate = bpp_proxy(image)
    assert rate.shape == (3,)
    assert torch.isfinite(rate).all()


def test_proxy_is_differentiable() -> None:
    image = torch.rand(1, 3, 16, 16, requires_grad=True)
    bpp_proxy(image).sum().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_proxy_rejects_non_bchw_input() -> None:
    with pytest.raises(ValueError, match="4D BCHW"):
        bpp_proxy(torch.rand(3, 16, 16))
