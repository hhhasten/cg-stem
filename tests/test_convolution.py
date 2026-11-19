"""Тести двовимірної згортки."""

from __future__ import annotations

import numpy as np
import pytest

from cgstem.ops.convolution import convolve2d
from cgstem.ops.kernels import gaussian_kernel


def test_box_kernel_keeps_constant_image() -> None:
    """Однорідне ядро не змінює однорідне зображення."""
    image = np.full((3, 3), 2.0)
    kernel = np.full((3, 3), 1 / 9)
    result = convolve2d(image, kernel)
    np.testing.assert_allclose(result, image)


def test_zero_padding_uses_only_overlapping_pixels() -> None:
    """З нульовим доповненням кут бере 4 пікселі, центр усі 9."""
    image = np.ones((3, 3), dtype=np.float64)
    kernel = np.full((3, 3), 1 / 9)
    result = convolve2d(image, kernel, mode="constant")
    expected = np.array(
        [
            [4, 6, 4],
            [6, 9, 6],
            [4, 6, 4],
        ],
        dtype=np.float64,
    ) / 9
    np.testing.assert_allclose(result, expected)


def test_identity_kernel_keeps_image() -> None:
    """Одиничне ядро повертає те саме зображення."""
    image = np.arange(9, dtype=np.float64).reshape(3, 3)
    kernel = np.zeros((3, 3), dtype=np.float64)
    kernel[1, 1] = 1.0
    result = convolve2d(image, kernel)
    np.testing.assert_allclose(result, image)


def test_even_kernel_is_rejected() -> None:
    """Ядро парного розміру не допускається."""
    image = np.ones((3, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        convolve2d(image, np.ones((2, 2)))


def test_three_dimensional_image_is_rejected() -> None:
    """Кольорове зображення не допускається."""
    image = np.ones((3, 3, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        convolve2d(image, np.ones((3, 3)))


def test_gaussian_kernel_is_normalised() -> None:
    """Гаусове ядро має одиничну суму та симетричне."""
    kernel = gaussian_kernel(5, 1.0)
    assert kernel.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1])
