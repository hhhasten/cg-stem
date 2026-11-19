"""Тести метрик якості."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cgstem.metrics.mse import compute_mse
from cgstem.metrics.psnr import compute_psnr
from cgstem.metrics.ssim import compute_ssim


def test_mse_of_identical_images_is_zero() -> None:
    """Однакові зображення мають нульову похибку."""
    image = np.arange(9, dtype=np.uint8).reshape(3, 3)
    assert compute_mse(image, image) == 0.0


def test_mse_counts_squared_difference() -> None:
    """Похибка на зображенні 2x2 з різницею 0, 1, 2, 3 дорівнює 3.5."""
    original = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    processed = np.zeros((2, 2), dtype=np.uint8)
    assert compute_mse(original, processed) == pytest.approx(3.5)


def test_psnr_of_identical_images_is_infinite() -> None:
    """Для однакових зображень PSNR необмежений."""
    image = np.zeros((3, 3), dtype=np.uint8)
    assert math.isinf(compute_psnr(image, image))


def test_psnr_of_maximal_difference_is_zero() -> None:
    """Максимальна різниця дає 0 дБ."""
    black = np.zeros((3, 3), dtype=np.uint8)
    white = np.full((3, 3), 255, dtype=np.uint8)
    assert compute_psnr(black, white) == pytest.approx(0.0)


def test_psnr_of_unit_difference() -> None:
    """Різниця в одиницю дає 48.13 дБ."""
    original = np.zeros((3, 3), dtype=np.uint8)
    processed = np.ones((3, 3), dtype=np.uint8)
    assert compute_psnr(original, processed) == pytest.approx(48.1308, abs=1e-4)


def test_ssim_of_identical_images_is_one() -> None:
    """Для однакових зображень SSIM дорівнює одиниці."""
    image = (np.arange(225, dtype=np.uint8).reshape(15, 15) * 10).astype(np.uint8)
    assert compute_ssim(image, image) == pytest.approx(1.0)


def test_ssim_of_constant_images() -> None:
    """Дві однорідні ділянки 100 і 200 дають SSIM 0.800026."""
    first = np.full((15, 15), 100, dtype=np.uint8)
    second = np.full((15, 15), 200, dtype=np.uint8)
    assert compute_ssim(first, second) == pytest.approx(0.800026, abs=1e-5)


def test_ssim_rejects_small_image() -> None:
    """Зображення менше за вікно не обробляється."""
    small = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_ssim(small, small)


def test_metric_shape_mismatch_is_rejected() -> None:
    """Різні розміри зображень не допускаються."""
    first = np.zeros((3, 3), dtype=np.uint8)
    second = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_mse(first, second)
    with pytest.raises(ValueError):
        compute_ssim(first, second)
