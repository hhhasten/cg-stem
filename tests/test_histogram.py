"""Тести гістограмних перетворень."""

from __future__ import annotations

import numpy as np
import pytest

from cgstem.methods.histogram import clahe, clip_histogram

MANUAL_IMAGE = np.array(
    [
        [10, 20, 40, 40],
        [30, 40, 40, 40],
        [40, 40, 40, 40],
        [40, 40, 40, 40],
    ],
    dtype=np.uint8,
)


def test_clahe_manual_block_result() -> None:
    """Для блоку 2x2 зі значеннями 10, 20, 30, 40 таблиця дає 0, 85, 170, 255."""
    result = clahe(MANUAL_IMAGE, 2.0, (2, 2))
    assert result[0, 0] == 0
    assert result[1, 1] == 161


def test_clahe_keeps_constant_image() -> None:
    """Однорідне зображення не змінюється."""
    image = np.full((8, 8), 120, dtype=np.uint8)
    result = clahe(image, 2.0, (2, 2))
    np.testing.assert_array_equal(result, image)


def test_clahe_keeps_range_and_shape() -> None:
    """Результат має той самий розмір і тип uint8 у діапазоні 0..255."""
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(12, 10), dtype=np.uint8)
    result = clahe(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_clahe_grid_larger_than_image() -> None:
    """Кожен піксель у власному блоці не змінюється."""
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(3, 3), dtype=np.uint8)
    result = clahe(image, 2.0, (8, 8))
    np.testing.assert_array_equal(result, image)


def test_clip_histogram_keeps_total() -> None:
    """Розподіл надлишку не змінює суму гістограми."""
    hist = np.zeros(256, dtype=np.float64)
    hist[100] = 10
    hist[101] = 2
    clipped = clip_histogram(hist, 2.0)
    assert clipped.sum() == hist.sum()
    assert clipped.max() <= 2


def test_invalid_image_is_rejected() -> None:
    """Багатоканальне та порожнє зображення не допускаються."""
    with pytest.raises(ValueError):
        clahe(np.zeros((5, 5, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        clahe(np.zeros((0, 5), dtype=np.uint8))


def test_invalid_clip_limit_is_rejected() -> None:
    """Недостатнє обмеження не допускається."""
    with pytest.raises(ValueError):
        clahe(np.zeros((5, 5), dtype=np.uint8), 0.0)


def test_invalid_grid_is_rejected() -> None:
    """Сітка з нульовим або неповним розміром не допускається."""
    image = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        clahe(image, 2.0, (0, 8))
    with pytest.raises(ValueError):
        clahe(image, 2.0, (8,))
