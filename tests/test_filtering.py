"""Тести фільтрації зображень."""

from __future__ import annotations

import numpy as np
import pytest

from cgstem.methods.filtering import gaussian_blur, median_filter


def test_median_of_ordered_window_is_middle_value() -> None:
    """Медіана вікна 3x3 зі значень 1..9 дорівнює 5."""
    image = np.arange(1, 10, dtype=np.uint8).reshape(3, 3)
    result = median_filter(image, 3)
    assert result[1, 1] == 5


def test_median_removes_single_impulse() -> None:
    """Одиничний світлий піксель замінюється медіаною сусідів."""
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 255
    result = median_filter(image, 3)
    assert result[2, 2] == 0


def test_median_keeps_step_edge() -> None:
    """Межа між темною та світлою половинами не розмивається."""
    image = np.zeros((5, 5), dtype=np.uint8)
    image[:, 2:] = 255
    result = median_filter(image, 3)
    expected = image[1:-1, 1:-1]
    np.testing.assert_array_equal(result[1:-1, 1:-1], expected)


def test_gaussian_keeps_constant_image() -> None:
    """Однорідне зображення не змінюється."""
    image = np.full((9, 9), 120, dtype=np.uint8)
    result = gaussian_blur(image, 5, 1.0)
    np.testing.assert_array_equal(result, image)


def test_gaussian_spreads_impulse() -> None:
    """Гаусове ядро 3x3 розподіляє поодинокий піксель на сусідні."""
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 255
    result = gaussian_blur(image, 3, 1.0)
    assert result[2, 2] == pytest.approx(52, abs=1)
    assert result[2, 1] > 0
    assert result[2, 1] < result[2, 2]


def test_gaussian_does_not_exceed_range() -> None:
    """Після згладжування значення не виходять за межі 0..255."""
    image = np.full((7, 7), 255, dtype=np.uint8)
    result = gaussian_blur(image, 5, 3.0)
    assert result.max() <= 255
    assert result.min() >= 0


def test_even_window_is_rejected() -> None:
    """Парний розмір вікна не допускається."""
    image = np.zeros((7, 7), dtype=np.uint8)
    with pytest.raises(ValueError):
        median_filter(image, 4)
    with pytest.raises(ValueError):
        gaussian_blur(image, 4, 1.0)


def test_too_small_window_is_rejected() -> None:
    """Вікно менше трьох не допускається."""
    image = np.zeros((7, 7), dtype=np.uint8)
    with pytest.raises(ValueError):
        median_filter(image, 1)


def test_non_positive_sigma_is_rejected() -> None:
    """Недостатня сигма не допускається."""
    image = np.zeros((7, 7), dtype=np.uint8)
    with pytest.raises(ValueError):
        gaussian_blur(image, 3, 0.0)
