"""Тести морфології бінарних масок."""

from __future__ import annotations

import numpy as np
import pytest

from cgstem.methods.morphology import closing, dilate, erode, opening


def test_erode_removes_isolated_pixel() -> None:
    """Поодинокий піксель зникає після ерозії 3x3."""
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    assert not erode(mask, 3).any()


def test_erode_keeps_solid_mask() -> None:
    """Суцільна маска не звужується завдяки повторенню країв."""
    mask = np.ones((5, 5), dtype=bool)
    np.testing.assert_array_equal(erode(mask, 3), mask)


def test_dilate_grows_single_pixel_to_window() -> None:
    """Дилатація 3x3 перетворює один піксель на дев'ять."""
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    result = dilate(mask, 3)
    assert result.sum() == 9
    assert result[1:4, 1:4].all()


def test_opening_removes_isolated_pixel() -> None:
    """Відкриття прибирає об'єкт, менший за структурний елемент."""
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    assert not opening(mask, 3).any()


def test_closing_fills_single_hole() -> None:
    """Закриття заповнює отвір розміром в один піксель."""
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    np.testing.assert_array_equal(closing(mask, 3), np.ones((5, 5), dtype=bool))


def test_opening_keeps_solid_square() -> None:
    """Квадрат, більший за вікно, не змінюється."""
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    np.testing.assert_array_equal(opening(mask, 3), mask)


def test_even_and_small_window_are_rejected() -> None:
    """Парне вікно та вікно менше трьох не допускаються."""
    mask = np.ones((5, 5), dtype=bool)
    with pytest.raises(ValueError):
        erode(mask, 4)
    with pytest.raises(ValueError):
        dilate(mask, 1)
