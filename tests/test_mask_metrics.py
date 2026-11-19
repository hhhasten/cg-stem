"""Тести показників бінарних масок."""

from __future__ import annotations

import numpy as np
import pytest

from cgstem.metrics.mask import (
    count_components,
    foreground_fraction,
    largest_component_area,
)


def test_foreground_fraction() -> None:
    """Половина пікселів переднього плану дає 0.5."""
    mask = np.array([[True, False], [False, True]])
    assert foreground_fraction(mask) == pytest.approx(0.5)


def test_two_separate_blocks() -> None:
    """Два розділені квадрати 2x2 дають дві компоненти площею 4."""
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:2, 0:2] = True
    mask[4:6, 4:6] = True
    assert count_components(mask) == 2
    assert largest_component_area(mask) == 4


def test_diagonal_pixels_are_connected() -> None:
    """Діагональні пікселі утворюють одну компоненту за восьмизв'язністю."""
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    assert count_components(mask) == 1
    assert largest_component_area(mask) == 2


def test_empty_mask_has_no_components() -> None:
    """Порожня маска не має компонент."""
    mask = np.zeros((5, 5), dtype=bool)
    assert count_components(mask) == 0
    assert largest_component_area(mask) == 0


def test_invalid_mask_is_rejected() -> None:
    """Одновимірна та порожня маска не допускаються."""
    with pytest.raises(ValueError):
        count_components(np.zeros(5, dtype=bool))
    with pytest.raises(ValueError):
        foreground_fraction(np.zeros((0, 5), dtype=bool))
