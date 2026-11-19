"""Тести сегментації методом Оцу."""

from __future__ import annotations

import numpy as np
import pytest

from cgstem.methods.segmentation import otsu_mask, otsu_threshold


def test_otsu_finds_gap_between_two_modes() -> None:
    """Для суміші 0 і 200 поріг дорівнює 0, а маска містить вісім темних пікселів."""
    image = np.array([[0] * 2] * 4 + [[200] * 2] * 4, dtype=np.uint8)
    assert otsu_threshold(image) == 0
    assert otsu_mask(image).sum() == 8


def test_otsu_separates_dark_band() -> None:
    """У зображенні 3x3 із шістьма нулями та трьома 255 передній план містить шість пікселів."""
    image = np.array([[0, 0, 0], [0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    assert otsu_mask(image).sum() == 6


def test_otsu_mask_is_boolean() -> None:
    """Маска має булевий тип і форму оригіналу."""
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(9, 7), dtype=np.uint8)
    result = otsu_mask(image)
    assert result.dtype == np.bool_
    assert result.shape == image.shape


def test_otsu_on_constant_image_gives_empty_mask() -> None:
    """На однорідному світлому зображенні передній план відсутній."""
    image = np.full((5, 5), 255, dtype=np.uint8)
    assert otsu_mask(image).sum() == 0


def test_invalid_image_is_rejected() -> None:
    """Багатоканальне та порожнє зображення не допускаються."""
    with pytest.raises(ValueError):
        otsu_threshold(np.zeros((5, 5, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        otsu_mask(np.zeros((0, 5), dtype=np.uint8))
