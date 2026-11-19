"""Спільні перевірки для метрик якості."""

from __future__ import annotations

import numpy as np

from cgstem.types import FloatImage, GrayImage


def check_same_shape(original: GrayImage, processed: GrayImage) -> None:
    """Перевіряє, що зображення мають однакову форму."""
    if original.shape != processed.shape:
        raise ValueError(
            f"Різні розміри зображень: {original.shape} і {processed.shape}"
        )


def to_float(image: GrayImage) -> FloatImage:
    """Переводить зображення у формат float64."""
    return image.astype(np.float64)
