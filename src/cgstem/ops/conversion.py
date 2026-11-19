"""Перетворення типів зображень."""

from __future__ import annotations

import numpy as np

from cgstem.types import FloatImage, GrayImage


def to_uint8(image: FloatImage) -> GrayImage:
    """Обмежує значення діапазоном 0..255 і переводить у uint8."""
    rounded = np.rint(image)
    return np.clip(rounded, 0, 255).astype(np.uint8)
