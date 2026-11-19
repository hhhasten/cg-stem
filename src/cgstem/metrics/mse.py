"""Середньоквадратична похибка."""

from __future__ import annotations

import numpy as np

from cgstem.metrics.common import check_same_shape, to_float
from cgstem.types import GrayImage


def compute_mse(original: GrayImage, processed: GrayImage) -> float:
    """Обчислює середньоквадратичну похибку між двома зображеннями."""
    check_same_shape(original, processed)
    difference = to_float(original) - to_float(processed)
    return float(np.mean(difference ** 2))
