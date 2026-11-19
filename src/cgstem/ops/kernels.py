"""Ядра згортки."""

from __future__ import annotations

import numpy as np

from cgstem.types import FloatImage


def gaussian_kernel(size: int, sigma: float) -> FloatImage:
    """Повертає нормоване гаусове ядро заданого розміру."""
    if size % 2 == 0:
        raise ValueError(f"Розмір ядра повинен бути непарним, отримано {size}")
    if sigma <= 0:
        raise ValueError(f"Сигма повинна бути додатною, отримано {sigma}")
    radius = size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    squared = axis[:, None] ** 2 + axis[None, :] ** 2
    weights = np.exp(-squared / (2 * sigma ** 2))
    return weights / weights.sum()
