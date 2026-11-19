"""Двовимірна згортка."""

from __future__ import annotations

import numpy as np

from cgstem.types import FloatImage


def convolve2d(image: FloatImage, kernel: FloatImage,
               mode: str = "reflect") -> FloatImage:
    """Виконує двовимірну згортку з доповненням країв."""
    check_inputs(image, kernel)
    rows, cols = kernel.shape
    pad_rows = rows // 2
    pad_cols = cols // 2
    padded = np.pad(image, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode=mode)
    height, width = image.shape
    result = np.zeros_like(image, dtype=np.float64)
    for row in range(rows):
        for col in range(cols):
            weight = kernel[rows - 1 - row, cols - 1 - col]
            window = padded[row:row + height, col:col + width]
            result += weight * window
    return result


def check_inputs(image: FloatImage, kernel: FloatImage) -> None:
    """Перевіряє вимірність і розміри ядра."""
    if image.ndim != 2:
        raise ValueError(f"Очікується двовимірне зображення, отримано {image.ndim} виміри")
    if kernel.ndim != 2:
        raise ValueError(f"Очікується двовимірне ядро, отримано {kernel.ndim} виміри")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError(f"Ядро повинно мати непарні розміри, отримано {kernel.shape}")
