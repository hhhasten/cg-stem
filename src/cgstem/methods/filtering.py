"""Фільтрація зображень."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from cgstem.ops.conversion import to_uint8
from cgstem.ops.convolution import convolve2d
from cgstem.ops.kernels import gaussian_kernel
from cgstem.types import GrayImage


def median_filter(image: GrayImage, size: int = 3) -> GrayImage:
    """Прибирає імпульсний шум медіаною у вікні заданого розміру."""
    check_window(size)
    pad = size // 2
    padded = np.pad(image, pad, mode="reflect")
    windows = sliding_window_view(padded, (size, size))
    return np.median(windows, axis=(-2, -1)).astype(np.uint8)


def gaussian_blur(image: GrayImage, size: int = 5,
                  sigma: float = 1.0) -> GrayImage:
    """Згладжує зображення гаусовим ядром."""
    check_window(size)
    if sigma <= 0:
        raise ValueError(f"Сигма повинна бути додатною, отримано {sigma}")
    kernel = gaussian_kernel(size, sigma)
    smoothed = convolve2d(image.astype(np.float64), kernel)
    return to_uint8(smoothed)


def check_window(size: int) -> None:
    """Перевіряє, що розмір вікна непарний і не менший за три."""
    if size < 3 or size % 2 == 0:
        raise ValueError(f"Розмір вікна повинен бути непарним і >= 3, отримано {size}")
