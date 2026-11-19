"""Морфологія бінарних масок."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from cgstem.ops.masks import check_mask
from cgstem.ops.windows import check_window
from cgstem.types import Mask


def erode(mask: Mask, size: int = 3) -> Mask:
    """Звужує маску: піксель лишається, коли все вікно належить масці."""
    check_window(size)
    windows = mask_windows(mask, size)
    return np.all(windows, axis=(-2, -1))


def dilate(mask: Mask, size: int = 3) -> Mask:
    """Розширює маску: піксель додається, коли вікно зачіпає маску."""
    check_window(size)
    windows = mask_windows(mask, size)
    return np.any(windows, axis=(-2, -1))


def opening(mask: Mask, size: int = 3) -> Mask:
    """Прибирає дрібні об'єкти: ерозія, потім дилатація."""
    return dilate(erode(mask, size), size)


def closing(mask: Mask, size: int = 3) -> Mask:
    """Заповнює дрібні отвори: дилатація, потім ерозія."""
    return erode(dilate(mask, size), size)


def mask_windows(mask: Mask, size: int) -> np.ndarray:
    """Повертає ковзні вікна маски з доповненням країв."""
    check_mask(mask)
    pad = size // 2
    padded = np.pad(mask, pad, mode="edge")
    return sliding_window_view(padded, (size, size))
