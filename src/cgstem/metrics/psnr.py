"""Пікове відношення сигналу до шуму."""

from __future__ import annotations

import math

from cgstem.metrics.mse import compute_mse
from cgstem.types import GrayImage


def compute_psnr(original: GrayImage, processed: GrayImage,
                 data_range: int = 255) -> float:
    """Обчислює PSNR у децибелах, для однакових зображень повертає нескінченність."""
    mse = compute_mse(original, processed)
    if mse == 0:
        return math.inf
    return 10 * math.log10(data_range ** 2 / mse)
