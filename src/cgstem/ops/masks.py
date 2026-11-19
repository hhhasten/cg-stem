"""Перевірка бінарних масок."""

from __future__ import annotations

from cgstem.types import Mask


def check_mask(mask: Mask) -> None:
    """Перевіряє, що маска двовимірна і не порожня."""
    if mask.ndim != 2:
        raise ValueError(f"Очікується двовимірна маска, отримано {mask.ndim} виміри")
    if mask.size == 0:
        raise ValueError("Маска не повинна бути порожньою")
