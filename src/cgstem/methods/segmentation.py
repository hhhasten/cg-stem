"""Сегментація зображень."""

from __future__ import annotations

import numpy as np

from cgstem.types import GrayImage, Mask

HIST_SIZE = 256


def otsu_threshold(image: GrayImage) -> int:
    """Обчислює поріг Оцу, що максимізує міжкласову дисперсію."""
    check_image(image)
    hist = np.bincount(image.ravel(), minlength=HIST_SIZE).astype(np.float64)
    levels = np.arange(HIST_SIZE, dtype=np.float64)
    total = hist.sum()
    weight_back = np.cumsum(hist)
    weight_front = total - weight_back
    sum_back = np.cumsum(hist * levels)
    sum_front = sum_back[-1] - sum_back
    valid = (weight_back > 0) & (weight_front > 0)
    mean_back = np.divide(sum_back, weight_back, out=np.zeros_like(sum_back), where=valid)
    mean_front = np.divide(sum_front, weight_front, out=np.zeros_like(sum_front), where=valid)
    variance = weight_back * weight_front * (mean_back - mean_front) ** 2
    variance = np.where(valid, variance, 0.0)
    return int(np.argmax(variance))


def otsu_mask(image: GrayImage) -> Mask:
    """Виділяє темні піксели, що не перевищують поріг Оцу."""
    threshold = otsu_threshold(image)
    return image <= threshold


def check_image(image: GrayImage) -> None:
    """Перевіряє, що зображення двовимірне і не порожнє."""
    if image.ndim != 2:
        raise ValueError(f"Очікується двовимірне зображення, отримано {image.ndim} виміри")
    if image.size == 0:
        raise ValueError("Зображення не повинно бути порожнім")
