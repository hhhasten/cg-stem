"""Індекс структурної подібності."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cgstem.metrics.common import check_same_shape, to_float
from cgstem.ops.convolution import convolve2d
from cgstem.ops.kernels import gaussian_kernel
from cgstem.types import FloatImage, GrayImage

WINDOW_SIZE = 11
WINDOW_SIGMA = 1.5
CONSTANT_1 = 0.01
CONSTANT_2 = 0.03


@dataclass(frozen=True)
class LocalStatistics:
    """Локальні характеристики двох зображень у кожній точці."""

    mean_first: FloatImage
    mean_second: FloatImage
    variance_first: FloatImage
    variance_second: FloatImage
    covariance: FloatImage


def compute_ssim(original: GrayImage, processed: GrayImage,
                 data_range: int = 255) -> float:
    """Обчислює середній індекс структурної подібності."""
    check_same_shape(original, processed)
    check_size(original.shape)
    window = gaussian_kernel(WINDOW_SIZE, WINDOW_SIGMA)
    statistics = local_statistics(to_float(original), to_float(processed), window)
    similarity = similarity_map(statistics, data_range)
    return float(np.mean(crop_borders(similarity)))


def local_statistics(first: FloatImage, second: FloatImage,
                     window: FloatImage) -> LocalStatistics:
    """Обчислює локальні середні, дисперсії та коваріацію."""
    mean_first = convolve2d(first, window)
    mean_second = convolve2d(second, window)
    variance_first = convolve2d(first * first, window) - mean_first ** 2
    variance_second = convolve2d(second * second, window) - mean_second ** 2
    covariance = convolve2d(first * second, window) - mean_first * mean_second
    return LocalStatistics(
        mean_first=mean_first,
        mean_second=mean_second,
        variance_first=variance_first,
        variance_second=variance_second,
        covariance=covariance,
    )


def similarity_map(statistics: LocalStatistics, data_range: int) -> FloatImage:
    """Обчислює карту подібності за формулою SSIM."""
    first_constant = (CONSTANT_1 * data_range) ** 2
    second_constant = (CONSTANT_2 * data_range) ** 2
    numerator = (
        2 * statistics.mean_first * statistics.mean_second + first_constant
    ) * (2 * statistics.covariance + second_constant)
    denominator = (
        statistics.mean_first ** 2
        + statistics.mean_second ** 2
        + first_constant
    ) * (
        statistics.variance_first
        + statistics.variance_second
        + second_constant
    )
    return numerator / denominator


def crop_borders(similarity: FloatImage) -> FloatImage:
    """Відкидає краї карти подібності, де вікно виходить за межі зображення."""
    pad = WINDOW_SIZE // 2
    return similarity[pad:-pad, pad:-pad]


def check_size(shape: tuple[int, ...]) -> None:
    """Перевіряє, що зображення вміщує вікно повністю."""
    if shape[0] < WINDOW_SIZE or shape[1] < WINDOW_SIZE:
        raise ValueError(
            f"Для SSIM потрібне зображення не менше {WINDOW_SIZE}x{WINDOW_SIZE}, "
            f"отримано {shape[0]}x{shape[1]}"
        )
