"""Аналіз текстур через матрицю суміжності рівнів сірого."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cgstem.types import FloatImage, GrayImage

HIST_SIZE = 256
ANGLE_OFFSETS = {
    0: (0, 1),
    45: (-1, 1),
    90: (-1, 0),
    135: (-1, -1),
}


@dataclass(frozen=True)
class GlcmProperties:
    """Властивості матриці суміжності."""

    contrast: float
    energy: float
    homogeneity: float
    correlation: float


def glcm_matrix(image: GrayImage, distance: int = 1,
                angle: int = 0) -> FloatImage:
    """Повертає нормовану матрицю суміжності для заданих відстані та кута."""
    check_image(image)
    if distance < 1:
        raise ValueError(f"Відстань повинна бути додатною, отримано {distance}")
    offset = angle_offset(angle, distance)
    source, target = offset_slices(image, offset)
    if source.size == 0:
        raise ValueError(f"Відстань {distance} перевищує розміри зображення")
    indices = source.astype(np.intp) * HIST_SIZE + target
    counts = np.bincount(indices.ravel(), minlength=HIST_SIZE ** 2)
    return counts.reshape(HIST_SIZE, HIST_SIZE) / counts.sum()


def angle_offset(angle: int, distance: int) -> tuple[int, int]:
    """Переводить кут у зсув рядка та стовпця."""
    if angle not in ANGLE_OFFSETS:
        raise ValueError(f"Кут повинен бути одним із {sorted(ANGLE_OFFSETS)}, отримано {angle}")
    step_row, step_col = ANGLE_OFFSETS[angle]
    return step_row * distance, step_col * distance


def offset_slices(image: GrayImage,
                  offset: tuple[int, int]) -> tuple[GrayImage, GrayImage]:
    """Повертає пари зрізів джерела та цілі без виходу за межі зображення."""
    step_row, step_col = offset
    rows = image.shape[0]
    cols = image.shape[1]
    source = image[max(0, -step_row):rows - max(0, step_row),
                   max(0, -step_col):cols - max(0, step_col)]
    target = image[max(0, step_row):rows + min(0, step_row),
                   max(0, step_col):cols + min(0, step_col)]
    return source, target


def glcm_properties(matrix: FloatImage) -> GlcmProperties:
    """Обчислює контраст, енергію, гомогенність і кореляцію матриці."""
    check_matrix(matrix)
    levels = np.arange(matrix.shape[0], dtype=np.float64)
    rows, cols = np.meshgrid(levels, levels, indexing="ij")
    contrast = np.sum((rows - cols) ** 2 * matrix)
    energy = np.sqrt(np.sum(matrix ** 2))
    homogeneity = np.sum(matrix / (1 + np.abs(rows - cols)))
    correlation = compute_correlation(matrix, levels)
    return GlcmProperties(
        contrast=float(contrast),
        energy=float(energy),
        homogeneity=float(homogeneity),
        correlation=correlation,
    )


def compute_correlation(matrix: FloatImage, levels: FloatImage) -> float:
    """Обчислює кореляцію рівнів сірого за матрицею суміжності."""
    row_mass = matrix.sum(axis=1)
    col_mass = matrix.sum(axis=0)
    mean_row = np.sum(row_mass * levels)
    mean_col = np.sum(col_mass * levels)
    row_deviation = levels - mean_row
    col_deviation = levels - mean_col
    covariance = np.sum(matrix * row_deviation[:, None] * col_deviation[None, :])
    row_variance = np.sum(row_mass * row_deviation ** 2)
    col_variance = np.sum(col_mass * col_deviation ** 2)
    denominator = np.sqrt(row_variance * col_variance)
    if denominator == 0:
        return 0.0
    return float(covariance / denominator)


def glcm_features(image: GrayImage, distances: tuple[int, ...] = (1, 2, 3),
                  angles: tuple[int, ...] = (0, 45, 90, 135)) -> dict[str, float]:
    """Усереднює властивості GLCM за всіма відстанями та кутами."""
    if not distances or not angles:
        raise ValueError("Потрібна хоча б одна відстань і один кут")
    totals = {"contrast": 0.0, "energy": 0.0,
              "homogeneity": 0.0, "correlation": 0.0}
    count = 0
    for distance in distances:
        for angle in angles:
            properties = glcm_properties(glcm_matrix(image, distance, angle))
            totals["contrast"] += properties.contrast
            totals["energy"] += properties.energy
            totals["homogeneity"] += properties.homogeneity
            totals["correlation"] += properties.correlation
            count += 1
    return {name: value / count for name, value in totals.items()}


def check_image(image: GrayImage) -> None:
    """Перевіряє, що зображення двовимірне і не порожнє."""
    if image.ndim != 2:
        raise ValueError(f"Очікується двовимірне зображення, отримано {image.ndim} виміри")
    if image.size == 0:
        raise ValueError("Зображення не повинно бути порожнім")


def check_matrix(matrix: FloatImage) -> None:
    """Перевіряє, що матриця квадратна і не порожня."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Очікується квадратна матриця, отримано {matrix.shape}")
    if matrix.size == 0:
        raise ValueError("Матриця не повинна бути порожньою")
