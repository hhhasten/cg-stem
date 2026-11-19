"""Гістограмні перетворення."""

from __future__ import annotations

import numpy as np

from cgstem.types import FloatImage, GrayImage

HIST_SIZE = 256


def clahe(image: GrayImage, clip_limit: float = 2.0,
          grid: tuple[int, int] = (8, 8)) -> GrayImage:
    """Вирівнює гістограму локально з обмеженням і білінійним змішуванням."""
    check_params(image, clip_limit, grid)
    rows, cols = actual_grid(image.shape, grid)
    luts, row_edges, col_edges = build_luts(image, rows, cols, clip_limit)
    result = interpolate(image, luts, row_edges, col_edges)
    return np.clip(np.rint(result), 0, HIST_SIZE - 1).astype(np.uint8)


def check_params(image: GrayImage, clip_limit: float,
                 grid: tuple[int, int]) -> None:
    """Перевіряє вимірність зображення, обмеження та розмір сітки."""
    if image.ndim != 2:
        raise ValueError(f"Очікується двовимірне зображення, отримано {image.ndim} виміри")
    if image.size == 0:
        raise ValueError("Зображення не повинно бути порожнім")
    if clip_limit <= 0:
        raise ValueError(f"Обмеження повинно бути додатним, отримано {clip_limit}")
    if len(grid) != 2 or grid[0] < 1 or grid[1] < 1:
        raise ValueError(f"Сітка повинна містити два додатні розміри, отримано {grid}")


def actual_grid(shape: tuple[int, int], grid: tuple[int, int]) -> tuple[int, int]:
    """Обмежує кількість блоків розмірами зображення."""
    rows = min(grid[0], shape[0])
    cols = min(grid[1], shape[1])
    return rows, cols


def tile_edges(length: int, tiles: int) -> np.ndarray:
    """Повертає межі блоків уздовж однієї осі."""
    return np.linspace(0, length, tiles + 1, dtype=int)


def build_luts(image: GrayImage, rows: int, cols: int,
               clip_limit: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Будує таблиці перетворення для кожного блоку сітки."""
    row_edges = tile_edges(image.shape[0], rows)
    col_edges = tile_edges(image.shape[1], cols)
    luts = np.zeros((rows, cols, HIST_SIZE), dtype=np.float64)
    for row in range(rows):
        for col in range(cols):
            tile = image[row_edges[row]:row_edges[row + 1],
                         col_edges[col]:col_edges[col + 1]]
            luts[row, col] = tile_lut(tile, clip_limit)
    return luts, row_edges, col_edges


def tile_lut(tile: GrayImage, clip_limit: float) -> FloatImage:
    """Обчислює вирівняну таблицю перетворення для одного блоку."""
    hist = np.bincount(tile.ravel(), minlength=HIST_SIZE).astype(np.float64)
    if np.count_nonzero(hist) <= 1:
        return np.arange(HIST_SIZE, dtype=np.float64)
    clipped = clip_histogram(hist, clip_limit)
    total = clipped.sum()
    cdf = np.cumsum(clipped)
    cdf_min = cdf[clipped > 0].min()
    denominator = total - cdf_min
    scaled = (cdf - cdf_min) / denominator * (HIST_SIZE - 1)
    return np.clip(scaled, 0, HIST_SIZE - 1)


def clip_histogram(hist: FloatImage, clip_limit: float) -> FloatImage:
    """Обмежує висоту стовпців і розподіляє надлишок по решті кошиків."""
    tile_size = hist.sum()
    threshold = max(int(clip_limit * tile_size / HIST_SIZE), 1)
    clipped = np.minimum(hist, threshold)
    excess = int(hist.sum() - clipped.sum())
    clipped += excess // HIST_SIZE
    clipped[:excess % HIST_SIZE] += 1
    return clipped


def interpolate(image: GrayImage, luts: np.ndarray, row_edges: np.ndarray,
                col_edges: np.ndarray) -> FloatImage:
    """Змішує таблиці сусідніх блоків за відстанями до їхніх центрів."""
    result = np.zeros(image.shape, dtype=np.float64)
    for row_pair in axis_pairs(row_edges):
        for col_pair in axis_pairs(col_edges):
            result += block_contribution(image, luts, row_pair, col_pair)
    return result


def block_contribution(image: GrayImage, luts: np.ndarray,
                       row_pair: tuple[np.ndarray, np.ndarray],
                       col_pair: tuple[np.ndarray, np.ndarray]) -> FloatImage:
    """Повертає внесок одного кута білінійної інтерполяції."""
    row_index, row_factor = row_pair
    col_index, col_factor = col_pair
    pixel_index = image.astype(np.intp)
    sample = luts[row_index[:, None], col_index[None, :], pixel_index]
    return row_factor[:, None] * col_factor[None, :] * sample


def axis_pairs(edges: np.ndarray) -> tuple:
    """Повертає пари сусідніх блоків і ваг для кожної координати."""
    centers = (edges[:-1] + edges[1:] - 1) / 2
    coords = np.arange(edges[-1])
    position = np.interp(coords, centers, np.arange(centers.size))
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, centers.size - 1)
    weight = position - low
    return (low, 1 - weight), (high, weight)
