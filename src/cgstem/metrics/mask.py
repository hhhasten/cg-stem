"""Показники бінарних масок."""

from __future__ import annotations

import numpy as np

from cgstem.ops.masks import check_mask
from cgstem.types import Mask

NEIGHBORS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def foreground_fraction(mask: Mask) -> float:
    """Повертає частку пікселів переднього плану."""
    check_mask(mask)
    return float(np.mean(mask))


def count_components(mask: Mask) -> int:
    """Повертає кількість зв'язаних компонент за восьмизв'язністю."""
    return len(component_sizes(mask))


def largest_component_area(mask: Mask) -> int:
    """Повертає площу найбільшої зв'язаної компоненти."""
    sizes = component_sizes(mask)
    return max(sizes, default=0)


def component_sizes(mask: Mask) -> list[int]:
    """Повертає площі всіх зв'язаних компонент маски."""
    check_mask(mask)
    visited = np.zeros_like(mask, dtype=bool)
    sizes = []
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if not mask[row, col] or visited[row, col]:
                continue
            sizes.append(flood_size(mask, visited, row, col))
    return sizes


def flood_size(mask: Mask, visited: Mask, row: int, col: int) -> int:
    """Обходить одну компоненту в глибину і повертає її площу."""
    size = 0
    visited[row, col] = True
    stack = [(row, col)]
    while stack:
        current_row, current_col = stack.pop()
        size += 1
        for offset_row, offset_col in NEIGHBORS:
            next_row = current_row + offset_row
            next_col = current_col + offset_col
            if not inside(mask, next_row, next_col):
                continue
            if not mask[next_row, next_col] or visited[next_row, next_col]:
                continue
            visited[next_row, next_col] = True
            stack.append((next_row, next_col))
    return size


def inside(mask: Mask, row: int, col: int) -> bool:
    """Перевіряє, що координати належать масці."""
    return 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]
