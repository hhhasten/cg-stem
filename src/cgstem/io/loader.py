"""Завантаження зображень у градаціях сірого."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from cgstem.types import GrayImage


def load_image(path: Path) -> GrayImage:
    """Завантажує зображення і приводить його до 8 біт у градаціях сірого."""
    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            return np.array(gray, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Не вдалося прочитати зображення: {path}") from error
