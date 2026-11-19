"""Збереження зображень на диск."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as PilImage

from cgstem.types import Image, GrayImage


def save_image(path: Path, image: Image | GrayImage) -> None:
    """Зберігає зображення, створюючи потрібні каталоги."""
    path.parent.mkdir(parents=True, exist_ok=True)
    PilImage.fromarray(np.asarray(image)).save(path)
