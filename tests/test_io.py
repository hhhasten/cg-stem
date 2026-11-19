"""Тести читання та збереження зображень."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cgstem.io.loader import load_image
from cgstem.io.saver import save_image


def test_rgba_image_is_converted_to_gray(tmp_path: Path) -> None:
    """Зображення з альфа-каналом перетворюється у двовимірний масив."""
    path = tmp_path / "rgba.bmp"
    Image.new("RGBA", (4, 5), (200, 100, 50, 255)).save(path)
    image = load_image(path)
    assert image.shape == (5, 4)
    assert image.dtype == np.uint8
    assert image[0, 0] == pytest.approx(124, abs=2)


def test_missing_file_raises_value_error(tmp_path: Path) -> None:
    """Відсутній файл дає зрозумілу помилку."""
    with pytest.raises(ValueError):
        load_image(tmp_path / "missing.bmp")


def test_corrupted_file_raises_value_error(tmp_path: Path) -> None:
    """Пошкоджений файл дає зрозумілу помилку."""
    path = tmp_path / "broken.bmp"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_image(path)


def test_saved_image_is_read_back_unchanged(tmp_path: Path) -> None:
    """Збережене зображення читається без змін."""
    image = np.arange(9, dtype=np.uint8).reshape(3, 3) * 10
    path = tmp_path / "nested" / "result.png"
    save_image(path, image)
    restored = load_image(path)
    np.testing.assert_array_equal(restored, image)
