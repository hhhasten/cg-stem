"""Спільні типи для роботи із зображеннями.

Image: кольорове зображення, форма (H, W, 3), uint8.
GrayImage: напівтонове зображення, форма (H, W), uint8.
FloatImage: проміжні обчислення, форма (H, W), float64, діапазон 0.0..1.0.
Mask: бінарна маска, форма (H, W), bool.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
GrayImage = NDArray[np.uint8]
FloatImage = NDArray[np.float64]
Mask = NDArray[np.bool_]
