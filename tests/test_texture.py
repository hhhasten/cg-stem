"""Тести аналізу текстур через GLCM."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cgstem.methods.texture import glcm_features, glcm_matrix, glcm_properties

SAMPLE = np.array([[0, 1], [2, 3]], dtype=np.uint8)


def test_glcm_horizontal_pair_probabilities() -> None:
    """Для зсуву вправо матриця містить пари 0-1 і 2-3 з імовірністю 0.5."""
    matrix = glcm_matrix(SAMPLE, 1, 0)
    assert matrix[0, 1] == pytest.approx(0.5)
    assert matrix[2, 3] == pytest.approx(0.5)
    assert matrix.sum() == pytest.approx(1.0)


def test_glcm_properties_manual_values() -> None:
    """Ручний розрахунок для пар (0,1) і (2,3)."""
    properties = glcm_properties(glcm_matrix(SAMPLE, 1, 0))
    assert properties.contrast == pytest.approx(1.0)
    assert properties.energy == pytest.approx(math.sqrt(0.5))
    assert properties.homogeneity == pytest.approx(0.5)
    assert properties.correlation == pytest.approx(1.0)


def test_glcm_vertical_properties() -> None:
    """Для зсуву вгору пари (2,0) і (3,1) дають контраст 4."""
    properties = glcm_properties(glcm_matrix(SAMPLE, 1, 90))
    assert properties.contrast == pytest.approx(4.0)
    assert properties.homogeneity == pytest.approx(1 / 3)
    assert properties.correlation == pytest.approx(1.0)


def test_glcm_constant_image() -> None:
    """Однорідна текстура має нульовий контраст, одиничну енергію та гомогенність."""
    image = np.full((4, 4), 50, dtype=np.uint8)
    properties = glcm_properties(glcm_matrix(image, 1, 0))
    assert properties.contrast == pytest.approx(0.0)
    assert properties.energy == pytest.approx(1.0)
    assert properties.homogeneity == pytest.approx(1.0)
    assert properties.correlation == pytest.approx(0.0)


def test_features_average_two_angles() -> None:
    """Середнє за двома кутами дорівнює середньому їхніх властивостей."""
    horizontal = glcm_properties(glcm_matrix(SAMPLE, 1, 0))
    vertical = glcm_properties(glcm_matrix(SAMPLE, 1, 90))
    features = glcm_features(SAMPLE, (1,), (0, 90))
    expected = (horizontal.contrast + vertical.contrast) / 2
    assert features["contrast"] == pytest.approx(expected)
    assert set(features) == {"contrast", "energy", "homogeneity", "correlation"}


def test_invalid_parameters_are_rejected() -> None:
    """Невідомий кут, нульова відстань і завелика відстань не допускаються."""
    with pytest.raises(ValueError):
        glcm_matrix(SAMPLE, 1, 30)
    with pytest.raises(ValueError):
        glcm_matrix(SAMPLE, 0, 0)
    with pytest.raises(ValueError):
        glcm_matrix(SAMPLE, 10, 0)


def test_invalid_image_and_matrix_are_rejected() -> None:
    """Багатоканальне зображення та неквадратна матриця не допускаються."""
    with pytest.raises(ValueError):
        glcm_matrix(np.zeros((3, 3, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        glcm_properties(np.zeros((2, 3), dtype=np.float64))
