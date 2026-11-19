"""Тести конфігурації пайплайна."""

from __future__ import annotations

import pytest

from cgstem.pipeline.config import (
    apply_overrides,
    load_config,
    select_methods,
    split_assignment,
)

CONFIG_TEXT = """median:
  sizes: [3, 5]
otsu: {}
"""


def test_load_config(tmp_path) -> None:
    """Читає дозволені розділи YAML."""
    path = tmp_path / "methods.yaml"
    path.write_text(CONFIG_TEXT, encoding="utf-8")
    config = load_config(path)
    assert config["median"]["sizes"] == [3, 5]
    assert "otsu" in config


def test_unknown_method_is_rejected(tmp_path) -> None:
    """Невідомий розділ конфігурації не допускається."""
    path = tmp_path / "methods.yaml"
    path.write_text("unknown:\n  value: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_overrides_replace_nested_value(tmp_path) -> None:
    """Перекриття замінює вкладене значення."""
    path = tmp_path / "methods.yaml"
    path.write_text(CONFIG_TEXT, encoding="utf-8")
    config = apply_overrides(load_config(path), ["median.sizes=[7]"])
    assert config["median"]["sizes"] == [7]


def test_invalid_assignment_is_rejected(tmp_path) -> None:
    """Перекриття без методу або значення не допускається."""
    path = tmp_path / "methods.yaml"
    path.write_text(CONFIG_TEXT, encoding="utf-8")
    config = load_config(path)
    with pytest.raises(ValueError):
        apply_overrides(config, ["median.sizes"])
    with pytest.raises(ValueError):
        split_assignment("median.sizes")


def test_select_methods(tmp_path) -> None:
    """Вибір лишає тільки вказані розділи."""
    path = tmp_path / "methods.yaml"
    path.write_text(CONFIG_TEXT, encoding="utf-8")
    config = load_config(path)
    selected = select_methods(config, ["otsu"])
    assert list(selected) == ["otsu"]
    with pytest.raises(ValueError):
        select_methods(config, ["unknown"])
