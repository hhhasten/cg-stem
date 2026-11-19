"""Конфігурація пайплайна."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

ALLOWED_METHODS = ("median", "gaussian", "clahe", "otsu", "morphology", "glcm")


def load_config(path: Path) -> dict:
    """Читає YAML-конфігурацію методів і перевіряє назви розділів."""
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Конфігурація повинна бути словником: {path}")
    unknown = set(config) - set(ALLOWED_METHODS)
    if unknown:
        raise ValueError(f"Невідомі методи в конфігурації: {sorted(unknown)}")
    return config


def apply_overrides(config: dict, assignments: list[str]) -> dict:
    """Перекриває параметри значеннями виду метод.параметр=значення."""
    updated = copy.deepcopy(config)
    for assignment in assignments:
        method, parameter, value = split_assignment(assignment)
        if method not in updated:
            raise ValueError(f"Невідомий метод у перекритті: {method}")
        updated[method][parameter] = yaml.safe_load(value)
    return updated


def split_assignment(assignment: str) -> tuple[str, str, str]:
    """Розбирає перекриття на метод, параметр і значення."""
    if "=" not in assignment or "." not in assignment.split("=", 1)[0]:
        raise ValueError(f"Очікується метод.параметр=значення, отримано {assignment}")
    key, value = assignment.split("=", 1)
    method, parameter = key.split(".", 1)
    return method, parameter, value


def select_methods(config: dict, names: list[str]) -> dict:
    """Лишає в конфігурації тільки вибрані методи."""
    if not names:
        return config
    unknown = set(names) - set(config)
    if unknown:
        raise ValueError(f"Невідомі вибрані методи: {sorted(unknown)}")
    return {name: value for name, value in config.items() if name in names}
