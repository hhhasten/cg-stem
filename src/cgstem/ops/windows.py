"""Перевірка розміру ковзного вікна."""

from __future__ import annotations


def check_window(size: int) -> None:
    """Перевіряє, що розмір вікна непарний і не менший за три."""
    if size < 3 or size % 2 == 0:
        raise ValueError(f"Розмір вікна повинен бути непарним і >= 3, отримано {size}")
