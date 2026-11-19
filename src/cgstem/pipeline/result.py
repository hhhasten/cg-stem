"""Результат застосування одного методу."""

from __future__ import annotations

from dataclasses import dataclass

from cgstem.types import GrayImage, Mask


@dataclass(frozen=True)
class MethodResult:
    """Містить вихід методу, його стадію та метрики."""

    method: str
    variant: str
    stage: str
    output: GrayImage | Mask | None
    metrics: dict[str, float]
