"""Збереження результатів пайплайна."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from cgstem.dataset import ManifestEntry
from cgstem.io.saver import save_image
from cgstem.pipeline.result import MethodResult
from cgstem.types import GrayImage, Mask

BASE_FIELDS = ["path", "group", "subject", "alteration", "method", "variant"]
METRIC_NAMES = [
    "mse", "psnr", "ssim",
    "foreground", "components", "largest",
    "contrast", "energy", "homogeneity", "correlation",
]
SUMMARY_FIELDS = ["method", "variant", "images"] + METRIC_NAMES


def save_result(out_dir: Path, entry: ManifestEntry, result: MethodResult) -> None:
    """Зберігає вихід методу окремо як фінальний і як результат стадії."""
    if result.output is None:
        return
    source = Path(entry.path)
    relative = Path(result.method) / result.variant / f"{source.stem}{source.suffix.lower()}"
    image = as_uint8_image(result.output)
    save_image(out_dir / "images" / relative, image)
    save_image(out_dir / "stages" / result.stage / relative, image)


def as_uint8_image(image: GrayImage | Mask) -> GrayImage:
    """Приводить зображення або маску до uint8."""
    values = np.asarray(image)
    if values.dtype == np.bool_:
        return values.astype(np.uint8) * 255
    return values.astype(np.uint8, copy=False)


def metric_row(entry: ManifestEntry, result: MethodResult) -> dict:
    """Формує рядок CSV для одного результату."""
    row = {
        "path": entry.path,
        "group": entry.group,
        "subject": entry.subject,
        "alteration": entry.alteration,
        "method": result.method,
        "variant": result.variant,
    }
    for name in METRIC_NAMES:
        row[name] = result.metrics.get(name, "")
    return row


def write_metrics(path: Path, rows: list[dict]) -> None:
    """Записує показники кожного знімка в CSV."""
    write_table(path, BASE_FIELDS + METRIC_NAMES, rows)


def write_summary(path: Path, rows: list[dict]) -> None:
    """Записує усереднені показники за методами в CSV."""
    write_table(path, SUMMARY_FIELDS, rows)


def write_table(path: Path, field_names: list[str], rows: list[dict]) -> None:
    """Записує таблицю з заданими стовпцями в CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    """Усереднює метрики за методом і варіантом."""
    grouped = {}
    for row in rows:
        key = (row["method"], row["variant"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (method, variant), items in grouped.items():
        summary.append(summary_row(method, variant, items))
    return summary


def summary_row(method: str, variant: str, items: list[dict]) -> dict:
    """Формує усереднений рядок для одного методу."""
    result = {"method": method, "variant": variant, "images": len(items)}
    for name in METRIC_NAMES:
        values = [item.get(name, "") for item in items if item.get(name, "") != ""]
        result[name] = sum(values) / len(values) if values else ""
    return result
