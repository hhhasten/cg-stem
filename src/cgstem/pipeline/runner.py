"""Запуск пайплайна по вибірці."""

from __future__ import annotations

from pathlib import Path

from cgstem.dataset import ManifestEntry
from cgstem.io.loader import load_image
from cgstem.pipeline.stages import process_image
from cgstem.pipeline.storage import metric_row, save_result


def run_pipeline(root: Path, entries: list[ManifestEntry], config: dict,
                 out_dir: Path, limit: int = 0) -> list[dict]:
    """Обробляє вибірку, зберігає виходи та повертає рядки метрик."""
    selected = entries[:limit] if limit > 0 else entries
    rows = []
    for entry in selected:
        image = load_image(root / entry.path)
        for result in process_image(image, config):
            save_result(out_dir, entry, result)
            rows.append(metric_row(entry, result))
    return rows
