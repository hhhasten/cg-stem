"""Тести пайплайна та збереження результатів."""

from __future__ import annotations

import csv

import numpy as np

from cgstem.dataset import ManifestEntry
from cgstem.io.saver import save_image
from cgstem.pipeline.runner import run_pipeline
from cgstem.pipeline.storage import summarize, write_metrics, write_summary

CONFIG = {
    "median": {"sizes": [3]},
    "gaussian": {"size": 3, "sigma": 1.0},
    "clahe": {"clip_limit": 2.0, "grid": [2, 2]},
    "otsu": {},
    "morphology": {"size": 3},
    "glcm": {"distances": [1], "angles": [0]},
}


def make_entry(path: str) -> ManifestEntry:
    """Створює запис маніфесту для тестового знімка."""
    return ManifestEntry(
        path=path,
        group="real",
        subject=1,
        hand="Left",
        finger="index",
        alteration="",
        sha256="",
    )


def test_pipeline_writes_outputs_and_metrics(tmp_path) -> None:
    """Пайплайн створює сім результатів, файли стадій і рядки метрик."""
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
    save_image(tmp_path / "img.bmp", image)
    out_dir = tmp_path / "out"
    rows = run_pipeline(tmp_path, [make_entry("img.bmp")], CONFIG, out_dir)
    assert [row["method"] for row in rows] == [
        "median", "gaussian", "clahe", "otsu", "opening", "closing", "glcm",
    ]
    assert rows[0]["psnr"] != ""
    assert rows[3]["foreground"] != ""
    assert rows[6]["contrast"] != ""
    assert (out_dir / "images" / "median" / "size3" / "img.bmp").exists()
    assert (out_dir / "stages" / "filtering" / "median" / "size3" / "img.bmp").exists()
    assert (out_dir / "stages" / "segmentation" / "otsu" / "default" / "img.bmp").exists()
    assert (out_dir / "stages" / "morphology" / "opening" / "size3" / "img.bmp").exists()
    assert not (out_dir / "images" / "glcm").exists()


def test_summary_averages_rows() -> None:
    """Зведення усереднює метрики за методом і варіантом."""
    rows = [
        {"method": "median", "variant": "size3", "psnr": 10.0, "ssim": 0.5},
        {"method": "median", "variant": "size3", "psnr": 20.0, "ssim": 0.7},
    ]
    summary = summarize(rows)
    assert summary[0]["images"] == 2
    assert summary[0]["psnr"] == 15.0
    assert summary[0]["ssim"] == 0.6


def test_metrics_and_summary_files_are_csv(tmp_path) -> None:
    """Таблиці метрик і зведення записуються з заголовками."""
    rows = [{"path": "img.bmp", "group": "real", "subject": 1, "alteration": "",
             "method": "otsu", "variant": "default", "mse": "", "psnr": "",
             "ssim": "", "foreground": 0.5, "components": 2, "largest": 40,
             "contrast": "", "energy": "", "homogeneity": "", "correlation": ""}]
    write_metrics(tmp_path / "metrics.csv", rows)
    write_summary(tmp_path / "summary.csv", summarize(rows))
    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert metric_rows[0]["method"] == "otsu"
    assert summary_rows[0]["images"] == "1"
    assert summary_rows[0]["foreground"] == "0.5"
