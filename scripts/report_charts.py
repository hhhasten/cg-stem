"""Діаграми метрик для звіту."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cgstem.paths import repo_root

FILTER_LABELS = {
    ("median", "size3"): "Медіанний 3x3",
    ("median", "size5"): "Медіанний 5x5",
    ("median", "size7"): "Медіанний 7x7",
    ("gaussian", "size5_sigma1.0"): "Гаусів 5x5",
    ("clahe", "clip2.0_grid8x8"): "CLAHE 8x8",
}
MASK_LABELS = {
    "otsu": "Оцу",
    "opening": "Відкриття",
    "closing": "Закриття",
}


def read_summary(path: Path) -> list[dict]:
    """Читає зведення метрик із CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metrics_figure(summary_path: Path, path: Path) -> None:
    """Будує діаграми PSNR і SSIM для фільтрів."""
    rows = [row for row in read_summary(summary_path)
            if (row["method"], row["variant"]) in FILTER_LABELS]
    labels = [FILTER_LABELS[(row["method"], row["variant"])] for row in rows]
    psnr = [float(row["psnr"]) for row in rows]
    ssim = [float(row["ssim"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    bar_panel(axes[0], labels, psnr, "PSNR, дБ", "{:.2f}")
    bar_panel(axes[1], labels, ssim, "SSIM", "{:.3f}")
    figure.tight_layout()
    save_figure(figure, path)


def bar_panel(axis: plt.Axes, labels: list[str], values: list[float],
              title: str, pattern: str) -> None:
    """Малює горизонтальну діаграму з підписаними значеннями."""
    positions = np.arange(len(labels))
    axis.barh(positions, values, color="#3d6f8e", height=0.6)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlim(0, max(values) * 1.18)
    for position, value in zip(positions, values):
        axis.text(value * 1.02, position, pattern.format(value), va="center", fontsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#d9d9d9", linewidth=0.7)
    axis.set_axisbelow(True)


def mask_figure(summary_path: Path, path: Path) -> None:
    """Будує діаграми показників бінарних масок."""
    rows = [row for row in read_summary(summary_path)
            if row["method"] in MASK_LABELS]
    labels = [MASK_LABELS[row["method"]] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    chart_panel(axes[0], labels, [float(row["foreground"]) for row in rows],
                "Частка переднього плану", "{:.3f}")
    chart_panel(axes[1], labels, [float(row["components"]) for row in rows],
                "Кількість компонент", "{:.1f}")
    chart_panel(axes[2], labels, [float(row["largest"]) for row in rows],
                "Площа найбільшої", "{:.0f}")
    figure.tight_layout()
    save_figure(figure, path)


def chart_panel(axis: plt.Axes, labels: list[str], values: list[float],
                title: str, pattern: str) -> None:
    """Малює стовпчикову діаграму для одного показника."""
    positions = np.arange(len(labels))
    axis.bar(positions, values, color="#6d8f6a", width=0.55)
    axis.set_xticks(positions, labels)
    axis.set_title(title, fontsize=10)
    axis.set_ylim(0, max(values) * 1.22)
    for position, value in zip(positions, values):
        axis.text(position, value * 1.03, pattern.format(value),
                  ha="center", fontsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axis.set_axisbelow(True)


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Зберігає діаграму у PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Будує діаграми метрик."""
    plt.switch_backend("Agg")
    root = repo_root()
    image_dir = root / "docs" / "report" / "img"
    metrics_dir = root / "out" / "metrics"
    metrics_figure(metrics_dir / "summary.csv", image_dir / "metrics_chart.png")
    mask_figure(metrics_dir / "summary.csv", image_dir / "mask_chart.png")


if __name__ == "__main__":
    main()
