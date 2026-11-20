"""Рисунки зображень для звіту."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cgstem.dataset import ManifestEntry, read_manifest
from cgstem.io.loader import load_image
from cgstem.methods.filtering import gaussian_blur, median_filter
from cgstem.methods.histogram import clahe
from cgstem.methods.morphology import closing, opening
from cgstem.methods.segmentation import otsu_mask
from cgstem.paths import repo_root


def find_entry(entries: list[ManifestEntry], group: str,
               alteration: str) -> ManifestEntry:
    """Знаходить знімок суб'єкта 1 із вказівного пальця."""
    for entry in entries:
        if entry.group != group or entry.alteration != alteration:
            continue
        if entry.subject == 1 and entry.finger == "index":
            return entry
    raise ValueError(f"Не знайдено знімок групи {group} зі зміною {alteration}")


def image_panel(axis: plt.Axes, image: np.ndarray, title: str) -> None:
    """Додає зображення без осей із підписом."""
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    axis.set_title(title, fontsize=10)
    axis.axis("off")


def filtering_figure(root: Path, entries: list[ManifestEntry], path: Path) -> None:
    """Будує порівняння оригіналу з трьома фільтрами."""
    entry = find_entry(entries, "real", "")
    image = load_image(root / entry.path)
    panels = [
        (image, "Оригінал"),
        (median_filter(image, 3), "Медіанний 3x3"),
        (gaussian_blur(image, 5, 1.0), "Гаусів 5x5"),
        (clahe(image, 2.0, (8, 8)), "CLAHE 8x8"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(8, 8))
    for axis, (panel, title) in zip(axes.ravel(), panels):
        image_panel(axis, panel, title)
    figure.tight_layout()
    save_figure(figure, path)


def mask_figure(root: Path, entries: list[ManifestEntry], path: Path) -> None:
    """Будує порівняння маски Оцу з морфологічними операціями."""
    entry = find_entry(entries, "real", "")
    image = load_image(root / entry.path)
    mask = otsu_mask(image)
    panels = [
        (image, "Оригінал"),
        (mask.astype(np.uint8) * 255, "Маска Оцу"),
        (opening(mask).astype(np.uint8) * 255, "Відкриття 3x3"),
        (closing(mask).astype(np.uint8) * 255, "Закриття 3x3"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(8, 8))
    for axis, (panel, title) in zip(axes.ravel(), panels):
        image_panel(axis, panel, title)
    figure.tight_layout()
    save_figure(figure, path)


def altered_figure(root: Path, entries: list[ManifestEntry], path: Path) -> None:
    """Будує порівняння для зміненого знімка."""
    entry = find_entry(entries, "altered", "Obl")
    image = load_image(root / entry.path)
    mask = otsu_mask(image)
    panels = [
        (image, "Оригінал, стирання"),
        (median_filter(image, 3), "Медіанний 3x3"),
        (mask.astype(np.uint8) * 255, "Маска Оцу"),
        (closing(mask).astype(np.uint8) * 255, "Закриття 3x3"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for axis, (panel, title) in zip(axes, panels):
        image_panel(axis, panel, title)
    figure.tight_layout()
    save_figure(figure, path)


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Зберігає рисунок у PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Будує рисунки зображень."""
    plt.switch_backend("Agg")
    root = repo_root()
    entries = read_manifest(root / "data" / "manifest.tsv")
    image_dir = root / "docs" / "report" / "img"
    filtering_figure(root, entries, image_dir / "filtering_comparison.png")
    mask_figure(root, entries, image_dir / "mask_comparison.png")
    altered_figure(root, entries, image_dir / "altered_comparison.png")


if __name__ == "__main__":
    main()
