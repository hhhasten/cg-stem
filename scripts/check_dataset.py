"""Перевірка вибірки зображень у data/raw."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def find_images(root: Path) -> list[Path]:
    """Повертає відсортований перелік файлів зображень у каталозі."""
    return sorted(
        path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )


def describe(path: Path) -> tuple[str, str]:
    """Повертає розмір і колірний режим зображення."""
    with Image.open(path) as image:
        return f"{image.width}x{image.height}", image.mode


def collect(images: list[Path]) -> tuple[Counter, Counter, list[str]]:
    """Збирає статистику за розмірами, режимами та пошкодженими файлами."""
    sizes = Counter()
    modes = Counter()
    broken = []
    for path in images:
        try:
            size, mode = describe(path)
        except (UnidentifiedImageError, OSError):
            broken.append(path.name)
            continue
        sizes[size] += 1
        modes[mode] += 1
    return sizes, modes, broken


def report(root: Path, images: list[Path], sizes: Counter, modes: Counter,
           broken: list[str]) -> None:
    """Друкує зведення по вибірці."""
    print(f"Каталог: {root}")
    print(f"Зображень: {len(images)}")
    print(f"Пошкоджених: {len(broken)}")
    for name in broken:
        print(f"  {name}")
    print("Розміри:")
    for size, count in sizes.most_common():
        print(f"  {size}: {count}")
    print("Режими:")
    for mode, count in modes.most_common():
        print(f"  {mode}: {count}")


def main() -> None:
    """Запускає перевірку вибірки."""
    root = Path(__file__).resolve().parents[1] / "data" / "raw"
    images = find_images(root)
    if not images:
        print(f"Зображень не знайдено: {root}")
        return
    sizes, modes, broken = collect(images)
    report(root, images, sizes, modes, broken)


if __name__ == "__main__":
    main()
