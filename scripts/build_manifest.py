"""Побудова маніфесту вибірки з датасету SOCOFing."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from cgstem.dataset import ManifestEntry, file_digest, parse_name, write_manifest
from cgstem.paths import repo_root

EXPECTED_SIZE = (96, 103)
FINGERS_PER_SUBJECT = 10


def parse_args() -> argparse.Namespace:
    """Розбирає аргументи командного рядка."""
    parser = argparse.ArgumentParser(description="Побудова маніфесту вибірки")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/SOCOFing"))
    parser.add_argument("--output", type=Path, default=Path("data/manifest.tsv"))
    parser.add_argument("--subjects", type=int, default=12)
    parser.add_argument("--altered-subjects", type=int, default=8)
    parser.add_argument("--alteration", default="Obl")
    return parser.parse_args()


def to_relative(path: Path, root: Path) -> str:
    """Повертає шлях відносно кореня репозиторію у форматі POSIX."""
    return path.resolve().relative_to(root).as_posix()


def collect(directory: Path, root: Path, group: str, subjects: int,
            alteration: str) -> list[ManifestEntry]:
    """Відбирає файли за номером суб'єкта, типом зміни та розміром."""
    entries = []
    for path in sorted(directory.glob("*.BMP")):
        subject, hand, finger, file_alteration = parse_name(path.stem)
        if subject > subjects:
            continue
        if file_alteration != alteration:
            continue
        with Image.open(path) as image:
            if image.size != EXPECTED_SIZE:
                continue
        entries.append(
            ManifestEntry(
                path=to_relative(path, root),
                group=group,
                subject=subject,
                hand=hand,
                finger=finger,
                alteration=file_alteration,
                sha256=file_digest(path),
            )
        )
    return entries


def check_count(entries: list[ManifestEntry], expected: int, label: str) -> None:
    """Зупиняє побудову, якщо кількість записів не збігається з очікуваною."""
    if len(entries) != expected:
        raise SystemExit(f"{label}: очікувалось {expected}, знайдено {len(entries)}")


def main() -> None:
    """Будує маніфест і записує його у файл."""
    args = parse_args()
    root = repo_root()
    raw_dir = root / args.raw_dir if not args.raw_dir.is_absolute() else args.raw_dir

    real = collect(raw_dir / "Real", root, "real", args.subjects, "")
    altered = collect(
        raw_dir / "Altered" / "Altered-Hard",
        root,
        "altered",
        args.altered_subjects,
        args.alteration,
    )

    check_count(real, args.subjects * FINGERS_PER_SUBJECT, "Реальні знімки")
    check_count(altered, args.altered_subjects * FINGERS_PER_SUBJECT, "Змінені знімки")

    output = root / args.output if not args.output.is_absolute() else args.output
    entries = sorted(real + altered, key=lambda entry: entry.path)
    write_manifest(output, entries)
    print(f"Записано {len(entries)} знімків: реальних {len(real)}, змінених {len(altered)}")
    print(f"Маніфест: {output}")


if __name__ == "__main__":
    main()
