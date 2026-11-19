"""Маніфест вибірки: розбір імен, читання та запис."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

FIELD_NAMES = ["path", "group", "subject", "hand", "finger", "alteration", "sha256"]


@dataclass(frozen=True)
class ManifestEntry:
    """Запис вибірки: файл зображення та його ознаки."""

    path: str
    group: str
    subject: int
    hand: str
    finger: str
    alteration: str
    sha256: str


def parse_name(stem: str) -> tuple[int, str, str, str]:
    """Розбирає ім'я файлу на суб'єкта, руку, палець і тип зміни."""
    subject_part, rest = stem.split("__")
    tokens = rest.split("_")
    hand = tokens[1]
    finger = tokens[2]
    alteration = tokens[4] if len(tokens) > 4 else ""
    return int(subject_part), hand, finger, alteration


def file_digest(path: Path) -> str:
    """Повертає sha256 вмісту файлу."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    """Записує маніфест у TSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(FIELD_NAMES)
        for entry in entries:
            writer.writerow(
                [
                    entry.path,
                    entry.group,
                    entry.subject,
                    entry.hand,
                    entry.finger,
                    entry.alteration,
                    entry.sha256,
                ]
            )


def read_manifest(path: Path) -> list[ManifestEntry]:
    """Читає маніфест із TSV."""
    entries = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entries.append(
                ManifestEntry(
                    path=row["path"],
                    group=row["group"],
                    subject=int(row["subject"]),
                    hand=row["hand"],
                    finger=row["finger"],
                    alteration=row["alteration"],
                    sha256=row["sha256"],
                )
            )
    return entries
