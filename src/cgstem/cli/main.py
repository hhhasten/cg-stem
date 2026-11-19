"""Командний рядок обробки вибірки."""

from __future__ import annotations

import argparse
from pathlib import Path

from cgstem.dataset import read_manifest
from cgstem.paths import repo_root
from cgstem.pipeline.config import apply_overrides, load_config, select_methods
from cgstem.pipeline.runner import run_pipeline
from cgstem.pipeline.storage import summarize, write_metrics, write_summary


def parse_args() -> argparse.Namespace:
    """Розбирає аргументи командного рядка."""
    parser = argparse.ArgumentParser(description="Обробка вибірки зображень")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.tsv"))
    parser.add_argument("--config", type=Path, default=Path("configs/methods.yaml"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--methods", default="")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="METHOD.KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    """Запускає обробку вибірки з параметрами командного рядка."""
    args = parse_args()
    root = repo_root()
    config = load_config(root / args.config)
    config = apply_overrides(config, args.overrides)
    config = select_methods(config, selected_names(args.methods))
    entries = read_manifest(root / args.manifest)
    rows = run_pipeline(root, entries, config, root / args.out, args.limit)
    metrics_dir = root / args.out / "metrics"
    write_metrics(metrics_dir / "metrics.csv", rows)
    write_summary(metrics_dir / "summary.csv", summarize(rows))
    print(f"Оброблено знімків: {len({row['path'] for row in rows})}")
    print(f"Рядків метрик: {len(rows)}")
    print(f"Таблиця: {metrics_dir / 'metrics.csv'}")


def selected_names(value: str) -> list[str]:
    """Розбирає список методів, переданий через кому."""
    return [name.strip() for name in value.split(",") if name.strip()]


if __name__ == "__main__":
    main()
