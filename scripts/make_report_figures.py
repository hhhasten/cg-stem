"""Побудова всіх рисунків для звіту."""

from __future__ import annotations

from pathlib import Path

from report_charts import main as build_charts
from report_figures import main as build_images

from cgstem.paths import repo_root


def main() -> None:
    """Будує рисунки зображень і діаграми метрик."""
    build_images()
    build_charts()
    print(f"Рисунки: {Path(repo_root()) / 'docs' / 'report' / 'img'}")


if __name__ == "__main__":
    main()
