"""Збірка звіту у DOCX."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from make_report_figures import main as build_figures
from report_layout import postprocess
from report_style import dump_reference, style_reference

REPORT_TITLE = "Аналіз та обробка зображень відбитків пальців"


def find_pandoc() -> str:
    """Знаходить виконуваний файл Pandoc."""
    candidate = shutil.which("pandoc")
    if candidate:
        return candidate
    local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
    fallback = local_app_data / "Pandoc" / "pandoc.exe"
    if fallback.exists():
        return str(fallback)
    raise SystemExit(
        "Pandoc не знайдено. Встановіть його командою "
        "winget install JohnMacFarlane.Pandoc"
    )


def build_document(pandoc: str, report_path: Path, output_path: Path) -> None:
    """Запускає Pandoc із шаблоном оформлення."""
    reference = Path(tempfile.gettempdir()) / "cgstem_report_reference.docx"
    dump_reference(pandoc, reference)
    style_reference(reference)
    command = [
        pandoc,
        str(report_path),
        "--from=markdown",
        "--to=docx",
        "--standalone",
        f"--reference-doc={reference}",
        "--toc",
        "--toc-depth=2",
        "--metadata=toc-title=ЗМІСТ",
        "--metadata=lang=uk-UA",
        f"--resource-path={report_path.parent}",
        "--columns=100",
        f"--output={output_path}",
    ]
    subprocess.run(command, check=True, cwd=report_path.parent)


def main() -> None:
    """Будує рисунки та DOCX звіту."""
    root = Path(__file__).resolve().parents[1]
    report_path = root / "docs" / "report" / "report.md"
    output_path = root / "docs" / "report" / "report.docx"
    build_figures()
    build_document(find_pandoc(), report_path, output_path)
    postprocess(output_path, REPORT_TITLE)
    print(f"Звіт: {output_path}")


if __name__ == "__main__":
    main()
