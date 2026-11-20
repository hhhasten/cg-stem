"""Післяобробка DOCX звіту."""

# pylint: disable=protected-access

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from report_title import add_title_page

BREAK_BEFORE = (
    "АНОТАЦІЯ", "1 ВСТУП", "2 ОПИС", "3 МЕТОДИ", "4 РЕЗУЛЬТАТИ",
    "5 ОБГОВОРЕННЯ", "6 ВИСНОВКИ", "СПИСОК ДЖЕРЕЛ",
    "ДОДАТОК А", "ДОДАТОК Б",
)


def postprocess(path: Path, title: str) -> None:
    """Додає титульну сторінку, номери сторінок і форматування таблиць."""
    document = Document(path)
    document.core_properties.title = title
    document.core_properties.author = "Здобувач"
    add_title_page(document)
    format_tables(document)
    format_captions(document)
    keep_code_together(document)
    add_page_breaks(document)
    add_footer(document)
    enable_field_update(document)
    document.save(path)


def format_tables(document: Document) -> None:
    """Задає сітку, шрифт і повторення заголовка таблиць."""
    for table in document.tables:
        apply_table_borders(table)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for row_index, row in enumerate(table.rows):
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.first_line_indent = Cm(0)
                    paragraph.paragraph_format.line_spacing = 1.0
                    for run in paragraph.runs:
                        run.font.name = "Times New Roman"
                        run.font.size = Pt(12)
                        run.bold = row_index == 0 or bool(run.bold)


def apply_table_borders(table) -> None:
    """Додає тонкі сірі межі всім клітинкам таблиці."""
    properties = table._tbl.tblPr
    existing = properties.find(qn("w:tblBorders"))
    if existing is not None:
        properties.remove(existing)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "808080")
        borders.append(element)
    properties.append(borders)


def format_captions(document: Document) -> None:
    """Вирівнює підписи та прибирає англійський префікс Pandoc."""
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text.startswith("Таблиця"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.first_line_indent = Cm(0)
            paragraph.paragraph_format.keep_with_next = True
        if text.startswith("Figure"):
            paragraph.text = re.sub(r"^Figure\s+\d+[:.]\s*", "Рисунок ", text)


def keep_code_together(document: Document) -> None:
    """Забороняє розривати лістинги між сторінками."""
    for paragraph in document.paragraphs:
        if paragraph.style.name == "Source Code":
            paragraph.paragraph_format.keep_together = True


def add_page_breaks(document: Document) -> None:
    """Починає великі розділи та додатки з нової сторінки."""
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text.startswith(BREAK_BEFORE):
            paragraph.paragraph_format.page_break_before = True


def add_footer(document: Document) -> None:
    """Додає номер сторінки у нижній колонтитул."""
    section = document.sections[0]
    section.different_first_page_header_footer = True
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin, instruction, end = make_page_field()
    run._r.append(begin)
    run._r.append(instruction)
    run._r.append(end)


def make_page_field() -> tuple:
    """Створює XML-поле номера сторінки."""
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    return begin, instruction, end


def enable_field_update(document: Document) -> None:
    """Просить Word оновити зміст під час відкриття."""
    settings = document.settings.element
    update = OxmlElement("w:updateFields")
    update.set(qn("w:val"), "true")
    settings.append(update)
    shade = OxmlElement("w:doNotShadeFormData")
    settings.append(shade)
