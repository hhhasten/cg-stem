"""Оформлення шаблону DOCX для звіту."""

from __future__ import annotations

import subprocess
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BODY_STYLES = ("Normal", "Body Text", "First Paragraph", "Compact")
CAPTION_STYLES = ("Caption", "Image Caption", "Table Caption")


def dump_reference(pandoc: str, path: Path) -> None:
    """Витягує стандартний шаблон DOCX із Pandoc."""
    result = subprocess.run(
        [pandoc, "--print-default-data-file", "reference.docx"],
        check=True, capture_output=True,
    )
    path.write_bytes(result.stdout)


def style_reference(path: Path) -> None:
    """Задає шрифти, відступи та поля сторінки."""
    document = Document(path)
    for name in BODY_STYLES:
        apply_body_format(get_style(document, name))
    apply_heading_format(get_style(document, "Heading 2"), 16, 12, 6)
    apply_heading_format(get_style(document, "Heading 3"), 14, 10, 4)
    apply_center_style(get_style(document, "TOC Heading"), 16, True)
    for name in CAPTION_STYLES:
        apply_caption_format(get_style(document, name))
    apply_code_format(get_style(document, "Source Code"))
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(1.5)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    document.save(path)


def get_style(document: Document, name: str):
    """Повертає стиль за назвою, створюючи його за потреби."""
    for style in document.styles:
        if style.name == name:
            return style
    return document.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)


def set_font(style, name: str, size: int) -> None:
    """Задає шрифт і розмір стилю без успадкування теми Word."""
    style.font.name = name
    style.font.size = Pt(size)
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for attribute in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        key = qn(f"w:{attribute}")
        if fonts.get(key) is not None:
            del fonts.attrib[key]
    fonts.set(qn("w:ascii"), name)
    fonts.set(qn("w:hAnsi"), name)
    fonts.set(qn("w:cs"), name)


def apply_body_format(style) -> None:
    """Задає основний текст: Times New Roman 14, інтервал 1.5, абзац."""
    set_font(style, "Times New Roman", 14)
    paragraph = style.paragraph_format
    paragraph.line_spacing = 1.5
    paragraph.first_line_indent = Cm(1.25)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.space_before = Pt(0)
    paragraph.space_after = Pt(0)


def apply_heading_format(style, size: int, before: int, after: int) -> None:
    """Задає формат заголовка розділу."""
    set_font(style, "Times New Roman", size)
    style.font.bold = True
    style.font.color.rgb = RGBColor(0, 0, 0)
    paragraph = style.paragraph_format
    paragraph.first_line_indent = Cm(0)
    paragraph.line_spacing = 1.5
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.space_before = Pt(before)
    paragraph.space_after = Pt(after)
    paragraph.keep_with_next = True


def apply_center_style(style, size: int, bold: bool) -> None:
    """Задає центрований стиль без відступу."""
    set_font(style, "Times New Roman", size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    paragraph = style.paragraph_format
    paragraph.first_line_indent = Cm(0)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.space_before = Pt(0)
    paragraph.space_after = Pt(12)


def apply_caption_format(style) -> None:
    """Задає стиль підписів до рисунків і таблиць."""
    set_font(style, "Times New Roman", 12)
    style.font.italic = False
    style.font.color.rgb = RGBColor(0, 0, 0)
    paragraph = style.paragraph_format
    paragraph.first_line_indent = Cm(0)
    paragraph.line_spacing = 1.0
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.space_before = Pt(4)
    paragraph.space_after = Pt(8)


def apply_code_format(style) -> None:
    """Задає моноширинний стиль для лістингів."""
    set_font(style, "Consolas", 9)
    paragraph = style.paragraph_format
    paragraph.first_line_indent = Cm(0)
    paragraph.line_spacing = 1.0
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.space_before = Pt(0)
    paragraph.space_after = Pt(0)
