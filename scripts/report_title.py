"""Титульна сторінка DOCX."""

# pylint: disable=protected-access

from __future__ import annotations

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Cm, Pt

TITLE_LINES = (
    ("МІНІСТЕРСТВО ОСВІТИ І НАУКИ УКРАЇНИ", 12, False),
    ("Київський національний університет імені Тараса Шевченка", 12, False),
    ("Факультет комп'ютерних наук та кібернетики", 12, False),
    ("Кафедра програмних систем і технологій", 12, False),
    ("", 12, False),
    ("STEM-ПРОЄКТ", 20, True),
    ("", 12, False),
    ("Аналіз та обробка зображень відбитків пальців", 14, True),
    ("класичними методами фільтрації, сегментації та аналізу текстур", 14, True),
    ("", 12, False),
    ("Здобувач: Пуш Володимир Сергійович", 14, False),
    ("Група: ІПЗ-23", 14, False),
    ("Керівник: Жабська Єлизавета Олегівна", 14, False),
    ("", 12, False),
    ("Київ, 2026", 12, False),
)


def add_title_page(document: Document) -> None:
    """Вставляє титульну сторінку перед змістом і текстом звіту."""
    body = document.element.body
    paragraphs = []
    for text, size, bold in TITLE_LINES:
        paragraph = document.add_paragraph()
        fill_title_paragraph(paragraph, text, size, bold)
        paragraphs.append(paragraph)
    break_paragraph = document.add_paragraph()
    break_paragraph.add_run().add_break(WD_BREAK.PAGE)
    paragraphs.append(break_paragraph)
    for offset, paragraph in enumerate(paragraphs):
        body.remove(paragraph._p)
        body.insert(offset, paragraph._p)


def fill_title_paragraph(paragraph, text: str, size: int, bold: bool) -> None:
    """Заповнює центрований абзац титульної сторінки."""
    run = paragraph.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold = bold
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.line_spacing = 1.0
    paragraph.paragraph_format.space_after = Pt(8)
