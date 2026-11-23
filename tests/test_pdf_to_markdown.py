"""Tests for utils/pdf_to_markdown.py.

tests/fixtures/sample_resume.pdf is a tiny made-up, 2-page PDF:
    page 1: "Alex Example" / "Data Engineer"
    page 2: "Skills: Python, SQL"
Tests copy it into `tmp_path` first, so the .md files they write never land
in tests/fixtures/.
"""

import shutil
import sys
from pathlib import Path

import pytest

from jobs_search.utils.pdf_to_markdown import clean_text, convert_pdf, main, pdf_to_text

SAMPLE_PDF = Path("tests/fixtures/sample_resume.pdf")


def copy_sample(tmp_path: Path) -> Path:
    """Copy the sample PDF into the test's temporary folder and return its new path."""
    destination = tmp_path / "resume.pdf"
    shutil.copy(SAMPLE_PDF, destination)
    return destination


def test_clean_text_tidies_spaces_and_blank_lines():
    messy = "\n\nName   \n\n\n\nSkills  \n\n"

    assert clean_text(messy) == "Name\n\nSkills\n"


def test_pdf_to_text_reads_every_page():
    text = pdf_to_text(SAMPLE_PDF)

    assert text == "Alex Example\nData Engineer\n\nSkills: Python, SQL\n"


def test_convert_pdf_writes_md_next_to_the_pdf(tmp_path):
    pdf_path = copy_sample(tmp_path)

    md_path = convert_pdf(pdf_path)

    assert md_path == tmp_path / "resume.md"
    assert md_path.read_text(encoding="utf-8").startswith("Alex Example")


def test_convert_pdf_never_overwrites_an_existing_md(tmp_path):
    pdf_path = copy_sample(tmp_path)
    (tmp_path / "resume.md").write_text("my edits", encoding="utf-8")

    with pytest.raises(FileExistsError):
        convert_pdf(pdf_path)

    # The edited file is untouched.
    assert (tmp_path / "resume.md").read_text(encoding="utf-8") == "my edits"


def test_convert_pdf_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_pdf(tmp_path / "nope.pdf")


def test_main_converts_files_named_on_the_command_line(tmp_path, monkeypatch):
    pdf_path = copy_sample(tmp_path)
    # Pretend the user typed: python -m jobs_search.utils.pdf_to_markdown resume.pdf
    monkeypatch.setattr(sys, "argv", ["pdf_to_markdown", str(pdf_path)])

    main()

    assert (tmp_path / "resume.md").exists()
