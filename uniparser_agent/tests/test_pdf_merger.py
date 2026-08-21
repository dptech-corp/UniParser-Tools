"""Tests for PDF merge helper."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from uniparser_agent.pdf2vqa.pdf_merger import merge_pdfs


def _write_blank_pdf(path: Path, n_pages: int) -> Path:
    writer = PdfWriter()
    for _ in range(n_pages):
        writer.add_blank_page(width=200, height=200)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        writer.write(fh)
    writer.close()
    return path


def test_merge_pdfs_sums_pages(tmp_path: Path):
    a = _write_blank_pdf(tmp_path / "a.pdf", 2)
    b = _write_blank_pdf(tmp_path / "b.pdf", 3)
    out = merge_pdfs([a, b], tmp_path / "merged.pdf")
    assert out.is_file()
    assert len(PdfReader(str(out)).pages) == 5


def test_merge_pdfs_rejects_missing(tmp_path: Path):
    a = _write_blank_pdf(tmp_path / "a.pdf", 1)
    with pytest.raises(FileNotFoundError):
        merge_pdfs([a, tmp_path / "missing.pdf"], tmp_path / "out.pdf")


def test_merge_pdfs_rejects_non_pdf(tmp_path: Path):
    txt = tmp_path / "notes.txt"
    txt.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Not a PDF"):
        merge_pdfs([txt], tmp_path / "out.pdf")
