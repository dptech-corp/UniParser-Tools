"""Merge local PDFs in order (question booklet then answer booklet)."""

from __future__ import annotations

from pathlib import Path


def merge_pdfs(paths: list[str | Path], output_path: str | Path) -> Path:
    """Append PDFs in order into ``output_path`` and return the resolved path."""
    from pypdf import PdfWriter

    if not paths:
        raise ValueError("At least one PDF path is required.")

    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        resolved.append(path)

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = PdfWriter()
    try:
        for path in resolved:
            writer.append(str(path))
        if len(writer.pages) == 0:
            raise ValueError("Merged PDF has no pages.")
        with out.open("wb") as fh:
            writer.write(fh)
    finally:
        writer.close()

    return out
