"""Unit tests for ``uniparser_tools.utils.pdf_render``.

``pdf_render`` is the permissive-licensed (pypdfium2 / PDFium, BSD-3/Apache-2.0)
replacement for the AGPL PyMuPDF (fitz) that the toolkit previously used only to
rasterise PDF pages / clipped regions. These tests pin the small fitz-compatible
contract the call sites rely on:

  * ``Document`` open / ``len`` / indexing, ``page.rect`` geometry
  * full-page render dims = ``round(size_pt * dpi / 72)``, RGB, packed samples
  * clipped ``get_pixmap`` == cropping the full render at the same scale
    (pixel-identical -- the property that makes the swap behaviour-preserving)
  * ``Rect`` helpers, tuple clips, degenerate-clip guard, render caching, save
"""

from __future__ import annotations

import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from uniparser_tools.utils import pdf_render


def _build_synthetic_pdf_bytes() -> bytes:
    """Two-page PDF: page0 200x300 with colored rects, page1 blank 400x200."""
    objects = [
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n",
        b"2 0 obj<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>endobj\n",
        (
            b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 300] "
            b"/Contents 4 0 R /Resources<< /ProcSet [/PDF] >> >>endobj\n"
        ),
    ]
    stream = b"1 0 0 rg\n20 40 80 120 re f\n0 0 1 rg\n100 150 60 80 re f\n"
    objects.append(f"4 0 obj<< /Length {len(stream)} >>stream\n".encode() + stream + b"endstream\nendobj\n")
    objects.append(
        b"5 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Resources<< /ProcSet [/PDF] >> >>endobj\n"
    )

    header = b"%PDF-1.4\n"
    body = b""
    offsets = [0]
    pos = len(header)
    for obj in objects:
        offsets.append(pos)
        body += obj
        pos += len(obj)

    xref = [b"xref\n", f"0 {len(offsets)}\n".encode(), b"0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref.append(f"{off:010d} 00000 n \n".encode())
    trailer = f"trailer<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{pos}\n%%EOF\n".encode()
    return header + body + b"".join(xref) + trailer


@pytest.fixture(scope="module")
def synthetic_pdf(tmp_path_factory) -> str:
    """Deterministic 2-page PDF with distinct sizes + page-0 content.

    Built in-process so CI needs no checked-in ``*.pdf`` (gitignored).
    """
    path = tmp_path_factory.mktemp("pdf_render") / "synthetic.pdf"
    path.write_bytes(_build_synthetic_pdf_bytes())
    return str(path)


# --- open / len / indexing / rect geometry ---------------------------------


def test_open_len_and_indexing(synthetic_pdf):
    doc = pdf_render.Document(synthetic_pdf)
    assert len(doc) == 2
    assert isinstance(doc[0], pdf_render.Page)
    doc.close()


def test_page_rect_geometry(synthetic_pdf):
    doc = pdf_render.Document(synthetic_pdf)
    r0, r1 = doc[0].rect, doc[1].rect
    assert (r0.width, r0.height) == (200.0, 300.0)
    assert (r1.width, r1.height) == (400.0, 200.0)
    # fitz reports a top-left origin at (0, 0); call sites add it explicitly.
    assert r0.top_left == (0.0, 0.0)
    doc.close()


# --- full-page render ------------------------------------------------------


@pytest.mark.parametrize("dpi", [72, 144, 100])
def test_render_page_dims_and_samples(synthetic_pdf, dpi):
    doc = pdf_render.Document(synthetic_pdf)
    pix = doc[0].get_pixmap(dpi=dpi)
    scale = dpi / 72.0
    assert pix.width == round(200 * scale)
    assert pix.height == round(300 * scale)
    # packed RGB bytes -> Image.frombytes round-trips (the exact call-site pattern)
    assert len(pix.samples) == pix.width * pix.height * 3
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    assert img.size == (pix.width, pix.height)
    assert img.mode == "RGB"
    doc.close()


# --- Rect helpers ----------------------------------------------------------


def test_rect_helpers():
    rect = pdf_render.Rect(10, 20, 110, 220)
    assert rect.width == 100 and rect.height == 200
    assert rect.top_left == (10, 20)
    assert tuple(rect) == (10.0, 20.0, 110.0, 220.0)


# --- clipped render: the behaviour-preserving property ---------------------


def test_clip_matches_full_crop_pixel_identical(synthetic_pdf):
    """A clipped pixmap must equal cropping the full-page render at the same
    scale -- proves the coordinate math (offset *and* size) is correct on a
    content-bearing page (no checked-in PDF; ``*.pdf`` is gitignored)."""
    doc = pdf_render.Document(synthetic_pdf)
    rect = doc[0].rect
    dpi = 120
    full_pix = doc[0].get_pixmap(dpi=dpi)
    full = Image.frombytes("RGB", (full_pix.width, full_pix.height), full_pix.samples)
    # an off-origin sub-rectangle (points, top-left origin)
    clip = pdf_render.Rect(0.15 * rect.width, 0.10 * rect.height, 0.65 * rect.width, 0.55 * rect.height)
    pix = pdf_render.get_pixmap(doc[0], clip=clip, dpi=dpi)
    got = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    s = dpi / 72.0
    box = (round(clip.x0 * s), round(clip.y0 * s), round(clip.x1 * s), round(clip.y1 * s))
    ref = full.crop(box)
    assert got.size == ref.size
    assert int(np.abs(np.asarray(got).astype(int) - np.asarray(ref).astype(int)).max()) == 0
    doc.close()


def test_clip_accepts_plain_tuple(synthetic_pdf):
    doc = pdf_render.Document(synthetic_pdf)
    pix = doc[0].get_pixmap(clip=(0, 0, 100, 150), dpi=72)
    assert (pix.width, pix.height) == (100, 150)
    doc.close()


def test_degenerate_clip_returns_white_1x1(synthetic_pdf):
    doc = pdf_render.Document(synthetic_pdf)
    # zero-area / inverted clip -> guarded 1x1 white tile, never a crash
    pix = doc[0].get_pixmap(clip=(50, 50, 50, 50), dpi=72)
    assert (pix.width, pix.height) == (1, 1)
    assert Image.frombytes("RGB", (1, 1), pix.samples).getpixel((0, 0)) == (255, 255, 255)
    doc.close()


def test_clip_clamped_to_page_bounds(synthetic_pdf):
    doc = pdf_render.Document(synthetic_pdf)
    # clip extends past the page; result is clamped to the page extent
    pix = doc[0].get_pixmap(clip=(-20, -20, 9999, 9999), dpi=72)
    assert (pix.width, pix.height) == (200, 300)
    doc.close()


# --- module function delegates to the page method --------------------------


def test_module_get_pixmap_delegates(synthetic_pdf):
    doc = pdf_render.Document(synthetic_pdf)
    page = doc[0]
    a = pdf_render.get_pixmap(page, dpi=72)
    b = page.get_pixmap(dpi=72)
    assert (a.width, a.height) == (b.width, b.height) == (200, 300)
    doc.close()


# --- render caching --------------------------------------------------------


def test_full_render_is_cached(synthetic_pdf):
    doc = pdf_render.Document(synthetic_pdf)
    first = doc._render_full(0, 144)
    second = doc._render_full(0, 144)
    assert first is second  # same (index, dpi) -> reused PIL image
    assert doc._render_full(1, 144) is not first  # different page -> distinct
    doc.close()


def test_render_cache_is_lru_bounded(synthetic_pdf):
    """The full-page cache must not grow without bound: a full raster can be
    tens of MB, so a long PDF would blow up memory. Bound it to ``cache_pages``
    with LRU eviction, keeping the recently-touched pages."""
    doc = pdf_render.Document(synthetic_pdf, cache_pages=1)
    a1 = doc._render_full(0, 72)
    assert list(doc._full_cache) == [(0, 72)]
    # touching page 1 evicts page 0 (capacity 1)
    doc._render_full(1, 72)
    assert list(doc._full_cache) == [(1, 72)]
    # page 0 must be re-rendered (fresh object) and still be correct
    a2 = doc._render_full(0, 72)
    assert a2 is not a1
    assert a2.size == a1.size
    doc.close()


def test_render_cache_lru_keeps_recent(synthetic_pdf):
    """With room for 2 pages, re-touching the older page keeps it hot so the
    next insert evicts the *other* one (true LRU, not FIFO)."""
    doc = pdf_render.Document(synthetic_pdf, cache_pages=2)
    doc._render_full(0, 72)
    doc._render_full(1, 72)
    doc._render_full(0, 72)  # page 0 now most-recently-used
    doc._render_full(0, 144)  # third distinct key -> evicts LRU == (1, 72)
    assert (1, 72) not in doc._full_cache
    assert (0, 72) in doc._full_cache
    doc.close()


# --- rotation consistency --------------------------------------------------


def test_rotated_page_size_and_render_agree(tmp_path):
    """The clip math assumes ``rect`` (from ``get_size``) and the rendered
    pixmap share one orientation. Pin that for an intrinsically /Rotate-90 page:
    the reported rect and the pixmap dims must both be the rotated extent."""
    src = pdfium.PdfDocument.new()
    page = src.new_page(200, 300)  # portrait before rotation
    page.set_rotation(90)
    path = tmp_path / "rotated.pdf"
    src.save(str(path))
    src.close()

    doc = pdf_render.Document(str(path))
    rect = doc[0].rect
    assert (rect.width, rect.height) == (300.0, 200.0)  # rotated extent
    pix = doc[0].get_pixmap(dpi=72)
    assert (pix.width, pix.height) == (300, 200)  # render matches rect -> clip math stays valid
    doc.close()


def test_rotated_page_clip_matches_full_crop(tmp_path):
    """Clip == full-crop must still hold once rotation is in play."""
    src = pdfium.PdfDocument.new()
    page = src.new_page(200, 300)
    page.set_rotation(90)
    path = tmp_path / "rotated2.pdf"
    src.save(str(path))
    src.close()

    doc = pdf_render.Document(str(path))
    dpi = 100
    full_pix = doc[0].get_pixmap(dpi=dpi)
    full = Image.frombytes("RGB", (full_pix.width, full_pix.height), full_pix.samples)
    clip = pdf_render.Rect(40, 30, 180, 150)
    pix = doc[0].get_pixmap(clip=clip, dpi=dpi)
    got = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    s = dpi / 72.0
    ref = full.crop((round(clip.x0 * s), round(clip.y0 * s), round(clip.x1 * s), round(clip.y1 * s)))
    assert got.size == ref.size
    assert np.array_equal(np.asarray(got), np.asarray(ref))
    doc.close()


# --- Pixmap normalizes to packed RGB ---------------------------------------


def test_pixmap_converts_non_rgb_to_rgb():
    """Call sites do ``Image.frombytes("RGB", (w, h), pix.samples)``; a render
    that comes back RGBA/LA/L must be normalized so ``samples`` stays 3-byte."""
    rgba = Image.new("RGBA", (5, 4), (10, 20, 30, 128))
    pix = pdf_render.Pixmap(rgba)
    assert pix.pil.mode == "RGB"
    assert (pix.width, pix.height) == (5, 4)
    assert len(pix.samples) == 5 * 4 * 3
    assert Image.frombytes("RGB", (5, 4), pix.samples).getpixel((0, 0)) == (10, 20, 30)


def test_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        pdf_render.Document(str(tmp_path / "does_not_exist.pdf"))


# --- save + context manager ------------------------------------------------


def test_pixmap_save(synthetic_pdf, tmp_path):
    doc = pdf_render.Document(synthetic_pdf)
    out = tmp_path / "page0.png"
    doc[0].get_pixmap(dpi=72).save(out)
    assert out.is_file() and out.stat().st_size > 0
    assert Image.open(out).size == (200, 300)
    doc.close()


def test_context_manager_closes(synthetic_pdf):
    with pdf_render.Document(synthetic_pdf) as doc:
        assert len(doc) == 2
    # cache cleared on close
    assert doc._full_cache == {}
