"""Permissive-licensed PDF page rasterizer (PDFium via ``pypdfium2``).

This module replaces the only thing UniParser-Tools used **PyMuPDF / fitz** for:
rendering a PDF page -- or a clipped sub-rectangle of one -- to a PIL image at a
target DPI.

Why: PyMuPDF is licensed **AGPL-3.0 (or a paid commercial license)**, whose
copyleft terms are a real risk to redistribute inside a product. ``pypdfium2``
wraps Google's PDFium and is licensed **Apache-2.0 / BSD-3-Clause** with no
copyleft obligation, so it is safe to ship.

To keep the swap low-risk, this module deliberately mirrors the *small* slice of
the fitz API the toolkit actually called, so the call sites only change their
import line::

    # before
    import fitz  # PyMuPDF
    from fitz.utils import get_pixmap

    # after
    from uniparser_tools.utils import pdf_render as fitz
    from uniparser_tools.utils.pdf_render import get_pixmap

Supported surface (everything the codebase used):

  * ``Document(path)``                          -> open, ``len()``, ``doc[i]``
  * ``page.rect`` -> ``.width`` / ``.height`` / ``.top_left``
  * ``page.get_pixmap(dpi=...)``                -> full-page ``Pixmap``
  * ``get_pixmap(page, clip=Rect, dpi=...)``    -> clipped ``Pixmap``
  * ``Rect(x0, y0, x1, y1)``
  * ``pix.width`` / ``pix.height`` / ``pix.samples`` (raw RGB bytes) / ``pix.save(path)``

Coordinate convention matches fitz: clip coordinates are PDF userspace points
(1/72 inch) with a **top-left origin, y growing downward** -- identical to
``fitz.Page.rect`` and ``fitz.Rect``. A clip is rendered as the corresponding
sub-region of the full page at the same DPI (full page is rendered once per
``(page, dpi)`` and cached, then cropped), reproducing fitz's clipped pixmap.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence, Tuple, Union

import pypdfium2 as pdfium
from PIL import Image


class Rect:
    """Minimal stand-in for ``fitz.Rect`` (the attributes the toolkit used)."""

    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, x0: float, y0: float, x1: float, y1: float):
        self.x0, self.y0, self.x1, self.y1 = float(x0), float(y0), float(x1), float(y1)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def top_left(self) -> Tuple[float, float]:
        return (self.x0, self.y0)

    def __iter__(self):
        return iter((self.x0, self.y0, self.x1, self.y1))

    def __repr__(self) -> str:
        return f"Rect({self.x0}, {self.y0}, {self.x1}, {self.y1})"


class Pixmap:
    """Wraps a rendered PIL image with the fitz ``Pixmap`` attributes used."""

    __slots__ = ("_img",)

    def __init__(self, img: Image.Image):
        self._img = img if img.mode == "RGB" else img.convert("RGB")

    @property
    def width(self) -> int:
        return self._img.width

    @property
    def height(self) -> int:
        return self._img.height

    @property
    def samples(self) -> bytes:
        # Raw, tightly-packed RGB bytes so that
        # ``Image.frombytes("RGB", (pix.width, pix.height), pix.samples)`` round-trips.
        return self._img.tobytes()

    @property
    def pil(self) -> Image.Image:
        return self._img

    def save(self, path) -> None:
        self._img.save(str(path))


def _to_xyxy(clip: Union[Rect, Sequence[float], None]) -> Union[Tuple[float, float, float, float], None]:
    if clip is None:
        return None
    if isinstance(clip, Rect):
        return (clip.x0, clip.y0, clip.x1, clip.y1)
    x0, y0, x1, y1 = clip
    return (float(x0), float(y0), float(x1), float(y1))


class Page:
    """A single page bound to its owning :class:`Document`."""

    __slots__ = ("_doc", "_index")

    def __init__(self, doc: "Document", index: int):
        self._doc = doc
        self._index = index

    @property
    def rect(self) -> Rect:
        w, h = self._doc._page_size(self._index)
        # fitz reports the page rect with a top-left origin at (0, 0); the
        # call sites add ``page.rect.top_left`` explicitly, so keep it (0, 0).
        return Rect(0.0, 0.0, w, h)

    def get_pixmap(self, clip: Union[Rect, Sequence[float], None] = None, dpi: int = 72) -> Pixmap:
        full = self._doc._render_full(self._index, dpi)
        box = _to_xyxy(clip)
        if box is None:
            return Pixmap(full)
        scale = dpi / 72.0
        x0, y0, x1, y1 = box
        # clip is in points relative to the page's top-left origin (rect.x0/y0 == 0)
        px0, py0 = min(x0, x1) * scale, min(y0, y1) * scale
        px1, py1 = max(x0, x1) * scale, max(y0, y1) * scale
        crop = (
            max(0, int(round(px0))),
            max(0, int(round(py0))),
            min(full.width, int(round(px1))),
            min(full.height, int(round(py1))),
        )
        if crop[2] <= crop[0] or crop[3] <= crop[1]:
            return Pixmap(Image.new("RGB", (1, 1), "white"))
        return Pixmap(full.crop(crop))


class Document:
    """Open a PDF and render its pages. Drop-in for the fitz usage in this repo.

    A full-page render is cached per ``(page_index, dpi)`` so that several clips
    on the same page (the caption-extraction access pattern) reuse one raster
    instead of re-rendering. The cache is a small **LRU** bounded by
    ``cache_pages`` -- unlike the original per-call fitz ``get_pixmap(clip=...)``,
    a full-page raster can be large (up to ~4096 px per side, tens of MB), so an
    unbounded cache would accumulate every page of a long PDF. The default keeps
    the last few pages, which covers same-page and adjacent cross-page groups
    while bounding peak memory.
    """

    def __init__(self, path, cache_pages: int = 8):
        self._doc = pdfium.PdfDocument(str(path))
        self._cache_pages = max(1, int(cache_pages))
        self._full_cache: "OrderedDict" = OrderedDict()  # (index, dpi) -> PIL.Image (RGB), LRU
        self._size_cache: dict = {}  # index -> (w_pt, h_pt)

    def __len__(self) -> int:
        return len(self._doc)

    def __getitem__(self, index: int) -> Page:
        return Page(self, index)

    def _page_size(self, index: int) -> Tuple[float, float]:
        size = self._size_cache.get(index)
        if size is None:
            w, h = self._doc[index].get_size()
            size = (float(w), float(h))
            self._size_cache[index] = size
        return size

    def _render_full(self, index: int, dpi: int) -> Image.Image:
        key = (index, int(dpi))
        img = self._full_cache.get(key)
        if img is not None:
            self._full_cache.move_to_end(key)  # mark most-recently-used
            return img
        img = self._doc[index].render(scale=dpi / 72.0).to_pil()
        if img.mode != "RGB":
            img = img.convert("RGB")
        self._full_cache[key] = img
        while len(self._full_cache) > self._cache_pages:
            self._full_cache.popitem(last=False)  # evict least-recently-used
        return img

    def close(self) -> None:
        try:
            self._doc.close()
        except Exception:
            pass
        self._full_cache.clear()
        self._size_cache.clear()

    def __enter__(self) -> "Document":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def get_pixmap(page: Page, clip: Union[Rect, Sequence[float], None] = None, dpi: int = 72) -> Pixmap:
    """Function form mirroring ``fitz.utils.get_pixmap(page, clip=..., dpi=...)``."""
    return page.get_pixmap(clip=clip, dpi=dpi)
