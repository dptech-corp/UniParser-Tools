from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

from uniparser_tools.cli.core.defaults import IMAGE_SUFFIXES


class InputKind(str, Enum):
    FILE = "file"
    IMAGE = "image"
    URL = "url"


@dataclass(frozen=True)
class ResolvedInput:
    kind: InputKind
    source_stem: str
    raw: str
    path: Path | None = None


def source_stem_from_path(path: Path) -> str:
    return path.stem or "document"


def display_label_for_input(resolved: ResolvedInput) -> str:
    if resolved.path is not None:
        return resolved.path.name
    segment = urlparse(resolved.raw).path.rstrip("/").rsplit("/", 1)[-1]
    return segment or "url_document"


def source_stem_from_url(url: str) -> str:
    segment = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
    if not segment:
        return "url_document"
    lower = segment.lower()
    for ext in (".pdf", ".png", ".jpg", ".jpeg", ".webp"):
        if lower.endswith(ext):
            segment = segment[: -len(ext)]
            break
    return segment or "url_document"


def resolve_input(raw: str) -> ResolvedInput | str:
    """Return ResolvedInput or an error message."""
    text = raw.strip()
    if not text:
        return "INPUT must not be empty."

    if text.startswith("http://") or text.startswith("https://"):
        return ResolvedInput(kind=InputKind.URL, source_stem=source_stem_from_url(text), raw=text)

    path = Path(text).expanduser().resolve()
    if not path.is_file():
        return f"File not found: {path}"

    if path.suffix.lower() in IMAGE_SUFFIXES:
        return ResolvedInput(
            kind=InputKind.IMAGE,
            source_stem=source_stem_from_path(path),
            raw=text,
            path=path,
        )
    return ResolvedInput(
        kind=InputKind.FILE,
        source_stem=source_stem_from_path(path),
        raw=text,
        path=path,
    )
