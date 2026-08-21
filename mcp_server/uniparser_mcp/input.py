"""Input resolution for uniparser_parse."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

from uniparser_mcp.defaults import IMAGE_SUFFIXES
from uniparser_mcp.schemas import ParseRequest


class InputKind(str, Enum):
    FILE = "file"
    IMAGE = "image"
    URL = "url"


@dataclass(frozen=True)
class ResolvedInput:
    kind: InputKind
    source_stem: str
    raw: str
    token_seed: str
    path: Path | None = None


def source_stem_from_path(path: Path) -> str:
    return path.stem or "document"


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


def display_label(resolved: ResolvedInput) -> str:
    if resolved.path is not None:
        return resolved.path.name
    segment = urlparse(resolved.raw).path.rstrip("/").rsplit("/", 1)[-1]
    return segment or "url_document"


def resolve_request(req: ParseRequest) -> ResolvedInput | str:
    if req.pdf_url:
        url = req.pdf_url.strip()
        if not url.startswith(("http://", "https://")):
            return "pdf_url must start with http:// or https://"
        return ResolvedInput(
            kind=InputKind.URL,
            source_stem=source_stem_from_url(url),
            raw=url,
            token_seed=url,
        )

    path_str = (req.file_path or req.image_path or "").strip()
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        return f"File not found: {path}"

    if req.image_path:
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            return f"Not a supported image type: {path.suffix}"
        return ResolvedInput(
            kind=InputKind.IMAGE,
            source_stem=source_stem_from_path(path),
            raw=path_str,
            token_seed=str(path),
            path=path,
        )

    return ResolvedInput(
        kind=InputKind.FILE,
        source_stem=source_stem_from_path(path),
        raw=path_str,
        token_seed=str(path),
        path=path,
    )
