"""Export UniParser block ``source`` fields into a local ``vqa_images`` directory."""

from __future__ import annotations

import base64
import hashlib
import re
import shutil
from pathlib import Path
from typing import Any


_DATA_URL_RE = re.compile(
    r"^data:image/(?P<fmt>[a-zA-Z0-9.+-]+);base64,(?P<data>.+)$",
    re.DOTALL,
)

# Types that may carry visual payloads under SCIENTIFIC_PAPER_TRIGGER.
IMAGE_SOURCE_TYPES = frozenset(
    {
        "figure",
        "image",
        "chart",
        "table",
        "figuregroup",
        "imagegroup",
        "molecule",
    }
)

_SKIP_EXPORT_TYPES = frozenset(
    {
        "figurecaption",
        "imagecaption",
        "paragraph",
        "title",
        "documenttitle",
        "equation",
        "expression",
        "hline",
        "pageheader",
        "pagefooter",
        "pagenumber",
    }
)


def _block_key(block: dict[str, Any]) -> tuple[Any, Any]:
    return (block.get("page"), block.get("block"))


def _ordered_dict_blocks(blocks: list[Any]) -> list[dict[str, Any]]:
    return sorted(
        [b for b in blocks if isinstance(b, dict)],
        key=lambda b: b.get("order") if b.get("order") is not None else 10**9,
    )


def iter_all_blocks(pages_tree: list[Any]) -> list[dict[str, Any]]:
    """Flatten pages including nested ``items`` at any depth (DFS, reading order)."""
    flat: list[dict[str, Any]] = []

    def _walk(blocks: list[Any]) -> None:
        for block in _ordered_dict_blocks(blocks):
            flat.append(block)
            items = block.get("items")
            if isinstance(items, list) and items:
                _walk(items)

    for page in pages_tree:
        if not isinstance(page, list):
            continue
        _walk(page)
    return flat


def _guess_ext(fmt: str | None, raw: bytes) -> str:
    if fmt:
        fmt = fmt.lower().replace("jpeg", "jpg")
        if fmt in {"jpg", "jpeg", "png", "gif", "webp", "bmp"}:
            return "jpg" if fmt == "jpeg" else fmt
    if raw.startswith(b"\x89PNG"):
        return "png"
    if raw.startswith(b"GIF8"):
        return "gif"
    if raw.startswith(b"RIFF") and b"WEBP" in raw[:16]:
        return "webp"
    return "jpg"


def _looks_like_filesystem_path(source: str) -> bool:
    """Heuristic: avoid treating long base64 blobs as paths (OSError: name too long)."""
    if len(source) >= 4096:
        return False
    if source.startswith(("http://", "https://", "data:")):
        return False
    return ("/" in source) or ("\\" in source) or bool(Path(source).suffix)


def decode_source_to_bytes(source: str) -> tuple[bytes, str] | None:
    """Return (bytes, extension) for a block source string, or None if unsupported."""
    source = source.strip()
    if not source:
        return None

    match = _DATA_URL_RE.match(source)
    if match:
        fmt = match.group("fmt")
        raw = base64.b64decode(match.group("data"), validate=False)
        return raw, _guess_ext(fmt, raw)

    if _looks_like_filesystem_path(source):
        path = Path(source)
        try:
            if path.is_file():
                raw = path.read_bytes()
                return raw, path.suffix.lstrip(".").lower() or "jpg"
        except OSError:
            pass

    try:
        raw = base64.b64decode(source, validate=False)
    except Exception:
        return None
    if len(raw) < 32:
        return None
    return raw, _guess_ext(None, raw)


def export_images_from_pages_tree(
    pages_tree_data: dict[str, Any] | list[Any],
    images_dir: str | Path,
) -> dict[tuple[Any, Any], Path]:
    """Decode / copy block ``source`` images into ``images_dir``.

    Returns a map from ``(page, block)`` to the written absolute path.
    Duplicate content hashes share one file on disk.
    """
    if isinstance(pages_tree_data, dict):
        pages = pages_tree_data.get("pages_tree")
        if pages is None:
            raise ValueError("Invalid pages_tree data: missing 'pages_tree' key")
    else:
        pages = pages_tree_data
    if not isinstance(pages, list):
        raise ValueError(f"Expected pages_tree list, got {type(pages)}")

    out_dir = Path(images_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hash_to_path: dict[str, Path] = {}
    key_to_path: dict[tuple[Any, Any], Path] = {}

    for block in iter_all_blocks(pages):
        btype = (block.get("type") or "").strip().lower()
        if btype in _SKIP_EXPORT_TYPES:
            continue

        source = block.get("source")
        if not isinstance(source, str) or not source.strip():
            continue

        decoded = decode_source_to_bytes(source)
        if decoded is None:
            continue
        raw, ext = decoded
        digest = hashlib.sha256(raw).hexdigest()
        if digest in hash_to_path:
            key_to_path[_block_key(block)] = hash_to_path[digest]
            continue

        filename = f"{digest}.{ext or 'jpg'}"
        dest = out_dir / filename
        if not dest.exists():
            copied = False
            if _looks_like_filesystem_path(source):
                src_path = Path(source.strip())
                try:
                    if src_path.is_file():
                        shutil.copy2(src_path, dest)
                        copied = True
                except OSError:
                    copied = False
            if not copied:
                dest.write_bytes(raw)
        hash_to_path[digest] = dest
        key_to_path[_block_key(block)] = dest

    return key_to_path
