"""Convert UniParser pages_tree into a flat LLM content list with ids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from uniparser_agent.pdf2vqa.image_export import (
    IMAGE_SOURCE_TYPES,
    export_images_from_pages_tree,
    iter_all_blocks,
)


SKIP_TYPES = frozenset({"hline", "pagebar", "pageheader", "pagefooter", "pagenote", "pagenumber", "watermark"})
TEXT_TYPES = frozenset({"paragraph", "title", "documenttitle"})
CAPTION_TYPES = frozenset(
    {
        "algorithmcaption",
        "expressioncaption",
        "figurecaption",
        "imagecaption",
        "tablecaption",
    }
)
IMAGE_TYPES = frozenset({"figure", "image", "chart", "table", "molecule", "figuregroup", "imagegroup"})


def _format_inline(content: str, content_type: str) -> str:
    normalized_type = content_type.strip().lower()
    if normalized_type in {"equation", "equationinline"}:
        if content.startswith(("$", r"\(", r"\[")):
            return content
        return f"${content}$"
    if normalized_type == "molecule":
        return content if content.startswith("`") else f"`{content}`"
    return content


def _inline_contents(block: dict[str, Any]) -> str:
    contents = block.get("contents")
    if not isinstance(contents, list) or not contents:
        return ""
    types = block.get("types")
    if not isinstance(types, list) or len(types) != len(contents):
        types = ["text"] * len(contents)
    return "".join(_format_inline(str(content), str(content_type)) for content, content_type in zip(contents, types))


def _table_text(block: dict[str, Any]) -> str:
    structure = block.get("structure") or block.get("html") or ""
    placeholders = block.get("placeholders")
    contents = block.get("contents")
    types = block.get("types")
    if (
        isinstance(structure, str)
        and isinstance(placeholders, list)
        and isinstance(contents, list)
        and len(placeholders) == len(contents)
    ):
        if not isinstance(types, list) or len(types) != len(contents):
            types = ["text"] * len(contents)
        for placeholder, content, content_type in zip(
            reversed(placeholders),
            reversed(contents),
            reversed(types),
        ):
            structure = structure.replace(
                str(placeholder),
                _format_inline(str(content), str(content_type)),
            )
        return structure.strip()
    return _inline_contents(block)


def _block_text(block: dict[str, Any]) -> str:
    btype = (block.get("type") or "").strip().lower()
    if btype == "equation":
        latex = (block.get("latex_repr") or "").strip()
        if latex:
            if latex.startswith("$$") or latex.startswith("$"):
                return latex
            return f"$$\n{latex}\n$$"
    if btype == "table":
        table_text = _table_text(block)
        if table_text:
            return table_text
    inline_text = _inline_contents(block)
    if inline_text:
        return inline_text.strip()
    if btype == "molecule":
        molecule_text = block.get("esmi") or block.get("smi") or block.get("caption")
        if isinstance(molecule_text, str) and molecule_text.strip():
            return f"`{molecule_text.strip()}`"
    return (block.get("text") or "").strip()


def _iter_blocks(pages_tree: list[Any]) -> list[dict[str, Any]]:
    """Flatten pages into a reading-order list including nested ``items``."""
    return iter_all_blocks(pages_tree)


def _caption_from_group(block: dict[str, Any]) -> list[str]:
    captions: list[str] = []
    items = block.get("items")
    if not isinstance(items, list):
        return captions
    for child in items:
        if not isinstance(child, dict):
            continue
        ctype = (child.get("type") or "").strip().lower()
        if ctype in CAPTION_TYPES:
            text = _block_text(child)
            if text:
                captions.append(text)
    return captions


def _resolve_image_source_block(block: dict[str, Any]) -> dict[str, Any] | None:
    """Return the block that actually holds image ``source`` (self or first child)."""
    source = block.get("source")
    if isinstance(source, str) and source.strip():
        return block
    items = block.get("items")
    if isinstance(items, list):
        for child in items:
            if not isinstance(child, dict):
                continue
            child_source = child.get("source")
            if isinstance(child_source, str) and child_source.strip():
                return child
    return None


def pages_tree_to_content_list(
    pages_tree_data: dict[str, Any] | list[Any],
    *,
    image_path_map: dict[tuple[Any, Any], Path] | None = None,
    image_prefix: str = "vqa_images",
) -> list[dict[str, Any]]:
    """Adapt UniParser pages_tree envelope (or raw list) to LLM content list."""
    if isinstance(pages_tree_data, dict):
        pages = pages_tree_data.get("pages_tree")
        if pages is None:
            raise ValueError("Invalid pages_tree data: missing 'pages_tree' key")
    else:
        pages = pages_tree_data

    if not isinstance(pages, list):
        raise ValueError(f"Expected pages_tree list, got {type(pages)}")

    image_path_map = image_path_map or {}
    content: list[dict[str, Any]] = []
    next_id = 0
    emitted_image_files: set[str] = set()

    for block in _iter_blocks(pages):
        btype = (block.get("type") or "").strip().lower()
        if btype in SKIP_TYPES or btype in CAPTION_TYPES:
            continue

        text = _block_text(block)

        if btype == "equation":
            source = block.get("source") or ""
            has_source = isinstance(source, str) and bool(source.strip())
            if not text and not has_source:
                continue
            content.append({"id": next_id, "type": "equation", "text": text})
            next_id += 1
            continue

        if btype in IMAGE_TYPES or btype in IMAGE_SOURCE_TYPES:
            source_block = _resolve_image_source_block(block)
            if source_block is not None:
                key = (source_block.get("page"), source_block.get("block"))
                img_file = image_path_map.get(key)
                if img_file is not None and img_file.is_file():
                    name = img_file.name
                    if name not in emitted_image_files:
                        captions = _caption_from_group(block)
                        if not captions:
                            desc = (block.get("desc") or source_block.get("desc") or "").strip()
                            if desc:
                                captions = [desc]
                        content.append(
                            {
                                "id": next_id,
                                "type": "image",
                                "img_path": f"{image_prefix}/{name}",
                                "image_caption": captions,
                            }
                        )
                        emitted_image_files.add(name)
                        next_id += 1
                    continue
            if btype in {"figuregroup", "imagegroup"}:
                continue

        if btype == "table" and text:
            content.append({"id": next_id, "type": "table", "table_body": text})
            next_id += 1
            continue

        if btype in TEXT_TYPES or text:
            if not text:
                continue
            content.append({"id": next_id, "type": "text", "text": text})
            next_id += 1
            continue

    return content


def adapt_pages_tree_file(
    pages_tree_path: str | Path,
    output_path: str | Path,
    *,
    images_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    path = Path(pages_tree_path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))

    if images_dir is None:
        images_dir = Path(output_path).expanduser().resolve().parent / "vqa_images"
    else:
        images_dir = Path(images_dir).expanduser().resolve()

    image_path_map = export_images_from_pages_tree(data, images_dir)
    content = pages_tree_to_content_list(data, image_path_map=image_path_map)

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return content
