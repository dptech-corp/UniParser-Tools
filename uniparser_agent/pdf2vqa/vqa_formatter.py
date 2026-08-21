"""Convert merged VQA pairs with Markdown image refs into ShareGPT VQA format."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_IMAGE_PATTERN = re.compile(r"!\[.*?\]\((.*?)\)")


def extract_image_paths(text: str) -> list[str]:
    return _IMAGE_PATTERN.findall(text or "")


def strip_image_tags(text: str) -> str:
    cleaned = _IMAGE_PATTERN.sub("", text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _build_user_content(question: str, images: list[str], placeholder: str = "<image>") -> str:
    prefix = "".join(placeholder for _ in images)
    question_clean = strip_image_tags(question)
    return f"{prefix}{question_clean}" if prefix else question_clean


def _build_assistant_content(answer: str, solution: str) -> str:
    ans_text = (answer or "").strip()
    sol_text = strip_image_tags(solution)
    if ans_text and sol_text:
        return f"{ans_text}\n\n{sol_text}"
    if ans_text:
        return ans_text
    return sol_text


def _index_images(images_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not images_dir.is_dir():
        return index
    for path in images_dir.iterdir():
        if path.is_file():
            index[path.name] = path.resolve()
    return index


def convert_vqa_pair_to_sharegpt(
    qa: dict[str, Any],
    *,
    image_index: dict[str, Path],
    base_dir: Path,
    placeholder: str = "<image>",
) -> dict[str, Any] | None:
    question = str(qa.get("question") or "").strip()
    answer = str(qa.get("answer") or "").strip()
    solution = str(qa.get("solution") or "").strip()
    if not question:
        return None

    abs_images: list[str] = []
    for rel in extract_image_paths(question) + extract_image_paths(solution):
        name = Path(rel).name
        if name in image_index:
            abs_images.append(str(image_index[name]))
        else:
            candidate = (base_dir / rel).resolve()
            abs_images.append(str(candidate))

    assistant = _build_assistant_content(answer, solution)
    if not assistant:
        return None

    return {
        "messages": [
            {"role": "user", "content": _build_user_content(question, abs_images, placeholder)},
            {"role": "assistant", "content": assistant},
        ],
        "images": abs_images,
    }


def write_sharegpt(
    merged_pairs: list[dict[str, Any]],
    images_dir: str | Path,
    output_json: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> Path:
    """Write ShareGPT JSON list; returns output path."""
    images_path = Path(images_dir).expanduser().resolve()
    out = Path(output_json).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(base_dir).expanduser().resolve() if base_dir else out.parent
    index = _index_images(images_path)

    records: list[dict[str, Any]] = []
    for qa in merged_pairs:
        item = convert_vqa_pair_to_sharegpt(qa, image_index=index, base_dir=root)
        if item is not None:
            records.append(item)

    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
