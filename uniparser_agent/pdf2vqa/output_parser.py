"""Parse LLM id-based VQA responses back into text VQA items."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _id_to_text(input_ids: str, content_list: list[dict[str, Any]], image_prefix: str = "vqa_images") -> str:
    texts: list[str] = []
    for raw_id in input_ids.replace(" ", "").split(","):
        if not raw_id:
            continue
        try:
            idx = int(raw_id)
        except ValueError:
            continue
        if idx < 0 or idx >= len(content_list):
            continue
        item = content_list[idx]
        if "text" in item and item["text"]:
            texts.append(str(item["text"]))
        elif "table_body" in item and item["table_body"]:
            texts.append(str(item["table_body"]))
        elif "img_path" in item:
            img_name = Path(str(item.get("img_path", ""))).name
            caption = item.get("image_caption") or ["image"]
            if isinstance(caption, list):
                alt = " ".join(str(c) for c in caption)
            else:
                alt = str(caption)
            texts.append(f"![{alt}]({image_prefix}/{img_name})")
    return "\n".join(texts)


def parse_llm_response(
    response: str,
    content_list: list[dict[str, Any]],
    *,
    image_prefix: str = "vqa_images",
) -> list[dict[str, Any]]:
    if "<empty>" in response and "</empty>" in response and "<vqa_pair>" not in response:
        return []

    qa_list: list[dict[str, Any]] = []
    for chapter_block in re.findall(r"<chapter>(.*?)</chapter>", response, flags=re.DOTALL):
        title_match = re.search(r"<title>(.*?)</title>", chapter_block, flags=re.DOTALL)
        chapter_title = _id_to_text(title_match.group(1).strip(), content_list, image_prefix) if title_match else ""
        for pair in re.findall(r"<vqa_pair>(.*?)</vqa_pair>", chapter_block, flags=re.DOTALL):
            q_match = re.search(r"<question>(.*?)</question>", pair, flags=re.DOTALL)
            a_match = re.search(r"<answer>(.*?)</answer>", pair, flags=re.DOTALL)
            s_match = re.search(r"<solution>(.*?)</solution>", pair, flags=re.DOTALL)
            label_match = re.search(r"<label>(.*?)</label>", pair, flags=re.DOTALL)
            if not label_match:
                continue
            if not ((q_match and label_match) or (a_match and label_match) or (s_match and label_match)):
                continue
            qa_list.append(
                {
                    "question": (_id_to_text(q_match.group(1).strip(), content_list, image_prefix) if q_match else ""),
                    "answer": a_match.group(1).strip() if a_match else "",
                    "solution": (_id_to_text(s_match.group(1).strip(), content_list, image_prefix) if s_match else ""),
                    "label": label_match.group(1).strip(),
                    "chapter_title": chapter_title,
                }
            )
    return qa_list


def write_vqa_jsonl(qa_list: list[dict[str, Any]], output_path: str | Path) -> Path:
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for qa in qa_list:
            fh.write(json.dumps(qa, ensure_ascii=False) + "\n")
    return out
