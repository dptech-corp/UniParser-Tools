"""Merge extracted question/answer fragments into final VQA pairs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def refine_title(title: str, strict_title_match: bool = False) -> str:
    title = re.sub(r"\s+", "", title)
    if strict_title_match:
        return title
    try:
        return re.search(r"\d+\.\d+|\d+", title).group()  # type: ignore[union-attr]
    except Exception:
        try:
            return re.search(r"[一二三四五六七八九零十百]+", title).group()  # type: ignore[union-attr]
        except Exception:
            return title


def merge_vqa_pairs(
    extracted: list[dict[str, Any]],
    *,
    strict_title_match: bool = False,
) -> list[dict[str, Any]]:
    question_list: list[dict[str, Any]] = []
    answer_list: list[dict[str, Any]] = []
    for data in extracted:
        if data.get("question"):
            question_list.append(dict(data))
        else:
            answer_list.append(dict(data))

    merged: list[dict[str, Any]] = []
    chapter_title = ""
    label = float("inf")
    questions: dict[tuple[str, int], dict[str, Any]] = {}
    answers: dict[tuple[str, int], dict[str, Any]] = {}
    already_complete = 0

    for data in question_list:
        label_match = re.search(r"\d+", str(data.get("label", "")))
        if label_match:
            data["label"] = label_match.group()
        if not data.get("chapter_title"):
            data["chapter_title"] = chapter_title
        try:
            data["label"] = int(data["label"])
        except Exception:
            continue

        if data["chapter_title"] and data["chapter_title"] != chapter_title:
            if data["label"] < label:
                chapter_title = data["chapter_title"]
            else:
                data["chapter_title"] = chapter_title
        label = data["label"]
        data["chapter_title"] = refine_title(data["chapter_title"], strict_title_match)

        if data["label"] <= 0:
            continue
        if data.get("answer") or data.get("solution"):
            already_complete += 1
            merged.append(
                {
                    "question_chapter_title": data["chapter_title"],
                    "answer_chapter_title": data["chapter_title"],
                    "label": data["label"],
                    "question": data["question"],
                    "answer": data.get("answer", ""),
                    "solution": data.get("solution", ""),
                }
            )
        else:
            questions[(data["chapter_title"], data["label"])] = data

    chapter_title = ""
    label = float("inf")
    for data in answer_list:
        label_match = re.search(r"\d+", str(data.get("label", "")))
        if label_match:
            data["label"] = label_match.group()
        if not data.get("chapter_title"):
            data["chapter_title"] = chapter_title
        try:
            data["label"] = int(data["label"])
        except Exception:
            continue

        if data["chapter_title"] and data["chapter_title"] != chapter_title:
            if data["label"] < label:
                chapter_title = data["chapter_title"]
            else:
                data["chapter_title"] = chapter_title
        label = data["label"]
        data["chapter_title"] = refine_title(data["chapter_title"], strict_title_match)

        if data["label"] <= 0:
            continue
        key = (data["chapter_title"], data["label"])
        existing = answers.get(key)
        if not existing:
            answers[key] = data
        else:
            if not existing.get("solution") and data.get("solution"):
                existing["solution"] = data["solution"]
            if not existing.get("answer") and data.get("answer"):
                existing["answer"] = data["answer"]

    for key, qdata in questions.items():
        if key not in answers:
            continue
        adata = answers[key]
        merged.append(
            {
                "question_chapter_title": qdata["chapter_title"],
                "answer_chapter_title": adata["chapter_title"],
                "label": key[1],
                "question": qdata["question"],
                "answer": adata.get("answer", ""),
                "solution": adata.get("solution", ""),
            }
        )

    return merged


def write_merged_jsonl(merged: list[dict[str, Any]], output_path: str | Path) -> Path:
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for item in merged:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return out


def jsonl_to_md(jsonl_path: str | Path, md_path: str | Path) -> Path:
    src = Path(jsonl_path).expanduser().resolve()
    out = Path(md_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as infile, out.open("w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            outfile.write(f"### Question {data['label']}\n\n")
            outfile.write(f"{data['question']}\n\n")
            outfile.write(f"**Answer:** {data['answer']}\n\n")
            if data.get("solution"):
                outfile.write(f"**Solution:**\n\n{data['solution']}\n\n")
            outfile.write("---\n\n")
    return out
