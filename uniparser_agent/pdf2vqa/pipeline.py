"""End-to-end VQA pipeline: UniParser parse → adapt → LLM extract → merge."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from uniparser_agent.llm import LLMConfig
from uniparser_agent.output_dir import create_unique_output_dir, resolve_output_dir
from uniparser_agent.parse.api_client import resolve_input
from uniparser_agent.parse.service import load_pages_tree, parse_document
from uniparser_agent.pdf2vqa.layout_adapter import adapt_pages_tree_file
from uniparser_agent.pdf2vqa.llm_client import VQALLMClient
from uniparser_agent.pdf2vqa.output_parser import parse_llm_response, write_vqa_jsonl
from uniparser_agent.pdf2vqa.pdf_merger import merge_pdfs
from uniparser_agent.pdf2vqa.prompts import build_vqa_extract_prompt
from uniparser_agent.pdf2vqa.vqa_formatter import write_sharegpt
from uniparser_agent.pdf2vqa.vqa_merger import jsonl_to_md, merge_vqa_pairs, write_merged_jsonl


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    return resolve_output_dir(output_dir, default=Path.cwd() / "vqa_out")


def _require_local_pdf(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if resolved.suffix.lower() != ".pdf":
        raise ValueError(f"{label} must be a local PDF file: {resolved}")
    return resolved


def run_vqa_pipeline(
    input_path: str | None = None,
    *,
    answer_pdf: str | None = None,
    pages_tree_path: str | None = None,
    output_dir: str | None = None,
    strict_title_match: bool = False,
    llm_config: LLMConfig | None = None,
    llm_client: VQALLMClient | None = None,
) -> dict[str, Any]:
    """Run pdf2vqa extraction.

    Primary path: ``input_path`` (pdf/url/image) → UniParser parse → extract.
    Dual PDF: ``input_path`` (question) + ``answer_pdf`` → merge → parse → extract.
    Bypass: ``pages_tree_path`` skips UniParser parse.
    """
    if answer_pdf and pages_tree_path:
        raise ValueError("Use either answer_pdf or pages_tree_path, not both.")
    if answer_pdf and not input_path:
        raise ValueError("answer_pdf requires input_path (question booklet PDF).")
    if not input_path and not pages_tree_path:
        raise ValueError("Provide input_path (pdf/url/image) or pages_tree_path.")

    started = time.time()
    pages_tree_bytes: bytes | None = None
    question_pdf: Path | None = None
    answer_path: Path | None = None
    if pages_tree_path:
        src_tree = Path(pages_tree_path).expanduser().resolve()
        if not src_tree.is_file():
            raise FileNotFoundError(f"pages_tree not found: {src_tree}")
        load_pages_tree(src_tree)
        pages_tree_bytes = src_tree.read_bytes()
    elif answer_pdf:
        assert input_path is not None
        question_pdf = _require_local_pdf(input_path, label="question PDF")
        answer_path = _require_local_pdf(answer_pdf, label="answer PDF")
    else:
        assert input_path is not None
        resolve_input(input_path)

    out = create_unique_output_dir(_resolve_output_dir(output_dir))
    return _run_vqa_pipeline_in_dir(
        out=out,
        started=started,
        input_path=input_path,
        pages_tree_bytes=pages_tree_bytes,
        question_pdf=question_pdf,
        answer_path=answer_path,
        strict_title_match=strict_title_match,
        llm_config=llm_config,
        llm_client=llm_client,
    )


def _run_vqa_pipeline_in_dir(
    *,
    out: Path,
    started: float,
    input_path: str | None,
    pages_tree_bytes: bytes | None,
    question_pdf: Path | None,
    answer_path: Path | None,
    strict_title_match: bool,
    llm_config: LLMConfig | None,
    llm_client: VQALLMClient | None,
) -> dict[str, Any]:
    parse_dir = out / "parse"
    parse_meta: dict[str, Any] = {}
    merged_pdf_path: Path | None = None

    if pages_tree_bytes is not None:
        parse_dir.mkdir(parents=True, exist_ok=True)
        dest_tree = parse_dir / "pages_tree.json"
        dest_tree.write_bytes(pages_tree_bytes)
        tree_path = dest_tree
        parse_meta = {"mode": "pages_tree", "pages_tree_path": str(tree_path)}
    else:
        assert input_path is not None
        parse_source = input_path
        if answer_path is not None:
            assert question_pdf is not None
            merge_dir = out / "merge"
            merge_dir.mkdir(parents=True, exist_ok=True)
            merged_pdf_path = merge_pdfs(
                [question_pdf, answer_path],
                merge_dir / "merged.pdf",
            )
            parse_source = str(merged_pdf_path)

        parse_result = parse_document(parse_source, output_dir=str(parse_dir))
        tree_path = Path(parse_result["pages_tree_path"])
        if answer_path is not None:
            assert question_pdf is not None and merged_pdf_path is not None
            parse_meta = {
                "mode": "dual_pdf",
                "question_pdf": str(question_pdf),
                "answer_pdf": str(answer_path),
                "merged_pdf": str(merged_pdf_path),
                "token": parse_result.get("token", ""),
                "pages_tree_path": parse_result["pages_tree_path"],
                "markdown_path": parse_result.get("markdown_path", ""),
            }
        else:
            parse_meta = {
                "mode": "parse",
                "source": input_path,
                "token": parse_result.get("token", ""),
                "pages_tree_path": parse_result["pages_tree_path"],
                "markdown_path": parse_result.get("markdown_path", ""),
            }

    load_pages_tree(tree_path)

    images_dir = out / "vqa_images"
    content_list_path = out / "llm_content_list.json"
    content_list = adapt_pages_tree_file(
        tree_path,
        content_list_path,
        images_dir=images_dir,
    )
    n_images = len(list(images_dir.glob("*"))) if images_dir.is_dir() else 0

    llm = llm_client or VQALLMClient(config=llm_config)
    system_prompt = build_vqa_extract_prompt()
    user_content = json.dumps(content_list, ensure_ascii=False)
    llm_started = time.time()
    raw_response = llm.chat(system_prompt=system_prompt, user_content=user_content)
    llm_elapsed = time.time() - llm_started

    raw_path = out / "llm_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")

    extracted = parse_llm_response(raw_response, content_list, image_prefix="vqa_images")
    extracted_path = out / "extracted_vqa.jsonl"
    write_vqa_jsonl(extracted, extracted_path)

    merged = merge_vqa_pairs(extracted, strict_title_match=strict_title_match)
    merged_jsonl = out / "merged_vqa_pairs.jsonl"
    merged_md = out / "merged_vqa_pairs.md"
    write_merged_jsonl(merged, merged_jsonl)
    jsonl_to_md(merged_jsonl, merged_md)

    sharegpt_path = out / "vqa_sharegpt.json"
    write_sharegpt(merged, images_dir, sharegpt_path, base_dir=out)

    paths: dict[str, str] = {
        "output_dir": str(out),
        "pages_tree": str(tree_path),
        "llm_content_list": str(content_list_path),
        "llm_raw_response": str(raw_path),
        "extracted_vqa": str(extracted_path),
        "merged_vqa_pairs_jsonl": str(merged_jsonl),
        "merged_vqa_pairs_md": str(merged_md),
        "vqa_images": str(images_dir),
        "vqa_sharegpt": str(sharegpt_path),
    }
    if merged_pdf_path is not None:
        paths["merged_pdf"] = str(merged_pdf_path)

    meta = {
        "parse": parse_meta,
        "llm": llm.meta(),
        "n_content_items": len(content_list),
        "n_vqa_images": n_images,
        "n_extracted": len(extracted),
        "n_merged_vqa": len(merged),
        "llm_elapsed_sec": round(llm_elapsed, 2),
        "total_elapsed_sec": round(time.time() - started, 2),
        "paths": paths,
    }
    meta_path = out / "run_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["paths"]["run_meta"] = str(meta_path)
    return meta
