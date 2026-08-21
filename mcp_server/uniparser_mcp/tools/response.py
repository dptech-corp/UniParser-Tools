"""Build tool responses from pipeline summaries."""

from __future__ import annotations

from uniparser_mcp.config import get_preview_chars
from uniparser_mcp.schemas import ParseSuccess


def build_message(summary: dict) -> str:
    lines = [
        "Parse complete.",
        f"Markdown: {summary['markdown_path']}",
        f"Layout: {summary['pages_tree_path']}",
        f"Output: {summary['output_dir']}",
    ]
    return "\n".join(lines)


def build_parse_success(summary: dict) -> ParseSuccess:
    content = summary.get("content", "")
    preview_limit = get_preview_chars()
    preview = content[:preview_limit] if preview_limit else ""
    return ParseSuccess(
        output_dir=summary["output_dir"],
        markdown_path=summary["markdown_path"],
        pages_tree_path=summary["pages_tree_path"],
        formatted_meta_path=summary["formatted_meta_path"],
        trigger_meta_path=summary.get("trigger_meta_path"),
        token=summary["token"],
        input_type=summary["input_type"],
        content_chars=summary["content_chars"],
        content_preview=preview,
        message=build_message(summary),
    )
