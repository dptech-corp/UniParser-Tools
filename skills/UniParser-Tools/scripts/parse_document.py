#!/usr/bin/env python3
"""
Parse a local PDF, local image, or supported source URL with UniParser-Tools.

Flow: submit → token → poll get_result until success → save pages_tree + Markdown.

Usage:
    python3 scripts/parse_document.py --file-path document.pdf
    python3 scripts/parse_document.py --pdf-url "https://example.com/paper.pdf"
    python3 scripts/parse_document.py --file-path paper.pdf --async
    python3 scripts/parse_document.py --file-path paper.pdf --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import (  # noqa: E402
    DEFAULT_HOST,
    config_error,
    fetch_markdown,
    fetch_pages_tree,
    parse_error,
    poll_until_success,
    print_success,
    resolve_output_dir,
    run_startup_checks,
    save_parse_results,
    scientific_paper_trigger_kwargs,
    source_stem_from_path,
    source_stem_from_url,
)


def main() -> int:
    if (code := run_startup_checks()) is not None:
        return code

    from uniparser_tools.api.clients import UniParserClient

    parser = argparse.ArgumentParser(description="Parse documents with UniParser-Tools (https://uniparser.dp.tech/)")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file-path", help="Local PDF file path")
    input_group.add_argument("--image-path", help="Local image path (snippet)")
    input_group.add_argument("--pdf-url", help="Source URL: http(s), s3, oss, tos, or file")
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory (default: ~/Uni-Parser-Skill/<source_stem>/)",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Submit with sync=false and poll get_result until success (default: sync=true)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists (use after user confirms)",
    )
    args = parser.parse_args()

    if args.file_path:
        path = Path(args.file_path).expanduser().resolve()
        if not path.is_file():
            return config_error(f"File not found: {path}")
        source_stem = source_stem_from_path(path)
        input_type = "file"
    elif args.image_path:
        path = Path(args.image_path).expanduser().resolve()
        if not path.is_file():
            return config_error(f"Image not found: {path}")
        source_stem = source_stem_from_path(path)
        input_type = "image"
    else:
        source_stem = source_stem_from_url(args.pdf_url)
        input_type = "url"

    out_dir, dir_code = resolve_output_dir(source_stem, args.output_dir, overwrite=args.overwrite)
    if dir_code is not None:
        return dir_code

    client = UniParserClient(host=DEFAULT_HOST, api_key=os.environ["UNIPARSER_API_KEY"])
    trigger_kwargs = scientific_paper_trigger_kwargs(sync=not args.async_mode)

    if args.file_path:
        trigger = client.trigger_file(file_path=str(path), **trigger_kwargs)
        stage = "trigger_file"
    elif args.image_path:
        trigger = client.trigger_snip(snip_path=str(path), **trigger_kwargs)
        stage = "trigger_snip"
    else:
        trigger = client.trigger_url(pdf_url=args.pdf_url, **trigger_kwargs)
        stage = "trigger_url"

    if trigger.get("status") != "success":
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trigger_error.json").write_text(
            json.dumps(trigger, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return parse_error(stage, trigger)

    token = trigger["token"]
    poll_result = poll_until_success(client, token)
    if isinstance(poll_result, int):
        out_dir.mkdir(parents=True, exist_ok=True)
        return poll_result

    pages_tree = fetch_pages_tree(client, token)
    if pages_tree.get("status") != "success":
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pages_tree_error.json").write_text(
            json.dumps(pages_tree, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return parse_error("get_result_pages_tree", pages_tree)

    formatted = fetch_markdown(client, token)
    if formatted.get("status") != "success":
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "formatted_error.json").write_text(
            json.dumps(formatted, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return parse_error("get_formatted", formatted)

    summary = save_parse_results(
        out_dir=out_dir,
        source_stem=source_stem,
        pages_tree=pages_tree,
        formatted=formatted,
    )
    summary["input_type"] = input_type
    print_success(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
