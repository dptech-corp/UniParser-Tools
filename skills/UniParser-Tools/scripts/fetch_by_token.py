#!/usr/bin/env python3
"""
Fetch pages_tree + Markdown for an existing UniParser job (poll until success).

Resolves the task token from --token or by computing to_token from the same input
used at submit time (--file-path / --image-path / --pdf-url).

Usage:
    python3 scripts/fetch_by_token.py --file-path "/path/to/document.pdf"
    python3 scripts/fetch_by_token.py --pdf-url "https://example.com/paper.pdf"
    python3 scripts/fetch_by_token.py --image-path "/path/to/figure.png"
    python3 scripts/fetch_by_token.py --file-path paper.pdf --output-dir ./out --overwrite
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import (  # noqa: E402
    DEFAULT_HOST,
    fetch_markdown,
    fetch_pages_tree,
    parse_error,
    poll_until_success,
    print_success,
    resolve_fetch_target,
    resolve_output_dir,
    run_startup_checks,
    save_parse_results,
)


def main() -> int:
    if (code := run_startup_checks()) is not None:
        return code

    from uniparser_tools.api.clients import UniParserClient

    parser = argparse.ArgumentParser(description="Fetch UniParser results for an existing job (by input or token)")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--token", help="UniParser task token (optional; prefer input flags)")
    input_group.add_argument("--file-path", help="Same local PDF path used in parse_document.py")
    input_group.add_argument("--image-path", help="Same local image path used in parse_document.py")
    input_group.add_argument("--pdf-url", help="Same source URL used in parse_document.py")
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory (default: ~/Uni-Parser-Skill/<source_stem>/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )
    args = parser.parse_args()

    client = UniParserClient(host=DEFAULT_HOST, api_key=os.environ["UNIPARSER_API_KEY"])

    resolved = resolve_fetch_target(
        token=args.token,
        file_path=args.file_path,
        image_path=args.image_path,
        pdf_url=args.pdf_url,
        client=client,
    )
    if isinstance(resolved, int):
        return resolved
    token, source_stem = resolved

    out_dir, dir_code = resolve_output_dir(source_stem, args.output_dir, overwrite=args.overwrite)
    if dir_code is not None:
        return dir_code

    poll_result = poll_until_success(client, token)
    if isinstance(poll_result, int):
        return poll_result

    pages_tree = fetch_pages_tree(client, token)
    if pages_tree.get("status") != "success":
        return parse_error("get_result_pages_tree", pages_tree)

    formatted = fetch_markdown(client, token)
    if formatted.get("status") != "success":
        return parse_error("get_formatted", formatted)

    summary = save_parse_results(
        out_dir=out_dir,
        source_stem=source_stem,
        pages_tree=pages_tree,
        formatted=formatted,
    )
    summary["fetched_by_token"] = True
    print_success(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
