#!/usr/bin/env python3
"""
Parse a local PDF, local image, or public PDF URL with UniParser-Tools.

Usage:
    python3 scripts/parse_document.py --file-path document.pdf
    python3 scripts/parse_document.py --image-path figure.png
    python3 scripts/parse_document.py --pdf-url "https://example.com/paper.pdf"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Shared helpers live next to this script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import (  # noqa: E402
    DEFAULT_HOST,
    config_error,
    default_output_dir,
    fetch_markdown,
    parse_error,
    print_success,
    run_startup_checks,
    save_markdown_result,
    scientific_paper_trigger_kwargs,
)


def main() -> int:
    # --- Startup checks (API key + package install) ---
    if (code := run_startup_checks()) is not None:
        return code

    from uniparser_tools.api.clients import UniParserClient

    parser = argparse.ArgumentParser(
        description="Parse documents with UniParser-Tools (https://uniparser.dp.tech/)"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file-path", help="Local PDF file path")
    input_group.add_argument("--image-path", help="Local image path (snippet)")
    input_group.add_argument("--pdf-url", help="Public PDF URL")
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory (default: system temp under uniparser/results/)",
    )
    args = parser.parse_args()

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir()
    )

    import os

    client = UniParserClient(host=DEFAULT_HOST, api_key=os.environ["UNIPARSER_API_KEY"])
    trigger_kwargs = scientific_paper_trigger_kwargs()

    if args.file_path:
        path = Path(args.file_path).expanduser().resolve()
        if not path.is_file():
            return config_error(f"File not found: {path}")
        trigger = client.trigger_file(file_path=str(path), **trigger_kwargs)
        source_label = path.name
        stage = "trigger_file"
    elif args.image_path:
        path = Path(args.image_path).expanduser().resolve()
        if not path.is_file():
            return config_error(f"Image not found: {path}")
        trigger = client.trigger_snip(snip_path=str(path), **trigger_kwargs)
        source_label = path.name
        stage = "trigger_snip"
    else:
        trigger = client.trigger_url(pdf_url=args.pdf_url, **trigger_kwargs)
        source_label = Path(args.pdf_url).name or "url_document"
        stage = "trigger_url"

    if trigger.get("status") != "success":
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trigger_error.json").write_text(
            json.dumps(trigger, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return parse_error(stage, trigger)

    token = trigger["token"]
    formatted = fetch_markdown(client, token)
    if formatted.get("status") != "success":
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "formatted_error.json").write_text(
            json.dumps(formatted, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return parse_error("get_formatted", formatted)

    summary = save_markdown_result(
        out_dir=out_dir,
        source_label=source_label,
        token=token,
        formatted=formatted,
        include_pages_tree=False,
    )
    summary["input_type"] = stage.replace("trigger_", "")
    print_success(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
