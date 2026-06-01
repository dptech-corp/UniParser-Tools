#!/usr/bin/env python3
"""
Fetch formatted Markdown (and optionally pages_tree) by an existing UniParser token.

Usage:
    python3 scripts/fetch_by_token.py --token "abc123..."
    python3 scripts/fetch_by_token.py --token "abc123..." --output-dir ./out --pages-tree
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
)


def main() -> int:
    if (code := run_startup_checks()) is not None:
        return code

    from uniparser_tools.api.clients import UniParserClient

    parser = argparse.ArgumentParser(description="Fetch UniParser results by token")
    parser.add_argument("--token", required=True, help="UniParser task token")
    parser.add_argument("--output-dir", "-o", help="Output directory")
    parser.add_argument(
        "--pages-tree",
        action="store_true",
        help="Also save pages_tree.json (large; off by default)",
    )
    args = parser.parse_args()

    token = args.token.strip()
    if not token:
        return config_error("Token must not be empty.")

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir()
    )

    import os

    client = UniParserClient(host=DEFAULT_HOST, api_key=os.environ["UNIPARSER_API_KEY"])
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
        source_label=f"token_{token[:8]}",
        token=token,
        formatted=formatted,
        client=client,
        include_pages_tree=args.pages_tree,
    )
    summary["fetched_by_token"] = True
    print_success(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
