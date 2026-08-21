from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from uniparser_agent.output_dir import (
    create_unique_output_dir,
    default_parse_output_dir,
    resolve_output_dir,
)
from uniparser_agent.parse.api_client import UniParserApiClient, resolve_input
from uniparser_agent.parse.config import get_api_key, get_base_url
from uniparser_agent.parse.storage import (
    complete_parse_job,
    save_stage_error,
    write_trigger_meta,
)


def make_client() -> UniParserApiClient:
    return UniParserApiClient(get_base_url(), get_api_key())


def parse_document(
    input_path: str,
    *,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Parse with scientific-paper defaults via HTTP API (no OpenCV)."""
    kind, source_stem, path = resolve_input(input_path)
    default_out = default_parse_output_dir(source_stem)
    preferred = resolve_output_dir(output_dir, default=default_out)
    client = make_client()

    try:
        out = create_unique_output_dir(preferred)
        if kind == "file":
            trigger = client.trigger_file(str(path))
            input_type = "file"
        elif kind == "image":
            trigger = client.trigger_snip(str(path))
            input_type = "image"
        else:
            trigger = client.trigger_url(input_path)
            input_type = "url"

        if trigger.get("status") != "success":
            save_stage_error(out, "trigger_error.json", trigger)
            raise RuntimeError(trigger.get("message") or trigger.get("description") or "trigger failed")

        token = trigger.get("token")
        if not token:
            raise RuntimeError("trigger response missing token")

        meta_path = write_trigger_meta(
            out,
            token=token,
            input_type=input_type,
            input_value=input_path,
        )

        summary = complete_parse_job(client, token, out_dir=out, source_stem=source_stem)
        return {
            "output_dir": summary["output_dir"],
            "pages_tree_path": summary["pages_tree_path"],
            "markdown_path": summary["markdown_path"],
            "token": summary.get("token", ""),
            "input_type": input_type,
            "source_stem": source_stem,
            "source": input_path,
            "trigger_meta_path": str(meta_path),
        }
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def load_pages_tree(pages_tree_path: str | Path) -> dict[str, Any]:
    path = Path(pages_tree_path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if "pages_tree" not in data:
        raise ValueError(f"Invalid pages_tree file (missing pages_tree key): {path}")
    return data
