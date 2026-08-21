"""Parse artifact persistence and polling."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from uniparser_agent.parse.api_client import PENDING_STATUSES, UniParserApiClient


POLL_INTERVAL_SEC = 3
POLL_TIMEOUT_SEC = 1800


def write_trigger_meta(
    out_dir: Path,
    *,
    token: str,
    input_type: str,
    input_value: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "trigger_meta.json"
    payload = {
        "token": token,
        "input_type": input_type,
        "input": input_value,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "parse_preset": "scientific-paper",
    }
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def save_parse_results(
    *,
    out_dir: Path,
    source_stem: str,
    pages_tree: dict[str, Any],
    formatted: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source_stem or "document"

    pages_tree_path = out_dir / "pages_tree.json"
    pages_tree_path.write_text(
        json.dumps(pages_tree, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    md_path = out_dir / f"{stem}.md"
    content = formatted.get("content", "")
    md_path.write_text(content, encoding="utf-8")

    meta = {k: v for k, v in formatted.items() if k != "content"}
    (out_dir / "formatted_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return {
        "output_dir": str(out_dir),
        "pages_tree_path": str(pages_tree_path),
        "markdown_path": str(md_path),
        "content_chars": len(content),
    }


def poll_until_success(client: UniParserApiClient, token: str) -> dict[str, Any]:
    deadline = time.time() + POLL_TIMEOUT_SEC
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = client.get_result(token, pages_tree=False)
        status = last.get("status")
        if status == "success":
            return last
        if status == "error":
            return last
        if status in PENDING_STATUSES or status is None:
            time.sleep(POLL_INTERVAL_SEC)
            continue
        return last
    return {
        "status": "error",
        "description": f"Timed out after {POLL_TIMEOUT_SEC}s waiting for parsing to finish.",
        "token": token,
        "last_status": last.get("status"),
    }


def complete_parse_job(
    client: UniParserApiClient,
    token: str,
    *,
    out_dir: Path,
    source_stem: str,
) -> dict[str, Any]:
    poll_result = poll_until_success(client, token)
    if poll_result.get("status") != "success":
        save_stage_error(out_dir, "poll_error.json", poll_result)
        raise RuntimeError(poll_result.get("description") or poll_result.get("message") or "poll failed")

    pages_tree = client.get_result(token, pages_tree=True)
    if pages_tree.get("status") != "success":
        save_stage_error(out_dir, "pages_tree_error.json", pages_tree)
        raise RuntimeError(pages_tree.get("description") or "get_result pages_tree failed")

    formatted = client.get_formatted(token)
    if formatted.get("status") != "success":
        save_stage_error(out_dir, "formatted_error.json", formatted)
        raise RuntimeError(formatted.get("description") or "get_formatted failed")

    summary = save_parse_results(
        out_dir=out_dir,
        source_stem=source_stem,
        pages_tree=pages_tree,
        formatted=formatted,
    )
    summary["token"] = token
    return summary


def save_stage_error(out_dir: Path, filename: str, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
