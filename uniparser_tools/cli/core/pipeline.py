from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from uniparser_tools.cli.core.defaults import PENDING_STATUSES, POLL_INTERVAL_SEC, POLL_TIMEOUT_SEC
from uniparser_tools.cli.core.errors import parse_error
from uniparser_tools.cli.core.input import InputKind, ResolvedInput, display_label_for_input
from uniparser_tools.cli.core.output import print_parsing_status, save_parse_results, write_trigger_meta
from uniparser_tools.cli.core.parse_options import resolve_trigger_kwargs, serialize_trigger_kwargs


def scientific_paper_trigger_kwargs(*, sync: bool = True) -> dict:
    return resolve_trigger_kwargs(sync=sync, overrides={})


def trigger_input(client, resolved: ResolvedInput, *, trigger_kwargs: dict) -> tuple[dict, str]:
    kwargs = trigger_kwargs
    if resolved.kind is InputKind.FILE:
        trigger = client.trigger_file(file_path=str(resolved.path), **kwargs)
        return trigger, "trigger_file"
    if resolved.kind is InputKind.IMAGE:
        trigger = client.trigger_snip(snip_path=str(resolved.path), **kwargs)
        return trigger, "trigger_snip"
    trigger = client.trigger_url(pdf_url=resolved.raw, **kwargs)
    return trigger, "trigger_url"


def poll_until_success(client, token: str) -> dict | int:
    deadline = time.time() + POLL_TIMEOUT_SEC
    last: dict[str, Any] = {}

    while time.time() < deadline:
        last = client.get_result(
            token,
            content=False,
            objects=False,
            pages_dict=False,
            pages_tree=False,
        )
        status = last.get("status")
        if status == "success":
            return last
        if status == "error":
            return parse_error("get_result_poll", last)
        if status in PENDING_STATUSES or status is None:
            time.sleep(POLL_INTERVAL_SEC)
            continue
        return parse_error("get_result_poll", last)

    return parse_error(
        "get_result_poll",
        {
            "status": "error",
            "description": f"Timed out after {POLL_TIMEOUT_SEC}s waiting for parsing to finish.",
            "token": token,
            "last_status": last.get("status"),
        },
    )


def fetch_pages_tree(client, token: str) -> dict:
    return client.get_result(token, pages_tree=True, objects=False)


def fetch_markdown(client, token: str) -> dict:
    from uniparser_tools.common.constant import FormatFlag

    return client.get_formatted(
        token,
        content=True,
        textual=FormatFlag.Markdown,
        table=FormatFlag.Markdown,
        equation=FormatFlag.Latex,
    )


def save_stage_error(out_dir: Path, filename: str, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def complete_fetch(
    client,
    token: str,
    *,
    out_dir: Path,
    source_stem: str,
) -> dict[str, Any] | int:
    poll_result = poll_until_success(client, token)
    if isinstance(poll_result, int):
        return poll_result

    pages_tree = fetch_pages_tree(client, token)
    if pages_tree.get("status") != "success":
        save_stage_error(out_dir, "pages_tree_error.json", pages_tree)
        return parse_error("get_result_pages_tree", pages_tree)

    formatted = fetch_markdown(client, token)
    if formatted.get("status") != "success":
        save_stage_error(out_dir, "formatted_error.json", formatted)
        return parse_error("get_formatted", formatted)

    summary = save_parse_results(
        out_dir=out_dir,
        source_stem=source_stem,
        pages_tree=pages_tree,
        formatted=formatted,
    )
    summary["token"] = token
    return summary


def run_parse(
    client,
    resolved: ResolvedInput,
    *,
    out_dir: Path,
    trigger_kwargs: dict,
) -> dict[str, Any] | int:
    print_parsing_status(display_label_for_input(resolved))
    trigger, stage = trigger_input(client, resolved, trigger_kwargs=trigger_kwargs)
    if trigger.get("status") != "success":
        save_stage_error(out_dir, "trigger_error.json", trigger)
        return parse_error(stage, trigger)

    token = trigger.get("token")
    if not token:
        return parse_error(stage, {"status": "error", "message": "trigger response missing token"})

    meta_path = write_trigger_meta(
        out_dir,
        token=token,
        input_type=resolved.kind.value,
        input_value=resolved.raw,
        trigger_kwargs=serialize_trigger_kwargs(trigger_kwargs),
    )

    summary = complete_fetch(
        client,
        token,
        out_dir=out_dir,
        source_stem=resolved.source_stem,
    )
    if isinstance(summary, int):
        return summary

    summary["input_type"] = resolved.kind.value
    summary["trigger_meta_path"] = str(meta_path)
    return summary
