"""Parse pipeline for uniparser_parse."""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.fastmcp import Context

from uniparser_mcp.defaults import (
    DIRECT_SYNC_UPLOAD_REQUEST_TIMEOUT,
    DIRECT_UPLOAD_REQUEST_TIMEOUT,
)
from uniparser_mcp.errors import input_error, parse_error, upload_error
from uniparser_mcp.input import InputKind, ResolvedInput, display_label, resolve_request
from uniparser_mcp.parse_options import resolve_trigger_kwargs, serialize_trigger_kwargs
from uniparser_mcp.pipeline.output import (
    resolve_output_dir,
    save_parse_results,
    save_stage_error,
    write_trigger_meta,
)
from uniparser_mcp.pipeline.poll import poll_until_success
from uniparser_mcp.schemas import ErrorResult, ParseRequest, ParseResult
from uniparser_mcp.tools.response import build_parse_success
from uniparser_tools.api.clients import UniParserClient
from uniparser_tools.common.constant import FormatFlag


async def _ctx_info(ctx: Context | None, message: str) -> None:
    if ctx is None:
        return
    if ctx.request_context:
        await ctx.info(message)


async def _trigger_input(
    client: UniParserClient,
    resolved: ResolvedInput,
    *,
    trigger_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if resolved.kind is InputKind.FILE:
        trigger = await asyncio.to_thread(
            client.trigger_file,
            str(resolved.path),
            token=None,
            server_generated_token=True,
            http_timeout=(
                DIRECT_SYNC_UPLOAD_REQUEST_TIMEOUT
                if trigger_kwargs.get("sync", True)
                else DIRECT_UPLOAD_REQUEST_TIMEOUT
            ),
            **trigger_kwargs,
        )
        if not isinstance(trigger, dict):
            trigger = {
                "status": "error",
                "message": "Direct upload returned an invalid response",
            }
        return trigger, "trigger_file"
    if resolved.kind is InputKind.IMAGE:
        trigger = await asyncio.to_thread(
            client.trigger_snip,
            str(resolved.path),
            token=None,
            server_generated_token=True,
            **trigger_kwargs,
        )
        return trigger, "trigger_snip"
    trigger = await asyncio.to_thread(
        client.trigger_url,
        resolved.raw,
        token=None,
        server_generated_token=True,
        **trigger_kwargs,
    )
    return trigger, "trigger_url"


async def _fetch_pages_tree(client: UniParserClient, token: str) -> dict[str, Any]:
    return await asyncio.to_thread(
        client.get_result,
        token,
        pages_tree=True,
        objects=False,
    )


async def _fetch_markdown(client: UniParserClient, token: str) -> dict[str, Any]:
    return await asyncio.to_thread(
        client.get_formatted,
        token,
        content=True,
        textual=FormatFlag.Markdown,
        table=FormatFlag.Markdown,
        equation=FormatFlag.Latex,
    )


async def _complete_by_token(
    client: UniParserClient,
    token: str,
    *,
    resolved: ResolvedInput,
    out_dir,
    ctx: Context | None,
    write_trigger_meta_file: bool,
    trigger_kwargs: dict[str, Any] | None = None,
) -> ParseResult | ErrorResult:
    poll_result = await poll_until_success(client, token)
    if isinstance(poll_result, ErrorResult):
        return poll_result

    pages_tree = await _fetch_pages_tree(client, token)
    if pages_tree.get("status") != "success":
        save_stage_error(out_dir, "pages_tree_error.json", pages_tree)
        return parse_error("get_result_pages_tree", pages_tree)

    formatted = await _fetch_markdown(client, token)
    if formatted.get("status") != "success":
        save_stage_error(out_dir, "formatted_error.json", formatted)
        return parse_error("get_formatted", formatted)

    summary = save_parse_results(
        out_dir=out_dir,
        source_stem=resolved.source_stem,
        pages_tree=pages_tree,
        formatted=formatted,
    )
    summary["token"] = token
    summary["input_type"] = resolved.kind.value
    if write_trigger_meta_file and trigger_kwargs is not None:
        meta_path = write_trigger_meta(
            out_dir,
            token=token,
            input_type=resolved.kind.value,
            input_value=resolved.raw,
            trigger_kwargs=serialize_trigger_kwargs(trigger_kwargs),
        )
        summary["trigger_meta_path"] = str(meta_path)
    return build_parse_success(summary)


async def run_parse(client: UniParserClient, req: ParseRequest, ctx: Context | None = None) -> ParseResult:
    resolved = resolve_request(req)
    if isinstance(resolved, str):
        return input_error(resolved)

    try:
        out_dir = resolve_output_dir(resolved.source_stem, req.output_dir)
    except (OSError, ValueError) as exc:
        return input_error(str(exc))

    trigger_kwargs = resolve_trigger_kwargs(
        sync=not req.async_mode,
        textual=req.textual,
        equation=req.equation,
        table=req.table,
        chart=req.chart,
        figure=req.figure,
        expression=req.expression,
        molecule=req.molecule,
    )

    await _ctx_info(ctx, f"Parsing {display_label(resolved)}")
    trigger, stage = await _trigger_input(client, resolved, trigger_kwargs=trigger_kwargs)

    if trigger.get("status") != "success":
        save_stage_error(out_dir, "trigger_error.json", trigger)
        if stage == "trigger_file" and trigger.get("error_type"):
            return upload_error(stage, trigger)
        return parse_error(stage, trigger)

    token = trigger.get("token")
    if not token:
        return parse_error(stage, {"status": "error", "message": "trigger response missing token"})

    return await _complete_by_token(
        client,
        token,
        resolved=resolved,
        out_dir=out_dir,
        ctx=ctx,
        write_trigger_meta_file=True,
        trigger_kwargs=trigger_kwargs,
    )
