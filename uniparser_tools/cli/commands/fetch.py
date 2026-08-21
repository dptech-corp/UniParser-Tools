from __future__ import annotations

from pathlib import Path

import click

from uniparser_tools.cli.core.config import ctx_flag, make_client
from uniparser_tools.cli.core.errors import input_error, missing_token_error
from uniparser_tools.cli.core.output import (
    emit_success,
    fetch_source_stem,
    print_parsing_status,
    resolve_fetch_output_dir,
)
from uniparser_tools.cli.core.pipeline import complete_fetch


@click.command("fetch")
@click.option("--token", required=True, help="Task token from a prior successful parse trigger response")
@click.option(
    "--output-dir",
    "-o",
    help="Preferred output directory; an available suffixed sibling is used if it exists",
)
@click.pass_context
def fetch_cmd(
    ctx: click.Context,
    token: str,
    output_dir: str | None,
) -> None:
    """Poll and download results for an existing job using its token."""
    resolved_token = (token or "").strip()
    if not resolved_token:
        raise SystemExit(missing_token_error())

    client, err = make_client(ctx)
    if err is not None:
        raise SystemExit(err)

    source_stem = fetch_source_stem(resolved_token)
    try:
        out_dir = resolve_fetch_output_dir(resolved_token, output_dir)
    except (OSError, ValueError) as exc:
        raise SystemExit(input_error(str(exc))) from exc

    print_parsing_status(source_stem)
    summary = complete_fetch(
        client,
        resolved_token,
        out_dir=Path(out_dir),
        source_stem=source_stem,
    )
    if isinstance(summary, int):
        raise SystemExit(summary)

    summary["fetched_by_token"] = True
    emit_success(summary, json_output=ctx_flag(ctx, "json_output"))
    raise SystemExit(0)
