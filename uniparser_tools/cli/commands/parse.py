from __future__ import annotations

import click

from uniparser_tools.cli.core.config import ctx_flag, make_client
from uniparser_tools.cli.core.errors import input_error
from uniparser_tools.cli.core.input import resolve_input
from uniparser_tools.cli.core.output import emit_success, resolve_output_dir
from uniparser_tools.cli.core.parse_options import PARSE_MODE_CHOICES, TEXTUAL_CHOICES, resolve_trigger_kwargs
from uniparser_tools.cli.core.pipeline import run_parse


def _parse_mode_option(name: str, help_text: str):
    return click.option(
        f"--{name}",
        type=click.Choice(PARSE_MODE_CHOICES, case_sensitive=False),
        default=None,
        help=help_text,
    )


@click.command("parse")
@click.argument("source", metavar="INPUT")
@click.option("--output-dir", "-o", help="Output directory (default: ~/Uni-Parser-Skill/<stem>/)")
@click.option("--overwrite", is_flag=True, help="Overwrite output directory if it already exists")
@click.option("--async", "async_mode", is_flag=True, help="Submit with sync=false and poll until success")
@click.option(
    "--textual",
    type=click.Choice(TEXTUAL_CHOICES, case_sensitive=False),
    default=None,
    help="Text parsing mode (default: ocr-hq).",
)
@_parse_mode_option("equation", "Equation parsing mode (default: ocr-hq).")
@_parse_mode_option("table", "Table parsing mode (default: ocr-hq).")
@_parse_mode_option("chart", "Chart parsing mode (default: base64).")
@_parse_mode_option("figure", "Figure parsing mode (default: base64).")
@_parse_mode_option("expression", "Chemical expression parsing mode (default: base64).")
@_parse_mode_option("molecule", "Molecule parsing mode (default: ocr-fast).")
@click.pass_context
def parse_cmd(
    ctx: click.Context,
    source: str,
    output_dir: str | None,
    overwrite: bool,
    async_mode: bool,
    textual: str | None,
    equation: str | None,
    table: str | None,
    chart: str | None,
    figure: str | None,
    expression: str | None,
    molecule: str | None,
) -> None:
    """Parse a local PDF/image or public PDF URL; save Markdown and pages_tree."""
    resolved = resolve_input(source)
    if isinstance(resolved, str):
        raise SystemExit(input_error(resolved))

    client, err = make_client(ctx)
    if err is not None:
        raise SystemExit(err)

    out_dir, dir_code = resolve_output_dir(resolved.source_stem, output_dir, overwrite=overwrite)
    if dir_code is not None:
        raise SystemExit(dir_code)

    trigger_kwargs = resolve_trigger_kwargs(
        sync=not async_mode,
        overrides={
            "textual": textual,
            "equation": equation,
            "table": table,
            "chart": chart,
            "figure": figure,
            "expression": expression,
            "molecule": molecule,
        },
    )

    code = run_parse(
        client,
        resolved,
        out_dir=out_dir,
        trigger_kwargs=trigger_kwargs,
    )
    if isinstance(code, int):
        raise SystemExit(code)

    emit_success(code, json_output=ctx_flag(ctx, "json_output"))
    raise SystemExit(0)
