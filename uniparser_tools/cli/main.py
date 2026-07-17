from __future__ import annotations

import click

from uniparser_tools.cli.commands.auth import auth_cmd
from uniparser_tools.cli.commands.fetch import fetch_cmd
from uniparser_tools.cli.commands.health import health_cmd
from uniparser_tools.cli.commands.parse import parse_cmd
from uniparser_tools.cli.commands.version import version_cmd


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--api-key",
    envvar="UNIPARSER_API_KEY",
    help="API key (X-API-Key); overrides env and ~/.uniparser/config.yaml",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Machine-readable JSON on stdout (must appear before the subcommand)",
)
@click.pass_context
def cli(ctx: click.Context, api_key: str | None, json_output: bool) -> None:
    """UniParser CLI — parse documents via https://uniparser.dp.tech/"""
    ctx.ensure_object(dict)
    ctx.obj["api_key"] = api_key
    ctx.obj["json_output"] = json_output


cli.add_command(auth_cmd)
cli.add_command(parse_cmd)
cli.add_command(fetch_cmd)
cli.add_command(health_cmd)
cli.add_command(version_cmd)
