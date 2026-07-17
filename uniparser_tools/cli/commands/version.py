from __future__ import annotations

import json

import click

from uniparser_tools.cli.core.config import ctx_flag, make_client
from uniparser_tools.cli.core.credentials import resolve_api_key_source


def format_remote_version(remote: object) -> str:
    if isinstance(remote, dict):
        if remote.get("status") is not None:
            return str(remote["status"])
        if remote.get("version") is not None:
            return str(remote["version"])
    return str(remote)


@click.command("version")
@click.pass_context
def version_cmd(ctx: click.Context) -> None:
    """Print local package version and remote service version."""
    import uniparser_tools

    local_version = uniparser_tools.__version__
    resolved = resolve_api_key_source(ctx)

    if not resolved.api_key:
        if ctx_flag(ctx, "json_output"):
            print(
                json.dumps(
                    {"local": local_version, "remote": None, "remote_skipped": True},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"uniparser-tools: {local_version}")
            print("remote: skipped (no API key)")
        raise SystemExit(0)

    client, err = make_client(ctx)
    if err is not None:
        raise SystemExit(err)

    remote = client.version()

    if ctx_flag(ctx, "json_output"):
        print(
            json.dumps(
                {"local": local_version, "remote": remote},
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    else:
        print(f"uniparser-tools: {local_version}")
        print(f"remote: {format_remote_version(remote)}")
    raise SystemExit(0)
