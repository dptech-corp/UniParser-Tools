from __future__ import annotations

import json

import click

from uniparser_tools.cli.core.config import ctx_flag, make_client


@click.command("health")
@click.pass_context
def health_cmd(ctx: click.Context) -> None:
    """Check UniParser service health."""
    client, err = make_client(ctx)
    if err is not None:
        raise SystemExit(err)

    result = client.health()
    if ctx_flag(ctx, "json_output"):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        status = result.get("status", result)
        print(f"Health: {status}")
    raise SystemExit(0 if result.get("status") != "error" else 1)
