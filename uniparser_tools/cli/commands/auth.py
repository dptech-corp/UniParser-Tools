from __future__ import annotations

import click

from uniparser_tools.cli.core.credentials import (
    API_KEY_URL,
    mask_api_key,
    resolve_api_key_source,
    save_api_key,
)


@click.command("auth")
@click.option("--show", is_flag=True, help="Show API key source and masked key")
@click.option("--verify", is_flag=True, help="Verify that an API key is configured (does not call the API)")
@click.pass_context
def auth_cmd(ctx: click.Context, show: bool, verify: bool) -> None:
    """Save, inspect, or verify API key configuration."""
    resolved = resolve_api_key_source(ctx)

    if show:
        if not resolved.api_key:
            print("No API key configured.")
            print("Run 'uniparser auth' to set up your API key.")
            raise SystemExit(1)
        print(f"API key source: {resolved.source}")
        print(f"API key: {mask_api_key(resolved.api_key)}")
        raise SystemExit(0)

    if verify:
        if not resolved.api_key:
            print("No API key configured.")
            raise SystemExit(1)
        print("API key is configured.")
        print(f"  Source: {resolved.source}")
        raise SystemExit(0)

    print("UniParser API Key Setup")
    print(f"Get your API key from: {API_KEY_URL}")
    print()

    if resolved.api_key:
        print(f"Current API key source: {resolved.source}")
        entered = click.prompt(
            "Enter new API key (or press Enter to keep current)",
            default="",
            show_default=False,
        ).strip()
    else:
        entered = click.prompt("Enter your API key", type=str).strip()

    if not entered:
        if resolved.api_key:
            print("Keeping existing API key.")
            raise SystemExit(0)
        print("API key is required.")
        raise SystemExit(1)

    save_api_key(entered)
    print("API key saved successfully")
