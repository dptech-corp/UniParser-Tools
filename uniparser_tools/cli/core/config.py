from __future__ import annotations

import click

from uniparser_tools.cli.core.credentials import resolve_api_key_source
from uniparser_tools.cli.core.defaults import UNIPARSER_BASE_URL
from uniparser_tools.cli.core.errors import config_error


def resolve_host() -> str:
    return UNIPARSER_BASE_URL.rstrip("/") + "/"


def ctx_flag(ctx: click.Context, name: str) -> bool:
    return bool((ctx.obj or {}).get(name))


def make_client(ctx: click.Context):
    api_key = resolve_api_key_source(ctx).api_key
    if not api_key:
        return None, config_error("No API key found. Run 'uniparser auth' to configure your API key.")
    try:
        from uniparser_tools.api.clients import UniParserClient
    except ImportError:
        return None, config_error(
            'uniparser_tools is not installed. Run: pip install "git+https://github.com/dptech-corp/UniParser-Tools.git"'
        )
    return UniParserClient(host=resolve_host(), api_key=api_key), None
