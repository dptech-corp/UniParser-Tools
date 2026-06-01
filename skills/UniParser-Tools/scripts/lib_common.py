"""Shared helpers for UniParser skill CLI scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_HOST = "https://uniparser.dp.tech/"
INSTALL_CMD = "git+https://github.com/dptech-corp/UniParser-Tools.git"


def emit_json_stderr(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)


def config_error(message: str) -> int:
    emit_json_stderr({"ok": False, "error": {"code": "CONFIG_ERROR", "message": message}})
    return 1


def parse_error(stage: str, result: dict) -> int:
    emit_json_stderr(
        {
            "ok": False,
            "error": {
                "code": "PARSE_ERROR",
                "message": result.get("description") or result.get("message") or str(result),
                "stage": stage,
            },
            "token": result.get("token"),
        }
    )
    return 1


def get_api_key() -> str | None:
    return (os.getenv("UNIPARSER_API_KEY") or "").strip() or None


def check_api_key() -> int | None:
    if get_api_key():
        return None
    return config_error(
        "UNIPARSER_API_KEY is not set. "
        "Register at https://uniparser.dp.tech/ then set the environment variable. "
        "See SKILL.md Configuration section for platform-specific commands."
    )


def _try_import_uniparser() -> bool:
    try:
        import uniparser_tools  # noqa: F401

        return True
    except ImportError:
        return False


def ensure_uniparser_installed() -> int | None:
    if _try_import_uniparser():
        return None
    subprocess.run(
        [sys.executable, "-m", "pip", "install", INSTALL_CMD],
        check=False,
    )
    if _try_import_uniparser():
        return None
    return config_error(
        "uniparser_tools is not installed. Run once: "
        f'pip install "{INSTALL_CMD}"'
    )


def run_startup_checks() -> int | None:
    """API key + package install. Returns exit code if checks fail."""
    code = check_api_key()
    if code is not None:
        return code
    return ensure_uniparser_installed()


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return Path(tempfile.gettempdir()) / "uniparser" / "results" / f"{stamp}_{short_id}"


def scientific_paper_trigger_kwargs() -> dict:
    from uniparser_tools.common.constant import ParseMode, ParseModeTextual

    return {
        "sync": True,
        "textual": ParseModeTextual.OCRHighQuality,
        "equation": ParseMode.OCRHighQuality,
        "table": ParseMode.OCRHighQuality,
        "chart": ParseMode.DumpBase64,
        "figure": ParseMode.DumpBase64,
        "expression": ParseMode.DumpBase64,
        "molecule": ParseMode.OCRFast,
    }


def fetch_markdown(client, token: str) -> dict:
    from uniparser_tools.common.constant import FormatFlag

    return client.get_formatted(
        token,
        content=True,
        textual=FormatFlag.Markdown,
        table=FormatFlag.Markdown,
        equation=FormatFlag.Latex,
    )


def save_markdown_result(
    *,
    out_dir: Path,
    source_label: str,
    token: str,
    formatted: dict,
    client=None,
    include_pages_tree: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = Path(source_label).name
    if not safe_stem or safe_stem == ".":
        safe_stem = "document"
    stem = Path(safe_stem).stem or "document"
    md_path = out_dir / f"{stem}.md"
    content = formatted.get("content", "")
    md_path.write_text(content, encoding="utf-8")
    (out_dir / "token.txt").write_text(token, encoding="utf-8")

    meta = {k: v for k, v in formatted.items() if k != "content"}
    (out_dir / "formatted_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    pages_tree_path = None
    if include_pages_tree and client is not None:
        structured = client.get_result(token, pages_tree=True, objects=False)
        pages_tree_path = out_dir / "pages_tree.json"
        pages_tree_path.write_text(
            json.dumps(structured, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    return {
        "ok": True,
        "token": token,
        "markdown_path": str(md_path),
        "output_dir": str(out_dir),
        "content_chars": len(content),
        "pages_tree_path": str(pages_tree_path) if pages_tree_path else None,
    }


def print_success(summary: dict) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Token: {summary['token']}", file=sys.stderr)
    print(f"Markdown saved to: {summary['markdown_path']}", file=sys.stderr)
    print(f"Output directory: {summary['output_dir']}", file=sys.stderr)
    if summary.get("pages_tree_path"):
        print(f"Pages tree saved to: {summary['pages_tree_path']}", file=sys.stderr)
