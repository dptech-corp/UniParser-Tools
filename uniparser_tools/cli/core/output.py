from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from uniparser_tools.cli.core.errors import dir_exists_error


def default_output_dir(source_stem: str) -> Path:
    return (Path.home() / "Uni-Parser-Skill" / source_stem).expanduser().resolve()


def fetch_source_stem(token: str) -> str:
    return f"token_{token[:8]}"


def default_fetch_output_dir(token: str) -> Path:
    return (Path.home() / "Uni-Parser-Skill" / fetch_source_stem(token)).expanduser().resolve()


def _ensure_output_dir(out: Path, *, overwrite: bool) -> tuple[Path | None, int | None]:
    if out.exists() and not overwrite:
        return None, dir_exists_error(out)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    return out, None


def resolve_output_dir(
    source_stem: str,
    output_dir: str | None,
    *,
    overwrite: bool,
) -> tuple[Path | None, int | None]:
    out = Path(output_dir).expanduser().resolve() if output_dir else default_output_dir(source_stem)
    return _ensure_output_dir(out, overwrite=overwrite)


def resolve_fetch_output_dir(
    token: str,
    output_dir: str | None,
    *,
    overwrite: bool,
) -> tuple[Path | None, int | None]:
    if output_dir:
        return resolve_output_dir(fetch_source_stem(token), output_dir, overwrite=overwrite)
    return _ensure_output_dir(default_fetch_output_dir(token), overwrite=overwrite)


def write_trigger_meta(
    out_dir: Path,
    *,
    token: str,
    input_type: str,
    input_value: str,
    trigger_kwargs: dict | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "trigger_meta.json"
    payload: dict[str, Any] = {
        "token": token,
        "input_type": input_type,
        "input": input_value,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    if trigger_kwargs is not None:
        payload["trigger_kwargs"] = trigger_kwargs
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def save_parse_results(
    *,
    out_dir: Path,
    source_stem: str,
    pages_tree: dict,
    formatted: dict,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source_stem or "document"

    pages_tree_path = out_dir / "pages_tree.json"
    pages_tree_path.write_text(
        json.dumps(pages_tree, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    md_path = out_dir / f"{stem}.md"
    content = formatted.get("content", "")
    md_path.write_text(content, encoding="utf-8")

    meta = {k: v for k, v in formatted.items() if k != "content"}
    (out_dir / "formatted_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "output_dir": str(out_dir),
        "pages_tree_path": str(pages_tree_path),
        "markdown_path": str(md_path),
        "content_chars": len(content),
    }


def emit_success(summary: dict, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    print(f"Token: {summary.get('token', '')}")
    print(f"Markdown: {summary['markdown_path']}")
    print(f"Pages tree: {summary['pages_tree_path']}")
    print(f"Output directory: {summary['output_dir']}")
    if summary.get("trigger_meta_path"):
        print(f"Trigger meta: {summary['trigger_meta_path']}", file=sys.stderr)


def print_parsing_status(label: str) -> None:
    print(f"Parsing... {label}", file=sys.stderr, flush=True)
