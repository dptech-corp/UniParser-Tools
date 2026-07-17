from __future__ import annotations

import json
import sys
from pathlib import Path


def emit_json_stderr(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)


def config_error(message: str) -> int:
    emit_json_stderr({"ok": False, "error": {"code": "CONFIG_ERROR", "message": message}})
    return 1


def input_error(message: str) -> int:
    emit_json_stderr({"ok": False, "error": {"code": "INPUT_ERROR", "message": message}})
    return 1


def missing_token_error() -> int:
    emit_json_stderr(
        {
            "ok": False,
            "error": {
                "code": "MISSING_TOKEN",
                "message": "--token is required. Use the token from a prior successful parse trigger response.",
            },
        }
    )
    return 1


def dir_exists_error(output_dir: Path) -> int:
    emit_json_stderr(
        {
            "ok": False,
            "error": {
                "code": "DIR_EXISTS",
                "message": (
                    f"Output directory already exists: {output_dir}. Re-run with --overwrite if you want to replace it."
                ),
                "output_dir": str(output_dir),
            },
        }
    )
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
