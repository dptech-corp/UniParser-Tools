from __future__ import annotations

import json
import sys


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


def _operation_error(code: str, stage: str, result: dict) -> int:
    payload = {
        "ok": False,
        "token": result.get("token"),
        "error": {
            "code": code,
            "message": result.get("description") or result.get("message") or str(result),
            "stage": stage,
        },
    }
    emit_json_stderr(payload)
    return 1


def parse_error(stage: str, result: dict) -> int:
    return _operation_error("PARSE_ERROR", stage, result)


def upload_error(stage: str, result: dict) -> int:
    return _operation_error("UPLOAD_ERROR", stage, result)


def token_not_found_error(token: str, *, attempts: int) -> int:
    emit_json_stderr(
        {
            "ok": False,
            "token": token,
            "error": {
                "code": "TOKEN_NOT_FOUND",
                "message": f"The service did not recognize this token after {attempts} checks.",
                "stage": "get_result_poll",
            },
        }
    )
    return 1
