"""Structured MCP errors."""

from __future__ import annotations

from uniparser_mcp.schemas import ErrorDetail, ErrorResult


def config_error(message: str) -> ErrorResult:
    return ErrorResult(error=ErrorDetail(code="CONFIG_ERROR", message=message))


def input_error(message: str) -> ErrorResult:
    return ErrorResult(error=ErrorDetail(code="INPUT_ERROR", message=message))


def _operation_error(code: str, stage: str, result: dict) -> ErrorResult:
    return ErrorResult(
        error=ErrorDetail(
            code=code,
            message=result.get("description") or result.get("message") or str(result),
            stage=stage,
            token=result.get("token"),
        )
    )


def parse_error(stage: str, result: dict) -> ErrorResult:
    return _operation_error("PARSE_ERROR", stage, result)


def upload_error(stage: str, result: dict) -> ErrorResult:
    return _operation_error("UPLOAD_ERROR", stage, result)


def token_not_found_error(token: str, *, attempts: int) -> ErrorResult:
    return ErrorResult(
        error=ErrorDetail(
            code="TOKEN_NOT_FOUND",
            message=f"The service did not recognize this token after {attempts} checks.",
            stage="get_result_poll",
            token=token,
        ),
    )
