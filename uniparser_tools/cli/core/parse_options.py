from __future__ import annotations

from typing import Any

from uniparser_tools.common.constant import ParseMode, ParseModeTextual


SEMANTIC_FIELDS = ("textual", "equation", "table", "chart", "figure", "expression", "molecule")

TEXTUAL_ALIASES: dict[str, ParseModeTextual] = {
    "disable": ParseModeTextual.Disable,
    "ocr-fast": ParseModeTextual.OCRFast,
    "ocr-hq": ParseModeTextual.OCRHighQuality,
    "digital": ParseModeTextual.DigitalExported,
    "base64": ParseModeTextual.DumpBase64,
}

PARSE_MODE_ALIASES: dict[str, ParseMode] = {
    "disable": ParseMode.Disable,
    "ocr-fast": ParseMode.OCRFast,
    "ocr-hq": ParseMode.OCRHighQuality,
    "base64": ParseMode.DumpBase64,
}

TEXTUAL_CHOICES = list(TEXTUAL_ALIASES.keys())
PARSE_MODE_CHOICES = list(PARSE_MODE_ALIASES.keys())

SCIENTIFIC_PAPER_DEFAULTS: dict[str, ParseMode | ParseModeTextual] = {
    "textual": ParseModeTextual.OCRHighQuality,
    "equation": ParseMode.OCRHighQuality,
    "table": ParseMode.OCRHighQuality,
    "chart": ParseMode.DumpBase64,
    "figure": ParseMode.DumpBase64,
    "expression": ParseMode.DumpBase64,
    "molecule": ParseMode.OCRFast,
}

_TEXTUAL_ALIAS_BY_MODE = {mode: alias for alias, mode in TEXTUAL_ALIASES.items()}
_PARSE_MODE_ALIAS_BY_MODE = {mode: alias for alias, mode in PARSE_MODE_ALIASES.items()}


def resolve_trigger_kwargs(*, sync: bool, overrides: dict[str, str | None]) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(SCIENTIFIC_PAPER_DEFAULTS)
    for field, value in overrides.items():
        if value is None:
            continue
        if field == "textual":
            kwargs[field] = TEXTUAL_ALIASES[value]
        else:
            kwargs[field] = PARSE_MODE_ALIASES[value]
    kwargs["sync"] = sync
    return kwargs


def serialize_trigger_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for field in SEMANTIC_FIELDS:
        if field not in kwargs:
            continue
        value = kwargs[field]
        if field == "textual":
            serialized[field] = _TEXTUAL_ALIAS_BY_MODE[value]
        else:
            serialized[field] = _PARSE_MODE_ALIAS_BY_MODE[value]
    if "sync" in kwargs:
        serialized["sync"] = kwargs["sync"]
    return serialized
