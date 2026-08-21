"""Parse mode aliases aligned with the uniparser CLI."""

from __future__ import annotations

from typing import Any

from uniparser_mcp.schemas import ParseModeChoice, TextualChoice
from uniparser_tools.common.constant import ParseMode, ParseModeTextual


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

SCIENTIFIC_PAPER_DEFAULTS: dict[str, ParseMode | ParseModeTextual] = {
    "textual": ParseModeTextual.OCRHighQuality,
    "equation": ParseMode.OCRHighQuality,
    "table": ParseMode.OCRHighQuality,
    "chart": ParseMode.DumpBase64,
    "figure": ParseMode.DumpBase64,
    "expression": ParseMode.DumpBase64,
    "molecule": ParseMode.OCRFast,
}


def resolve_trigger_kwargs(
    *,
    sync: bool,
    textual: TextualChoice,
    equation: ParseModeChoice,
    table: ParseModeChoice,
    chart: ParseModeChoice,
    figure: ParseModeChoice,
    expression: ParseModeChoice,
    molecule: ParseModeChoice,
) -> dict[str, Any]:
    return {
        "textual": TEXTUAL_ALIASES[textual.value],
        "equation": PARSE_MODE_ALIASES[equation.value],
        "table": PARSE_MODE_ALIASES[table.value],
        "chart": PARSE_MODE_ALIASES[chart.value],
        "figure": PARSE_MODE_ALIASES[figure.value],
        "expression": PARSE_MODE_ALIASES[expression.value],
        "molecule": PARSE_MODE_ALIASES[molecule.value],
        "sync": sync,
    }


def serialize_trigger_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    textual_by_mode = {mode: alias for alias, mode in TEXTUAL_ALIASES.items()}
    parse_by_mode = {mode: alias for alias, mode in PARSE_MODE_ALIASES.items()}
    serialized: dict[str, Any] = {}
    for field in ("textual", "equation", "table", "chart", "figure", "expression", "molecule"):
        if field not in kwargs:
            continue
        value = kwargs[field]
        if field == "textual":
            serialized[field] = textual_by_mode[value]
        else:
            serialized[field] = parse_by_mode[value]
    if "sync" in kwargs:
        serialized["sync"] = kwargs["sync"]
    return serialized
