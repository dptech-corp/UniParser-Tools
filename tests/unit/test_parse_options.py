"""Unit tests for parse trigger option resolution."""

from __future__ import annotations

from uniparser_tools.cli.core.parse_options import (
    PARSE_MODE_ALIASES,
    SCIENTIFIC_PAPER_DEFAULTS,
    TEXTUAL_ALIASES,
    resolve_trigger_kwargs,
    serialize_trigger_kwargs,
)
from uniparser_tools.common.constant import ParseMode, ParseModeTextual


class TestResolveTriggerKwargs:
    def test_defaults_match_scientific_paper(self) -> None:
        kwargs = resolve_trigger_kwargs(sync=True, overrides={})
        assert kwargs["sync"] is True
        assert kwargs["textual"] is ParseModeTextual.OCRHighQuality
        assert kwargs["equation"] is ParseMode.OCRHighQuality
        assert kwargs["table"] is ParseMode.OCRHighQuality
        assert kwargs["chart"] is ParseMode.DumpBase64
        assert kwargs["figure"] is ParseMode.DumpBase64
        assert kwargs["expression"] is ParseMode.DumpBase64
        assert kwargs["molecule"] is ParseMode.OCRFast

    def test_override_single_field(self) -> None:
        kwargs = resolve_trigger_kwargs(sync=False, overrides={"molecule": "disable"})
        assert kwargs["sync"] is False
        assert kwargs["molecule"] is ParseMode.Disable
        assert kwargs["table"] is SCIENTIFIC_PAPER_DEFAULTS["table"]

    def test_serialize_round_trip(self) -> None:
        kwargs = resolve_trigger_kwargs(
            sync=True,
            overrides={"textual": "digital", "molecule": "disable"},
        )
        serialized = serialize_trigger_kwargs(kwargs)
        assert serialized == {
            "textual": "digital",
            "equation": "ocr-hq",
            "table": "ocr-hq",
            "chart": "base64",
            "figure": "base64",
            "expression": "base64",
            "molecule": "disable",
            "sync": True,
        }

    def test_all_aliases_map_to_enums(self) -> None:
        for alias, mode in TEXTUAL_ALIASES.items():
            kwargs = resolve_trigger_kwargs(sync=True, overrides={"textual": alias})
            assert kwargs["textual"] is mode
        for alias, mode in PARSE_MODE_ALIASES.items():
            kwargs = resolve_trigger_kwargs(sync=True, overrides={"table": alias})
            assert kwargs["table"] is mode
