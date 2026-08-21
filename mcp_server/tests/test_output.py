from pathlib import Path

import pytest
from uniparser_mcp.pipeline.output import resolve_output_dir


def test_default_output_uses_configured_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "managed"
    monkeypatch.setenv("OUTPUT_DIR", str(output_root))

    actual = resolve_output_dir("paper", None)

    assert actual == output_root / "paper"
    assert actual.is_dir()


def test_existing_output_is_preserved_and_suffixed(tmp_path: Path) -> None:
    preferred = tmp_path / "paper"
    preferred.mkdir()
    (preferred / "old.txt").write_text("old", encoding="utf-8")

    actual = resolve_output_dir("paper", str(preferred))

    assert actual == tmp_path / "paper_1"
    assert actual.is_dir()
    assert (preferred / "old.txt").read_text(encoding="utf-8") == "old"


def test_existing_file_is_preserved_and_suffixed(tmp_path: Path) -> None:
    preferred = tmp_path / "paper"
    preferred.write_text("old", encoding="utf-8")

    actual = resolve_output_dir("paper", str(preferred))

    assert actual == tmp_path / "paper_1"
    assert preferred.read_text(encoding="utf-8") == "old"
