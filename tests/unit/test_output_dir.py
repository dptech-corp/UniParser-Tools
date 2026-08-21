"""Tests for collision-safe output directory allocation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from uniparser_tools.common.output_dir import create_unique_output_dir


def test_creates_preferred_directory(tmp_path: Path) -> None:
    preferred = tmp_path / "results"

    actual = create_unique_output_dir(preferred)

    assert actual == preferred
    assert actual.is_dir()


def test_uses_first_available_suffixed_sibling(tmp_path: Path) -> None:
    preferred = tmp_path / "results"
    preferred.mkdir()
    (preferred / "keep.txt").write_text("original", encoding="utf-8")
    preferred.with_name("results_1").mkdir()

    actual = create_unique_output_dir(preferred)

    assert actual == tmp_path / "results_2"
    assert (preferred / "keep.txt").read_text(encoding="utf-8") == "original"


def test_existing_file_is_preserved_and_suffixed(tmp_path: Path) -> None:
    preferred = tmp_path / "results"
    preferred.write_text("original", encoding="utf-8")

    actual = create_unique_output_dir(preferred)

    assert actual == tmp_path / "results_1"
    assert preferred.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("broken", [False, True])
def test_final_symlink_is_not_followed(tmp_path: Path, broken: bool) -> None:
    target = tmp_path / "target"
    if not broken:
        target.mkdir()
        (target / "keep.txt").write_text("keep", encoding="utf-8")
    preferred = tmp_path / "results"
    preferred.symlink_to(target, target_is_directory=True)

    actual = create_unique_output_dir(preferred)

    assert actual == tmp_path / "results_1"
    assert preferred.is_symlink()
    if not broken:
        assert (target / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("protected", [Path.home(), Path.cwd(), Path(Path.cwd().anchor)])
def test_rejects_protected_output_targets(protected: Path) -> None:
    with pytest.raises(ValueError, match="protected output directory"):
        create_unique_output_dir(protected)


def test_rejects_git_metadata_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Git metadata"):
        create_unique_output_dir(tmp_path / ".GIT" / "results")


def test_rejects_parent_directory_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid output directory name"):
        create_unique_output_dir(tmp_path / "nested" / "..")


def test_concurrent_allocations_are_unique(tmp_path: Path) -> None:
    preferred = tmp_path / "results"

    with ThreadPoolExecutor(max_workers=8) as executor:
        actual = list(executor.map(create_unique_output_dir, [preferred] * 8))

    assert len(set(actual)) == 8
    assert {path.name for path in actual} == {"results", *(f"results_{index}" for index in range(1, 8))}
    assert all(path.is_dir() for path in actual)
