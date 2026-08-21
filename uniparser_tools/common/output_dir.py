"""Collision-safe output directory allocation."""

from __future__ import annotations

from pathlib import Path


def _absolute_without_following_final_symlink(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    return raw.parent.resolve() / raw.name


def _validate_output_target(target: Path) -> None:
    protected = {
        Path(target.anchor).resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
    }
    if any(target == path or path.is_relative_to(target) for path in protected):
        raise ValueError(f"Refusing to use protected output directory: {target}")
    if target.name in {"", ".", ".."}:
        raise ValueError(f"Invalid output directory name: {target}")
    if any(part.casefold() == ".git" for part in target.parts):
        raise ValueError(f"Refusing to use Git metadata as an output directory: {target}")


def create_unique_output_dir(preferred: str | Path) -> Path:
    """Atomically create ``preferred`` or the first available suffixed sibling.

    Existing paths are never reused, followed, modified, or removed. For example,
    if ``results`` and ``results_1`` already exist, ``results_2`` is created.
    """
    target = _absolute_without_following_final_symlink(preferred)
    _validate_output_target(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    index = 0
    while True:
        candidate = target if index == 0 else target.with_name(f"{target.name}_{index}")
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            index += 1
            continue
        return candidate
