"""Safe output directory resolution."""

from __future__ import annotations

from pathlib import Path

from uniparser_tools.common.output_dir import create_unique_output_dir


def default_parse_output_dir(source_stem: str) -> Path:
    """Return a contained parse output directory for ``source_stem``."""
    if (
        not source_stem
        or source_stem in {".", ".."}
        or "/" in source_stem
        or "\\" in source_stem
        or Path(source_stem).name != source_stem
    ):
        raise ValueError(f"Unsafe source name for output directory: {source_stem!r}")

    base = (Path.home() / "Uni-Parser-Skill").resolve()
    candidate = base / source_stem
    if candidate == base or not candidate.is_relative_to(base):
        raise ValueError(f"Output directory escapes the managed root: {candidate}")
    return candidate


def resolve_output_dir(output_dir: str | Path | None, *, default: Path) -> Path:
    """Resolve an explicit output path or use the supplied safe default."""
    return Path(output_dir).expanduser() if output_dir else default


__all__ = ["create_unique_output_dir", "default_parse_output_dir", "resolve_output_dir"]
