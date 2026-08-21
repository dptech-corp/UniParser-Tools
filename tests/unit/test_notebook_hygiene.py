"""Regression checks for customer-facing examples and documentation."""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_TOKEN_PATTERN = re.compile(
    r"(?i)(?<![0-9a-f])(?:[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})(?![0-9a-f])"
)
HARDCODED_API_KEY_PATTERN = re.compile(r"""(?i)\b(?:api_key|UNIPARSER_API_KEY)\s*=\s*["'][^"']+["']""")


def test_playground_notebooks_have_no_saved_outputs_or_real_task_tokens() -> None:
    notebooks = sorted((REPO_ROOT / "playground").glob("*.ipynb"))
    assert notebooks

    for notebook_path in notebooks:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") == "code":
                assert cell.get("outputs", []) == [], f"{notebook_path}: cell {cell_index} has saved output"
                assert cell.get("execution_count") is None, f"{notebook_path}: cell {cell_index} has an execution count"
                source = "".join(cell.get("source", []))
                assert HARDCODED_API_KEY_PATTERN.search(source) is None, (
                    f"{notebook_path}: cell {cell_index} hardcodes an API key"
                )

            serialized_cell = json.dumps(cell, ensure_ascii=False)
            assert TASK_TOKEN_PATTERN.search(serialized_cell) is None, (
                f"{notebook_path}: cell {cell_index} contains a task-token-shaped value"
            )


def test_callback_pattern_uses_release_signature_contract() -> None:
    pattern_path = REPO_ROOT / "skills" / "UniParser-Tools" / "references" / "patterns.md"
    content = pattern_path.read_text(encoding="utf-8")

    assert 'data["checksum"]' not in content
    assert '"checksum":' not in content
    assert "X-UniParser-Signature" in content
    assert "request.get_data(cache=True)" in content


def test_customer_docs_include_result_retention_notice() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    skill = (REPO_ROOT / "skills" / "UniParser-Tools" / "SKILL.md").read_text(encoding="utf-8")
    notes = (REPO_ROOT / "skills" / "UniParser-Tools" / "references" / "notes.md").read_text(encoding="utf-8")

    assert "24 小时" in readme
    assert "retained for only 24 hours" in skill
    assert "retained for only **24 hours**" in notes


def test_skill_only_uses_tokens_from_successful_triggers() -> None:
    skill = (REPO_ROOT / "skills" / "UniParser-Tools" / "SKILL.md").read_text(encoding="utf-8")
    notes = (REPO_ROOT / "skills" / "UniParser-Tools" / "references" / "notes.md").read_text(encoding="utf-8")

    assert "Never use a token from a failed trigger" in skill
    assert "Save `token` only from success JSON or `trigger_meta.json`" in skill
    assert "candidate_token" not in skill
    assert "recoverable_token" not in skill
    assert "the `token` field in a failed parse stderr JSON" not in skill
    assert "--upload-mode" not in skill
    assert "TOS" not in skill
    assert "only persist the `token` from a successful trigger response" in notes
    assert "candidate_token" not in notes
    assert "recoverable_token" not in notes
