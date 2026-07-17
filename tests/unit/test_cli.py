"""Unit tests for the ``uniparser`` CLI (no network)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from uniparser_tools.cli.core.credentials import ApiKeySource
from uniparser_tools.cli.core.input import InputKind, resolve_input
from uniparser_tools.cli.main import cli
from uniparser_tools.common.constant import ParseMode, ParseModeTextual


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def env_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UNIPARSER_API_KEY", raising=False)


@pytest.fixture
def no_config_file(monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path("/nonexistent/uniparser-cli-test-config/config.yaml")
    monkeypatch.setattr("uniparser_tools.cli.core.credentials.CONFIG_PATH", config_path)


class TestCliConfig:
    def test_parse_without_api_key(
        self,
        runner: CliRunner,
        env_without_api_key: None,
        no_config_file: None,
        tmp_path: Path,
    ) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        result = runner.invoke(cli, ["parse", str(pdf)])
        assert result.exit_code == 1
        payload = json.loads(result.stderr.strip())
        assert payload["ok"] is False
        assert payload["error"]["code"] == "CONFIG_ERROR"
        assert "uniparser auth" in payload["error"]["message"]

    def test_parse_with_config_file_api_key(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("UNIPARSER_API_KEY", raising=False)
        config_dir = tmp_path / ".uniparser"
        config_dir.mkdir()
        config_path = config_dir / "config.yaml"
        config_path.write_text("api_key: file-key\n", encoding="utf-8")
        monkeypatch.setattr("uniparser_tools.cli.core.credentials.CONFIG_DIR", config_dir)
        monkeypatch.setattr("uniparser_tools.cli.core.credentials.CONFIG_PATH", config_path)

        mock_client = MagicMock()
        mock_client.trigger_file.return_value = {"status": "error", "description": "stop early"}
        monkeypatch.setattr("uniparser_tools.cli.commands.parse.make_client", lambda ctx: (mock_client, None))

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        out = tmp_path / "out"
        result = runner.invoke(cli, ["parse", str(pdf), "-o", str(out)])
        assert result.exit_code == 1
        mock_client.trigger_file.assert_called_once()

    def test_fetch_without_api_key(self, runner: CliRunner, env_without_api_key: None, no_config_file: None) -> None:
        result = runner.invoke(cli, ["fetch", "--token", "abc123"])
        assert result.exit_code == 1
        payload = json.loads(result.stderr.strip())
        assert payload["error"]["code"] == "CONFIG_ERROR"

    def test_fetch_requires_token_flag(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UNIPARSER_API_KEY", "test-key")
        result = runner.invoke(cli, ["fetch"])
        assert result.exit_code != 0


class TestInputResolve:
    def test_url_input(self) -> None:
        resolved = resolve_input("https://example.com/paper.pdf")
        assert not isinstance(resolved, str)
        assert resolved.kind is InputKind.URL
        assert resolved.source_stem == "paper"

    def test_missing_file(self) -> None:
        assert isinstance(resolve_input("/nonexistent/file.pdf"), str)


class TestParseInputErrors:
    def test_parse_missing_file_returns_input_error_without_api_key(
        self,
        runner: CliRunner,
        no_config_file: None,
    ) -> None:
        env = {k: v for k, v in os.environ.items() if k != "UNIPARSER_API_KEY"}
        result = runner.invoke(cli, ["parse", "/nonexistent/file.pdf"], env=env)
        assert result.exit_code == 1
        payload = json.loads(result.stderr.strip())
        assert payload["error"]["code"] == "INPUT_ERROR"

    def test_parse_missing_file_returns_input_error(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        no_config_file: None,
    ) -> None:
        monkeypatch.setenv("UNIPARSER_API_KEY", "test-key")
        result = runner.invoke(
            cli,
            ["parse", "/nonexistent/file.pdf"],
            env={**os.environ, "UNIPARSER_API_KEY": "test-key"},
        )
        assert result.exit_code == 1
        payload = json.loads(result.stderr.strip())
        assert payload["error"]["code"] == "INPUT_ERROR"
        assert "File not found" in payload["error"]["message"]


class TestParseCommand:
    def test_parse_success_writes_trigger_meta_and_json_token(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("UNIPARSER_API_KEY", "test-key")
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_client = MagicMock()
        mock_client.trigger_file.return_value = {"status": "success", "token": "tok-parse-1"}
        mock_client.get_result.side_effect = [
            {"status": "processing"},
            {"status": "success", "pages_tree": {}},
            {"status": "success", "pages_tree": {"tree": []}},
        ]
        mock_client.get_formatted.return_value = {"status": "success", "content": "# Hi"}

        monkeypatch.setattr("uniparser_tools.cli.commands.parse.make_client", lambda ctx: (mock_client, None))

        out = tmp_path / "out"
        result = runner.invoke(
            cli,
            ["--json", "parse", str(pdf), "-o", str(out)],
            env={**os.environ, "UNIPARSER_API_KEY": "test-key"},
        )
        assert result.exit_code == 0, result.stderr
        assert "Parsing... paper.pdf" in result.stderr
        payload = json.loads(result.stdout)
        assert payload["token"] == "tok-parse-1"
        assert (out / "trigger_meta.json").is_file()
        meta = json.loads((out / "trigger_meta.json").read_text(encoding="utf-8"))
        assert meta["token"] == "tok-parse-1"
        assert meta["trigger_kwargs"]["textual"] == "ocr-hq"
        assert meta["trigger_kwargs"]["sync"] is True
        assert "preset" not in meta
        assert (out / "paper.md").is_file()

    def test_parse_default_trigger_kwargs(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("UNIPARSER_API_KEY", "test-key")
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_client = MagicMock()
        mock_client.trigger_file.return_value = {"status": "error", "description": "stop"}
        monkeypatch.setattr("uniparser_tools.cli.commands.parse.make_client", lambda ctx: (mock_client, None))

        out = tmp_path / "out"
        runner.invoke(
            cli,
            ["parse", str(pdf), "-o", str(out)],
            env={**os.environ, "UNIPARSER_API_KEY": "test-key"},
        )

        _, call_kwargs = mock_client.trigger_file.call_args
        assert call_kwargs["sync"] is True
        assert call_kwargs["textual"] is ParseModeTextual.OCRHighQuality
        assert call_kwargs["equation"] is ParseMode.OCRHighQuality
        assert call_kwargs["table"] is ParseMode.OCRHighQuality
        assert call_kwargs["chart"] is ParseMode.DumpBase64
        assert call_kwargs["figure"] is ParseMode.DumpBase64
        assert call_kwargs["expression"] is ParseMode.DumpBase64
        assert call_kwargs["molecule"] is ParseMode.OCRFast

    def test_parse_molecule_disable_override(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("UNIPARSER_API_KEY", "test-key")
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_client = MagicMock()
        mock_client.trigger_file.return_value = {"status": "error", "description": "stop"}
        monkeypatch.setattr("uniparser_tools.cli.commands.parse.make_client", lambda ctx: (mock_client, None))

        out = tmp_path / "out"
        runner.invoke(
            cli,
            ["parse", str(pdf), "-o", str(out), "--molecule", "disable"],
            env={**os.environ, "UNIPARSER_API_KEY": "test-key"},
        )

        _, call_kwargs = mock_client.trigger_file.call_args
        assert call_kwargs["molecule"] is ParseMode.Disable
        assert call_kwargs["table"] is ParseMode.OCRHighQuality

    def test_parse_invalid_equation_mode_rejected(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("UNIPARSER_API_KEY", "test-key")
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        result = runner.invoke(
            cli,
            ["parse", str(pdf), "--equation", "digital"],
            env={**os.environ, "UNIPARSER_API_KEY": "test-key"},
        )
        assert result.exit_code != 0
        assert "digital" in result.stderr.lower() or "invalid" in result.stderr.lower()


class TestFetchCommand:
    def test_fetch_success(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("UNIPARSER_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.get_result.side_effect = [
            {"status": "success"},
            {"status": "success", "pages_tree": {"tree": []}},
        ]
        mock_client.get_formatted.return_value = {"status": "success", "content": "body"}

        monkeypatch.setattr("uniparser_tools.cli.commands.fetch.make_client", lambda ctx: (mock_client, None))

        out = tmp_path / "fetch-out"
        result = runner.invoke(
            cli,
            ["--json", "fetch", "--token", "abcdef123456", "-o", str(out)],
            env={**os.environ, "UNIPARSER_API_KEY": "test-key"},
        )
        assert result.exit_code == 0, result.stderr
        assert "Parsing... token_abcdef12" in result.stderr
        payload = json.loads(result.stdout)
        assert payload["token"] == "abcdef123456"
        assert payload["fetched_by_token"] is True
        assert (out / "token_abcdef12.md").is_file()


class TestHealthVersion:
    def test_health_json(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UNIPARSER_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.health.return_value = {"status": "success"}
        monkeypatch.setattr("uniparser_tools.cli.commands.health.make_client", lambda ctx: (mock_client, None))

        result = runner.invoke(cli, ["--json", "health"], env={**os.environ, "UNIPARSER_API_KEY": "test-key"})
        assert result.exit_code == 0
        assert json.loads(result.stdout)["status"] == "success"

    def test_version_human(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UNIPARSER_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.version.return_value = {"status": "success", "version": "1.0"}
        monkeypatch.setattr("uniparser_tools.cli.commands.version.make_client", lambda ctx: (mock_client, None))

        result = runner.invoke(cli, ["version"], env={**os.environ, "UNIPARSER_API_KEY": "test-key"})
        assert result.exit_code == 0
        assert "uniparser-tools:" in result.stdout

    def test_version_without_api_key(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, no_config_file: None
    ) -> None:
        monkeypatch.setattr(
            "uniparser_tools.cli.commands.version.resolve_api_key_source",
            lambda ctx: ApiKeySource(api_key=None, source=""),
        )
        env = {k: v for k, v in os.environ.items() if k != "UNIPARSER_API_KEY"}
        result = runner.invoke(cli, ["version"], env=env)
        assert result.exit_code == 0
        assert "uniparser-tools:" in result.stdout
        assert "remote: skipped (no API key)" in result.stdout

    def test_version_json_without_api_key(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, no_config_file: None
    ) -> None:
        monkeypatch.setattr(
            "uniparser_tools.cli.commands.version.resolve_api_key_source",
            lambda ctx: ApiKeySource(api_key=None, source=""),
        )
        env = {k: v for k, v in os.environ.items() if k != "UNIPARSER_API_KEY"}
        result = runner.invoke(cli, ["--json", "version"], env=env)
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert "local" in payload
        assert payload["remote"] is None
        assert payload["remote_skipped"] is True


class TestHelp:
    def test_parse_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["parse", "--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.stdout
        assert "--async" in result.stdout
        assert "--textual" in result.stdout
        assert "--molecule" in result.stdout
        assert "--verbose" not in result.stdout

    def test_fetch_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["fetch", "--help"])
        assert result.exit_code == 0
        assert "--token" in result.stdout


class TestAuthCommand:
    def test_auth_saves_api_key(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("UNIPARSER_API_KEY", raising=False)
        config_dir = tmp_path / ".uniparser"
        config_path = config_dir / "config.yaml"
        monkeypatch.setattr("uniparser_tools.cli.core.credentials.CONFIG_DIR", config_dir)
        monkeypatch.setattr("uniparser_tools.cli.core.credentials.CONFIG_PATH", config_path)

        result = runner.invoke(cli, ["auth"], input="my-secret-key\n")
        assert result.exit_code == 0, result.stderr
        assert "Enter your API key" in result.stdout
        assert "API key saved successfully" in result.stdout
        assert config_path.is_file()
        assert "my-secret-key" in config_path.read_text(encoding="utf-8")

    def test_auth_keeps_existing_on_enter(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("UNIPARSER_API_KEY", raising=False)
        config_dir = tmp_path / ".uniparser"
        config_dir.mkdir()
        config_path = config_dir / "config.yaml"
        config_path.write_text("api_key: existing-key\n", encoding="utf-8")
        monkeypatch.setattr("uniparser_tools.cli.core.credentials.CONFIG_DIR", config_dir)
        monkeypatch.setattr("uniparser_tools.cli.core.credentials.CONFIG_PATH", config_path)

        result = runner.invoke(cli, ["auth"], input="\n")
        assert result.exit_code == 0, result.stderr
        assert "Current API key source: config" in result.stdout
        assert "Keeping existing API key." in result.stdout
        assert config_path.read_text(encoding="utf-8") == "api_key: existing-key\n"

    def test_auth_interactive_shows_env_source(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        config_dir = tmp_path / ".uniparser"
        monkeypatch.setattr("uniparser_tools.cli.core.credentials.CONFIG_DIR", config_dir)
        monkeypatch.setattr("uniparser_tools.cli.core.credentials.CONFIG_PATH", config_dir / "config.yaml")
        env = {k: v for k, v in os.environ.items() if k != "UNIPARSER_API_KEY"}
        env["UNIPARSER_API_KEY"] = "env-key"

        result = runner.invoke(cli, ["auth"], input="\n", env=env)
        assert result.exit_code == 0, result.stderr
        assert "Current API key source: env" in result.stdout
        assert "Keeping existing API key." in result.stdout

    def test_auth_show_masked(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("UNIPARSER_API_KEY", raising=False)
        config_dir = tmp_path / ".uniparser"
        config_dir.mkdir()
        config_path = config_dir / "config.yaml"
        config_path.write_text("api_key: abcdefghijklmnop\n", encoding="utf-8")
        monkeypatch.setattr("uniparser_tools.cli.core.credentials.CONFIG_DIR", config_dir)
        monkeypatch.setattr("uniparser_tools.cli.core.credentials.CONFIG_PATH", config_path)

        result = runner.invoke(cli, ["auth", "--show"])
        assert result.exit_code == 0
        assert "API key source: config" in result.stdout
        assert "abcd...mnop" in result.stdout

    def test_auth_show_env_source(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        env = {k: v for k, v in os.environ.items() if k != "UNIPARSER_API_KEY"}
        env["UNIPARSER_API_KEY"] = "env-secret-key"
        result = runner.invoke(cli, ["auth", "--show"], env=env)
        assert result.exit_code == 0
        assert "API key source: env" in result.stdout
        assert "env-...-key" in result.stdout

    def test_auth_show_without_config(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("UNIPARSER_API_KEY", raising=False)
        monkeypatch.setattr("uniparser_tools.cli.core.credentials.read_api_key_from_config", lambda: None)
        env = {k: v for k, v in os.environ.items() if k != "UNIPARSER_API_KEY"}

        result = runner.invoke(cli, ["auth", "--show"], env=env)
        assert result.exit_code == 1
        assert "No API key configured." in result.stdout
        assert "uniparser auth" in result.stdout

    def test_auth_verify_success(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("UNIPARSER_API_KEY", raising=False)
        config_dir = tmp_path / ".uniparser"
        config_dir.mkdir()
        config_path = config_dir / "config.yaml"
        config_path.write_text("api_key: configured\n", encoding="utf-8")
        monkeypatch.setattr("uniparser_tools.cli.core.credentials.CONFIG_DIR", config_dir)
        monkeypatch.setattr("uniparser_tools.cli.core.credentials.CONFIG_PATH", config_path)

        result = runner.invoke(cli, ["auth", "--verify"])
        assert result.exit_code == 0
        assert "API key is configured." in result.stdout
        assert "Source: config" in result.stdout
