"""Lightweight UniParser HTTP client (no uniparser-tools / OpenCV)."""

from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from uniparser_agent.parse.options import SCIENTIFIC_PAPER_TRIGGER
from uniparser_agent.parse.transport import (
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_SYNC_REQUEST_TIMEOUT,
    RequestTimeout,
    UniParserHTTPTransport,
)


PENDING_STATUSES = frozenset({"undefined", "waiting", "processing"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"})


class UniParserApiClient:
    def __init__(
        self,
        host: str,
        api_key: str,
        *,
        request_timeout: RequestTimeout = DEFAULT_REQUEST_TIMEOUT,
        sync_request_timeout: RequestTimeout = DEFAULT_SYNC_REQUEST_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self._transport = UniParserHTTPTransport(
            host,
            api_key,
            request_timeout=request_timeout,
            session=session,
        )
        self.api_key = api_key
        self.host = self._transport.host
        self.request_timeout = request_timeout
        self.sync_request_timeout = sync_request_timeout
        self._user = uuid.uuid5(uuid.NAMESPACE_DNS, api_key)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> UniParserApiClient:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def to_token(self, task_id: str) -> str:
        return uuid.uuid5(self._user, task_id).hex

    def _trigger_data(
        self,
        *,
        trigger_kwargs: dict[str, Any] | None,
        allow_padding_snip: bool,
    ) -> dict[str, Any]:
        data = dict(SCIENTIFIC_PAPER_TRIGGER)
        if trigger_kwargs:
            data.update(trigger_kwargs)
        if not allow_padding_snip:
            data.pop("padding_snip", None)
        preset_layout = data.get("preset_layout")
        if isinstance(preset_layout, (list, dict)):
            data["preset_layout"] = json.dumps(preset_layout, ensure_ascii=False)
        return data

    def _trigger_http_timeout(
        self,
        sync: bool,
        http_timeout: RequestTimeout | None,
    ) -> RequestTimeout:
        if http_timeout is not None:
            return http_timeout
        return self.sync_request_timeout if sync else self.request_timeout

    def trigger_file(
        self,
        file_path: str,
        *,
        trigger_kwargs: dict[str, Any] | None = None,
        server_generated_token: bool = False,
        http_timeout: RequestTimeout | None = None,
    ) -> dict[str, Any]:
        token = None if server_generated_token else self.to_token(file_path)
        data = self._trigger_data(trigger_kwargs=trigger_kwargs, allow_padding_snip=True)
        data["token"] = token
        try:
            with open(file_path, "rb") as fh:
                return self._transport.request(
                    "POST",
                    "/trigger-file-async",
                    files={"file": fh},
                    data=data,
                    timeout=self._trigger_http_timeout(bool(data.get("sync", True)), http_timeout),
                    error_message="trigger file failed",
                    token=token,
                )
        except OSError as exc:
            return {
                "status": "error",
                "token": token,
                "message": "trigger file failed",
                "description": str(exc),
                "error_type": type(exc).__name__,
            }

    def trigger_url(
        self,
        pdf_url: str,
        *,
        trigger_kwargs: dict[str, Any] | None = None,
        server_generated_token: bool = False,
        http_timeout: RequestTimeout | None = None,
    ) -> dict[str, Any]:
        token = None if server_generated_token else self.to_token(pdf_url)
        data = self._trigger_data(trigger_kwargs=trigger_kwargs, allow_padding_snip=False)
        data["url"] = pdf_url
        data["token"] = token
        return self._transport.request(
            "POST",
            "/trigger-url-async",
            json=data,
            timeout=self._trigger_http_timeout(bool(data.get("sync", True)), http_timeout),
            error_message="trigger url failed",
            token=token,
        )

    def trigger_snip(
        self,
        snip_path: str,
        *,
        trigger_kwargs: dict[str, Any] | None = None,
        server_generated_token: bool = False,
        http_timeout: RequestTimeout | None = None,
    ) -> dict[str, Any]:
        token = None if server_generated_token else self.to_token(snip_path)
        data = self._trigger_data(trigger_kwargs=trigger_kwargs, allow_padding_snip=True)
        data["token"] = token
        try:
            raw = Path(snip_path).read_bytes()
            img_b64 = base64.b64encode(raw).decode("ascii")
            return self._transport.request(
                "POST",
                "/trigger-snip-async",
                data={"img": img_b64, **data},
                timeout=self._trigger_http_timeout(bool(data.get("sync", True)), http_timeout),
                error_message="trigger snip failed",
                token=token,
            )
        except OSError as exc:
            return {
                "status": "error",
                "token": token,
                "message": "trigger snip failed",
                "description": str(exc),
                "error_type": type(exc).__name__,
            }

    def get_result(
        self,
        token: str,
        *,
        pages_tree: bool = False,
        http_timeout: RequestTimeout | None = None,
    ) -> dict[str, Any]:
        payload = {
            "token": token,
            "content": False,
            "objects": False,
            "pages_dict": False,
            "pages_tree": pages_tree,
            "molecule_source": False,
        }
        return self._transport.request(
            "POST",
            "/get-result",
            json=payload,
            timeout=http_timeout,
            error_message="get result failed",
            token=token,
        )

    def get_formatted(
        self,
        token: str,
        *,
        http_timeout: RequestTimeout | None = None,
    ) -> dict[str, Any]:
        payload = {
            "token": token,
            "content": True,
            "objects": False,
            "pages_dict": False,
            "pages_tree": False,
            "molecule_source": False,
            "textual": "markdown",
            "table": "markdown",
            "molecule": "markdown",
            "chart": "markdown",
            "figure": "markdown",
            "expression": "markdown",
            "equation": "latex",
            "marginalia": False,
        }
        return self._transport.request(
            "POST",
            "/get-formatted",
            json=payload,
            timeout=http_timeout,
            error_message="get formatted failed",
            token=token,
        )


def resolve_input(raw: str) -> tuple[str, str, Path | None]:
    """Return (kind, source_stem, path) where kind is file|image|url."""
    text = raw.strip()
    if not text:
        raise ValueError("INPUT must not be empty.")
    if text.startswith("http://") or text.startswith("https://"):
        segment = urlparse(text).path.rstrip("/").rsplit("/", 1)[-1]
        stem = segment or "url_document"
        for ext in (".pdf", ".png", ".jpg", ".jpeg", ".webp"):
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
                break
        return "url", stem or "url_document", None

    path = Path(text).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    stem = path.stem or "document"
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return "image", stem, path
    return "file", stem, path
