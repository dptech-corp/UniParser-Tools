# API Reference

## UniParserClient Methods

| Method | Description |
|--------|-------------|
| `health(...)` | Check service/backend health |
| `version(...)` | Get frontend and model-backend version metadata |
| `get_constants(...)` | Get live enums, layout types, and token rules |
| `trigger_file(file_path, ...)` | Submit PDF file for parsing |
| `trigger_snip(snip_path, ...)` | Submit image for parsing |
| `trigger_url(pdf_url, ...)` | Submit HTTP(S), S3, OSS, or TOS source for parsing |
| `request_tos_upload_links(files, ...)` | Request presigned TOS upload targets |
| `upload_files_to_tos(file_paths, ...)` | Upload local files without starting a parse |
| `get_result(token, ...)` | Get raw parsing results |
| `get_formatted(token, ...)` | Get formatted output |
| `get_third_party_output(token, ...)` | Get a third-party-compatible result payload |

`version()` returns `default_version`, `backend_versions`, and backend
`capabilities` when the deployment exposes multiple parser models. Use a key
from `backend_versions` as the trigger `model_version`. Service-discovery
methods accept `http_timeout=`.

## Read-only account APIs

The parsing client exposes the same authenticated transport through
`parser.account`:

| Method | Description |
|--------|-------------|
| `get_current_user()` | Get the authenticated user/profile |
| `get_balance()` | Get balance, currency, account status, and permissions |
| `get_usage_summary(period)` | Get `current_month`, `last_month`, or rolling 30-day summary |
| `list_usage_records(page, size)` | Get paginated usage records from the last 14 days |
| `list_balance_transactions(page, size)` | Get paginated balance ledger |

Use `UniParserAccountClient` directly when parsing methods are not needed.
No account, API-key, balance, or administrative write operations are exposed.

### trigger_file() - Async Callback Parameters

```python
result = parser.trigger_file(
    file_path="./document.pdf",
    sync=False,  # Required for async mode
    callback_url="https://...",  # Your callback endpoint
    callback_secret="your-secret",  # For signature verification
    # ... other parse mode parameters
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sync` | bool | `True` | `False` enables async mode with callbacks |
| `callback_url` | str | `None` | HTTP POST endpoint for completion notification |
| `callback_secret` | str | `None` | Shared secret for HMAC-SHA256 payload verification |

The release/v1.3 callback body is the raw JSON result. Verify
`X-UniParser-Signature: sha256=<hex>` by computing HMAC-SHA256 over the exact
raw body bytes before JSON parsing. Use `Idempotency-Key` for deduplication and
inspect `X-UniParser-Callback-Attempt` for retries. The URL and secret must be
provided together, callbacks require `sync=False`, and deployments may enforce
a callback-host allowlist.

### trigger_snip() - Async Callback Parameters

Same parameters as `trigger_file()` for image parsing.

### trigger_url() - Async Callback Parameters

Same parameters as `trigger_file()` for URL-based parsing.

`padding_snip` is not accepted by the URL endpoint; `proxy` is URL-only.

### Common release/v1.3 trigger parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeout` | `1800` | Server-side parse budget in seconds |
| `http_timeout` | `None` | Per-call client HTTP timeout override |
| `inplace_update` | `False` | Update an existing task with the same token |
| `preset_layout` | `None` | JSON string or Python list describing known layout |
| `model_version` | `None` | Model version advertised by `/version` |
| `server_generated_token` | `False` | Ask the server to generate a token when none is supplied |

The default remains a deterministic client-generated token for backward
compatibility. `preset_layout` is serialized to a JSON string for all three
trigger endpoints, including `trigger_url`.

### TOS presigned upload

This is an advanced SDK-only path. The CLI and MCP local-PDF workflows use direct `trigger_file` uploads and do not
call TOS or perform automatic upload fallback.

```python
uploaded = parser.upload_files_to_tos(["./document.pdf"])
source_url = uploaded["files"][0]["source_url"]
result = parser.trigger_url(source_url, server_generated_token=True)
```

Uploading does not start parsing. Presigned `PUT` requests do not include the
UniParser API key. Treat URLs returned by `request_tos_upload_links()` as
short-lived bearer credentials: do not log or forward them. The high-level
`upload_files_to_tos()` helper removes each `upload_url` from its result after
the upload completes. TOS content uploads use the client's longer
`upload_request_timeout` default; pass `http_timeout=` to override one call.

## Parse Modes

**For textual:**
- `ParseModeTextual.DigitalExported` - Extract embedded text (best quality)
- `ParseModeTextual.OCRFast` - Fast OCR
- `ParseModeTextual.OCRHighQuality` - High quality OCR
- `ParseModeTextual.Disable` - Skip textual extraction

**For table, molecule, chart, figure, expression, equation:**
- `ParseMode.OCRFast` - Fast OCR
- `ParseMode.OCRHighQuality` - High quality OCR
- `ParseMode.DumpBase64` - Return raw image as base64
- `ParseMode.Disable` - Skip extraction (client default)

## Format Flags

| Flag | Description |
|------|-------------|
| `FormatFlag.Markdown` | Markdown format |
| `FormatFlag.Html` | HTML format |
| `FormatFlag.Latex` | LaTeX format |
| `FormatFlag.Plain` | Plain text |
| `FormatFlag.Markup` | Text with markup tags |

## Output Types

| Type | Description | Use Case |
|------|-------------|----------|
| `content=True` | Full text as string | LLM input, reading |
| `objects=True` | Flat list of semantic blocks | Semantic analysis |
| `pages_dict=True` | Dict organized by pages | Page-level processing |
| `pages_tree=True` | Nested tree with parent-child | Structure analysis |
| `molecule_source=True` | Include molecule source images | Chemical structure analysis |

**Note:** `get_formatted()` also supports `marginalia=True` to include page headers/footers/numbers.
Both result methods accept `http_timeout=` for large response payloads.

`get_third_party_output()` currently accepts
`ThirdPartyFormatter.MinerU`. The local `build_item()` / `dict2obj()`
conversion path understands release/v1.3 inline `contents + types`, molecule
`esmi`, and full-HTML table spans, and ignores unknown response fields.

## Ordering Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `OrderingMethod.GapTree` | Gap Tree algorithm (default) | General documents |
| `OrderingMethod.Naive` | Simple top-to-bottom | Single-column text |
| `OrderingMethod.XYCut` | XY Cut recursive projection | Multi-column layout |
| `OrderingMethod.XYCutExp` | Extended XY Cut | Complex layouts |

```python
from uniparser_tools.common.constant import OrderingMethod

# Use XYCut for multi-column papers
result = parser.trigger_file(
    file_path="./paper.pdf",
    ordering_method=OrderingMethod.XYCut,
)
```
