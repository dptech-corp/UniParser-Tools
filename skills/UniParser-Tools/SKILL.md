---
name: uniparser-tools
description: "Parse PDFs, document images, and public PDF URLs into structured Markdown via UniParser (https://uniparser.dp.tech/) — tables, equations as LaTeX, figures, and reading order. Use when the user wants to parse or extract a document, paper, patent, report, or PDF/image/URL into Markdown. Trigger terms: UniParser, uniparser_tools, 文档解析, PDF解析, 论文解析, 专利解析, PDF转Markdown, 表格提取, 公式识别, 化学分子, scientific paper, layout extraction, dp.tech, document parsing."
---

# UniParser-Tools Skill

Parse local PDFs, document images, and public PDF URLs into Markdown and structured layout JSON via [UniParser](https://uniparser.dp.tech/). Agents use the installed `uniparser` CLI—do not hand-write SDK code for the default workflow.

## Installation

Requires **Python 3.11+**. Install into the **same Python environment** that runs `uniparser`:

```bash
pip install "git+https://github.com/dptech-corp/UniParser-Tools.git"
```

PyPI is not published yet; use the git install above (no automatic `pip install`—if missing, exit with this command).

Verify:

```bash
uniparser --help
uniparser version
```

If `uniparser` is not found, the package is not installed in the active environment (no automatic `pip install`—exit with install guidance).

## Configuration

All requests use `https://uniparser.dp.tech/`. An API key is **mandatory** (HTTP header `X-API-Key`).

**Key sources** (highest priority first):

1. `uniparser --api-key YOUR_KEY …` (one-off)
2. Environment variable `UNIPARSER_API_KEY`
3. `~/.uniparser/config.yaml` (written by `uniparser auth`)

### Interactive setup (user machines)

```bash
uniparser auth
```

Apply for a key at [https://uniparser.dp.tech/](https://uniparser.dp.tech/) if needed. Prefer not pasting keys into chat.

### Agent pre-check

Before parsing, verify a key is configured:

```bash
uniparser auth --verify
```

- Exit code **0** → continue.
- Exit code **1** → stop; guide the user to `uniparser auth` or set `UNIPARSER_API_KEY`, then restart the terminal or Cursor.

Alternative (env only):

```bash
python3 -c "import os, sys; sys.exit(0 if os.getenv('UNIPARSER_API_KEY') else 1)"
```

**macOS / Linux:**

```bash
export UNIPARSER_API_KEY="your-api-key"
```

**Windows (PowerShell):**

```powershell
$env:UNIPARSER_API_KEY="your-api-key"
```

## Usage

### CLI reference

**Pipeline:** every trigger sends `token=None` with `server_generated_token=True`. Local PDFs use direct multipart upload with a 60-second upload window. Images use `trigger_snip`; public URLs use `trigger_url`. The CLI then polls `get_result` only with the token from a successful trigger, fetches `pages_tree` + Markdown, and saves them. Default `sync=true`; `--async` uses `sync=false` on the trigger only.

**Input:** one positional `INPUT` per run—the CLI detects type by path suffix or `http(s)://` URL:

- Local PDF → `uniparser parse "/path/to/document.pdf"`
- Local image (.png, .jpg, …) → `uniparser parse "/path/to/figure.png"`
- Public PDF URL → `uniparser parse "https://example.com/paper.pdf"`

```bash
uniparser parse "/path/to/document.pdf"
uniparser parse "/path/to/figure.png"
uniparser parse "https://example.com/paper.pdf"
```

Optional flags:

```bash
uniparser parse "./paper.pdf" -o "./results"
uniparser parse "./paper.pdf" --async
```

Recovery (existing server job—see **Common issues**):

```bash
uniparser fetch --token "TASK_TOKEN_FROM_PRIOR_RUN"
```

Valid recovery tokens come only from successful `uniparser --json parse …` stdout or `trigger_meta.json` written after a successful trigger. Never use a token from a failed trigger: it is diagnostic only and cannot be resumed with `fetch`.

**Default output for** `fetch` (when `-o` / `--output-dir` is omitted): `~/Uni-Parser-Skill/token_<prefix>/`, where `<prefix>` is the first 8 characters of the token (e.g. `~/Uni-Parser-Skill/token_a1b2c3d4/token_a1b2c3d4.md`). Passing a prior `parse` directory with `-o` treats it only as the preferred path; because that directory already exists, `fetch` writes to an available sibling such as `paper_1`. Always use the returned `output_dir`.

**Default output for** `parse` (when `-o` / `--output-dir` is omitted): `~/Uni-Parser-Skill/<source_stem>/`

`<source_stem>` = local file stem (`paper.pdf` → `paper`); for URLs, the last path segment with only `.pdf`/image suffix removed (`…/2606.05847` → `2606.05847`, not `2606`).

**Files written on success:**

- `pages_tree.json` — structured layout tree
- `{stem}.md` — full document Markdown
- `formatted_meta.json` — metadata without full `content`
- `trigger_meta.json` — task token and `trigger_kwargs` (for `uniparser fetch` recovery)

**Deliver to user:** open `{stem}.md` for full text; give the user the path and/or content. Mention `pages_tree.json` when layout structure matters. For reliable path extraction, prefer machine-readable output:

```bash
uniparser --json parse "/path/to/document.pdf"
```

Then read stdout JSON fields `markdown_path`, `pages_tree_path`, `output_dir`, and `token`.

**Parse options** (CLI flags; defaults match scientific-paper preset):


| Field               | Flag           | Default    |
| ------------------- | -------------- | ---------- |
| Text                | `--textual`    | `ocr-hq`   |
| Equation            | `--equation`   | `ocr-hq`   |
| Table               | `--table`      | `ocr-hq`   |
| Chart               | `--chart`      | `base64`   |
| Figure              | `--figure`     | `base64`   |
| Chemical expression | `--expression` | `base64`   |
| Molecule            | `--molecule`   | `ocr-fast` |


Choices: `disable`, `ocr-fast`, `ocr-hq`, `digital` (textual only), `base64`. `sync=true` by default (`--async` for `sync=false`).

Example:

```bash
uniparser parse paper.pdf --textual digital --molecule disable
```

## I/O contract


| Outcome              | Exit code | stdout                          | stderr                                                        |
| -------------------- | --------- | ------------------------------- | ------------------------------------------------------------- |
| Success (human mode) | 0         | Human-readable paths            | Progress (`Parsing... filename`)                              |
| Success (`--json`)   | 0         | Single JSON object (`ok: true`) | Progress (same as above)                                      |
| Failure              | 1         | (empty or unused)               | Single JSON line (`ok: false`, `error.code`, `error.message`) |


`--json` must appear **before** the subcommand:

```bash
uniparser --json parse paper.pdf    # correct
uniparser parse paper.pdf --json    # wrong
```

`parse` **success JSON fields** (`uniparser --json parse INPUT`):


| Field               | Meaning                                  |
| ------------------- | ---------------------------------------- |
| `ok`                | `true` on success                        |
| `output_dir`        | Result directory (absolute path)         |
| `markdown_path`     | Main Markdown file                       |
| `pages_tree_path`   | Layout tree JSON                         |
| `content_chars`     | Markdown body length                     |
| `token`             | Task token for `uniparser fetch --token` |
| `input_type`        | `file` / `image` / `url`                 |
| `trigger_meta_path` | Path to `trigger_meta.json`              |


If the preferred output directory already exists, the CLI creates an available sibling such as
`results_1` or `results_2`. Existing paths are never reused or deleted; use the returned `output_dir`.

Failure JSON may carry the standard `token` field for diagnostics. Never pass that value to `fetch`; use `fetch` only with a token from successful parse output or `trigger_meta.json`.

**Common error codes** (stderr JSON): `CONFIG_ERROR`, `INPUT_ERROR`, `UPLOAD_ERROR`, `PARSE_ERROR`, `TOKEN_NOT_FOUND`.

## Common issues

On failure, show stderr JSON `error.message`. Do not substitute vision-only reading for UniParser output.


| Problem                                                                  | Cause                                                                           | Solution                                                                                                                                                      |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_ERROR`                                                           | No API key or `uniparser` not installed                                         | **Configuration** + `pip install "git+https://github.com/dptech-corp/UniParser-Tools.git"`; `uniparser auth --verify`                                                                                  |
| `UPLOAD_ERROR` / write timeout                                           | Direct file upload failed; any token in the error is diagnostic only            | Re-run `uniparser parse`; the failed trigger cannot be resumed with `fetch`                                                                                  |
| `TOKEN_NOT_FOUND` / `status: undefined`                                  | The service did not recognize the token after three checks                      | Stop fetching; verify that the token came from successful output or `trigger_meta.json` and is still within the retention period                            |
| Job not done / long wait / CLI interrupted / `processing` / poll timeout | Sync or poll still running; or local process stopped while server job continues | Wait; do **not** start a second `uniparser parse` for the same input. Use saved `token` with `uniparser fetch --token TOKEN`; files appear only after exit 0  |
| `502 Bad Gateway` on URL input                                           | Server failed fetching or processing remote PDF                                 | Retry `uniparser parse "same url"` once; or download and `uniparser parse local.pdf`; or `uniparser fetch --token TOKEN` if a prior job exists                |
| `PARSE_ERROR`                                                            | Server `status: error` at trigger / poll / fetch                                | Read `error.message` and `stage`; match rows above; check `trigger_error.json` / `pages_tree_error.json` / `formatted_error.json` under output dir if present |


**Limits:** large PDFs may take 10–20+ minutes; public service ≤5 concurrent requests; PDF URLs must be publicly accessible. Parsing can be inaccurate, so verify critical content against the source. Results are retained for only 24 hours—fetch and save them promptly. See [Important notes](./references/notes.md). Save `token` only from success JSON or `trigger_meta.json`.

## Advanced

For callbacks, custom `ParseMode`, or SDK examples, see [Common patterns](./references/patterns.md) and [API reference](./references/api-reference.md).

Full CLI reference (flags, examples, JSON details): [CLI README](../../uniparser_tools/cli/README.md) in this repository.

Optional MCP server setup is in the [UniParser-Tools GitHub repo](https://github.com/dptech-corp/UniParser-Tools); it is separate from this CLI workflow.
Its `uniparser_parse` tool uses the same direct local-PDF upload, server-generated token, successful-trigger-only
token use, and three-check `undefined` limit as the CLI.

## Reference documents


| Topic           | File                                              |
| --------------- | ------------------------------------------------- |
| API reference   | [api-reference.md](./references/api-reference.md) |
| Common patterns | [patterns.md](./references/patterns.md)           |
| Data classes    | [data-classes.md](./references/data-classes.md)   |
| Layout types    | [layout-types.md](./references/layout-types.md)   |
| Utilities       | [utilities.md](./references/utilities.md)         |
| Important notes | [notes.md](./references/notes.md)                 |
