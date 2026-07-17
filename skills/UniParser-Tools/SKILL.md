---
name: uniparser-tools
description: "Parse PDFs, document images, and supported source URLs into structured Markdown via UniParser (https://uniparser.dp.tech/)—tables, equations as LaTeX, figures, and reading order. Use when the user wants to parse or extract a document, paper, patent, report, or PDF/image/URL into Markdown. Trigger terms: UniParser, uniparser_tools, 文档解析, PDF解析, 论文解析, 专利解析, PDF转Markdown, 表格提取, 公式识别, 化学分子, scientific paper, layout extraction, dp.tech, document parsing."
---

# UniParser-Tools Skill

Parse local PDFs, document images, and supported source URLs into Markdown and structured layout JSON via [UniParser](https://uniparser.dp.tech/). Agents run the bundled CLI scripts—do not hand-write SDK code for the default workflow.

## Installation

**Do not use** `pip install uniparser-tools` — the package is not reliably published on PyPI.

Install once into the **same Python environment** that runs the scripts below:

```bash
pip install "git+https://github.com/dptech-corp/UniParser-Tools.git"
```

`scripts/parse_document.py` and `scripts/fetch_by_token.py` check `import uniparser_tools` at startup; if missing, they exit with `CONFIG_ERROR` and the install command above (no automatic `pip install`).

Manual verify:

```bash
python3 -c "import uniparser_tools; print('ok')"
```

## Configuration

**`UNIPARSER_API_KEY` is mandatory** (HTTP header `X-API-Key`). All requests use `https://uniparser.dp.tech/`.

Check before parsing:

```bash
python3 -c "import os, sys; sys.exit(0 if os.getenv('UNIPARSER_API_KEY') else 1)"
```

- Exit code **0** → continue.
- Exit code **1** → stop and guide the user to set the key (do not guess).

If the key is missing:

1. Apply at [https://uniparser.dp.tech/](https://uniparser.dp.tech/).
2. Set the environment variable (see below).
3. Restart the terminal or Cursor.

**macOS / Linux:**

```bash
export UNIPARSER_API_KEY="your-api-key"
```

**Windows (PowerShell):**

```powershell
$env:UNIPARSER_API_KEY="your-api-key"
```

Prefer environment variables over pasting keys into chat.

## Usage

> **Working directory**: Run commands from this skill's root directory (folder containing `SKILL.md`).

### CLI reference

**Pipeline** (fixed in script): `submit → poll get_result until success → fetch pages_tree + Markdown → save`. Default `sync=true` (trigger blocks until server done; script still polls and fetches). `--async` uses `sync=false` on submit only.

**Input → flag** (use **one** per run):

- Local PDF → `--file-path`
- Local image (.png, .jpg, …) → `--image-path`
- Source URL (`http(s)`, `s3`, `oss`, `tos`, or `file`) → `--pdf-url`

```bash
python3 scripts/parse_document.py --file-path "/path/to/document.pdf"
python3 scripts/parse_document.py --image-path "/path/to/figure.png"
python3 scripts/parse_document.py --pdf-url "https://example.com/paper.pdf"
```

Optional flags:

```bash
python3 scripts/parse_document.py --file-path "./paper.pdf" --output-dir "./results"
python3 scripts/parse_document.py --file-path "./paper.pdf" --async
python3 scripts/parse_document.py --file-path "./paper.pdf" --overwrite
```

Recovery (existing server job—see **Common issues**):

```bash
python3 scripts/fetch_by_token.py --file-path "/path/to/document.pdf"
python3 scripts/fetch_by_token.py --pdf-url "https://example.com/paper.pdf"
```

**Default output** (when `--output-dir` is omitted): `~/Uni-Parser-Skill/<source_stem>/`

`<source_stem>` = local file stem (`paper.pdf` → `paper`); for URLs, the last path segment with only `.pdf`/image suffix removed (`…/2606.05847` → `2606.05847`, not `2606`).

**Files written on success:**

- `pages_tree.json` — structured layout tree
- `{stem}.md` — full document Markdown
- `formatted_meta.json` — metadata without full `content`

**Deliver to user:** read stdout JSON (`markdown_path`, `pages_tree_path`, `output_dir`); open `{stem}.md` for full text; give the user the path and/or content. Mention `pages_tree.json` when layout structure matters.

**Parse options** (fixed in script):

- `textual`, `equation`, `table`: OCRHighQuality
- `chart`, `figure`, `expression`: DumpBase64
- `molecule`: OCRFast
- `sync=true` by default (`--async` for `sync=false`)

## Common issues

On failure, show stderr JSON `error.message`. Do not substitute vision-only reading for UniParser output.

| Problem | Cause | Solution |
|---------|-------|----------|
| `CONFIG_ERROR` | Missing `UNIPARSER_API_KEY` or `uniparser_tools` not installed | **Configuration** + `pip install "git+https://github.com/dptech-corp/UniParser-Tools.git"`; re-check key one-liner |
| `DIR_EXISTS` | Output directory already exists | Ask user; re-run with `--overwrite` if they agree |
| `Token is duplicated` | Job for this API key + exact input already exists | Do **not** re-run `parse_document.py`. `python3 scripts/fetch_by_token.py` with the **same** flag and path/URL (e.g. `--file-path "/abs/path/paper.pdf"`) |
| Job not done / long wait / CLI interrupted / `processing` / poll timeout | Sync or poll still running; or local process stopped while server job continues | Wait; do **not** start a second `parse_document.py` for the same input. Re-run the same `fetch_by_token.py` command; files appear only after exit 0 |
| `502 Bad Gateway` on `--pdf-url` | Server failed fetching or processing remote PDF | Retry `--pdf-url` once; or download and `--file-path`; or `fetch_by_token.py --pdf-url "exact same url"` if a prior job exists |
| Wrong input flag | Images require `trigger_snip` | Use `--image-path`, not `--file-path` |
| `PARSE_ERROR` | Server `status: error` at trigger / poll / fetch | Read `error.message` and `stage`; match rows above; check `trigger_error.json` / `pages_tree_error.json` / `formatted_error.json` under output dir if present |

**Limits:** large PDFs may take 10–20+ minutes; public service ≤5 concurrent requests ([references/notes.md](./references/notes.md)); URL schemes and private object-store access depend on the target deployment. Recovery via `fetch_by_token.py` must use the **same** input string as the original parse—URL and file path are not interchangeable.

## Advanced

For callbacks, custom `ParseMode`, or SDK examples, see [references/patterns.md](./references/patterns.md) and [references/api-reference.md](./references/api-reference.md).

Optional MCP server setup is in the [UniParser-Tools GitHub repo](https://github.com/dptech-corp/UniParser-Tools); it is separate from this CLI workflow.

## Reference documents

| Topic | File |
|-------|------|
| API reference | [references/api-reference.md](./references/api-reference.md) |
| Common patterns | [references/patterns.md](./references/patterns.md) |
| Data classes | [references/data-classes.md](./references/data-classes.md) |
| Layout types | [references/layout-types.md](./references/layout-types.md) |
| Utilities | [references/utilities.md](./references/utilities.md) |
| Important notes | [references/notes.md](./references/notes.md) |
