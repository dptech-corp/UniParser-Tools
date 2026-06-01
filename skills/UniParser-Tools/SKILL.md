---
name: uniparser-tools
description: >-
  Parse PDFs, document images, and public PDF URLs into structured Markdown via UniParser
  (https://uniparser.dp.tech/)—tables, equations as LaTeX, figures, and reading order.
  Use this skill whenever the user wants to parse or extract a document, scientific paper,
  patent, report, or any PDF/image/URL into Markdown, even if they only say "parse this PDF",
  "extract this paper", "PDF to markdown", or upload a local file. Trigger terms: UniParser,
  uniparser_tools, 文档解析, PDF解析, 论文解析, 专利解析, PDF转Markdown, 表格提取, 公式识别,
  化学分子, scientific paper, layout extraction, dp.tech, document parsing.
---

# UniParser-Tools Skill

## Installation

**Do not use** `pip install uniparser-tools` — the package is not reliably published on PyPI.

Install once into the **same Python environment** that runs the scripts below:

```bash
pip install "git+https://github.com/dptech-corp/UniParser-Tools.git"
```

`scripts/parse_document.py` checks `import uniparser_tools` at startup; if missing, it runs the command above automatically once, then retries.

Manual verify:

```bash
python3 -c "import uniparser_tools; print('ok')"
```

## Configuration (required)

**`UNIPARSER_API_KEY` is mandatory** (HTTP header `X-API-Key`). All requests use `https://uniparser.dp.tech/`.

### Agent: check before parsing

```bash
python3 -c "import os, sys; sys.exit(0 if os.getenv('UNIPARSER_API_KEY') else 1)"
```

- Exit code **0** → continue to parse.
- Exit code **1** → stop and guide the user (do not guess a key).

### If the key is missing, tell the user

1. Apply for a key at [https://uniparser.dp.tech/](https://uniparser.dp.tech/) (registration or business contact).
2. Set the environment variable using one of the commands below.
3. Restart the terminal or Cursor so the new variable is visible.

**macOS / Linux (current shell):**

```bash
export UNIPARSER_API_KEY="your-api-key"
```

**macOS / Linux (persist in zsh):**

```bash
echo 'export UNIPARSER_API_KEY="your-api-key"' >> ~/.zshrc
source ~/.zshrc
```

**Windows CMD (current session):**

```cmd
set UNIPARSER_API_KEY=your-api-key
```

**Windows CMD (persist — open a new terminal after):**

```cmd
setx UNIPARSER_API_KEY "your-api-key"
```

**Windows PowerShell (current session):**

```powershell
$env:UNIPARSER_API_KEY="your-api-key"
```

**Windows PowerShell (persist for user):**

```powershell
[Environment]::SetEnvironmentVariable("UNIPARSER_API_KEY", "your-api-key", "User")
```

Do not ask users to paste keys into chat unless unavoidable; prefer environment variables.

## How to Use This Skill

> **Working directory**: Run commands from this skill's root directory (folder containing `SKILL.md`).

### Input → command

| User provides | Command |
|---------------|---------|
| Local PDF | `--file-path` |
| Local image (.png, .jpg, …) | `--image-path` |
| Public PDF URL | `--pdf-url` |

Use **one** of the three flags per run. Do not write ad-hoc Python SDK code for the default workflow.

### Parse (submit + fetch Markdown)

```bash
python3 scripts/parse_document.py --file-path "/path/to/document.pdf"
python3 scripts/parse_document.py --image-path "/path/to/figure.png"
python3 scripts/parse_document.py --pdf-url "https://example.com/paper.pdf"
```

Optional output directory:

```bash
python3 scripts/parse_document.py --file-path "./paper.pdf" --output-dir "./results"
```

Default parse options (scientific paper / high quality, fixed in the script):

- `textual`, `equation`, `table`: OCRHighQuality  
- `chart`, `figure`, `expression`: DumpBase64  
- `molecule`: OCRFast  
- `sync=true`  
- Result format: Markdown (`get_formatted` with `content=True`)

### Read results

1. **stdout**: JSON with `ok`, `token`, `markdown_path`, `output_dir`, `input_type`.
2. **stderr**: `Token: ...`, `Markdown saved to: ...`, `Output directory: ...`.
3. **Tell the user the `token` explicitly** in your reply.
4. Open the `.md` file for full text; avoid truncating unless the user only wants a summary.

**Files written (every successful parse):**

| File | Description |
|------|-------------|
| `{stem}.md` | Full document Markdown from `parse_document.py` (stem = input filename) |
| `token_{first8}.md` | Markdown from `fetch_by_token.py` only |
| `token.txt` | Same token as stdout (for reuse) |
| `formatted_meta.json` | Metadata without full `content` |

`pages_tree.json` is **not** produced by default.

### Token identity (same input → same token)

The server derives the token from **your API key** plus the **exact input string** passed to `trigger_*`:

| Input type | Token is based on |
|------------|-------------------|
| Local PDF | Absolute path string given to `--file-path` |
| Public URL | Exact URL string given to `--pdf-url` (e.g. with or without trailing slash matters) |
| Image | Absolute path given to `--image-path` |

**Important:** `--pdf-url` and `--file-path` for the same document produce **different tokens**. Always recover using the token for the **same flag and string** you submitted.

To compute the token for recovery (same environment and API key as parse):

```bash
python3 -c "
import os, sys
from uniparser_tools.api.clients import UniParserClient
client = UniParserClient(host='https://uniparser.dp.tech/', api_key=os.environ['UNIPARSER_API_KEY'])
# Use ONE of the following (must match the original parse input exactly):
print(client.to_token('/absolute/path/to/document.pdf'))
# print(client.to_token('https://arxiv.org/pdf/1234.56789'))
"
```

### Token reuse (no re-upload)

The same token can fetch results again from the server. If the user provides a token:

```bash
python3 scripts/fetch_by_token.py --token "THE_TOKEN" --output-dir "./results"
```

Optional structured tree (large):

```bash
python3 scripts/fetch_by_token.py --token "THE_TOKEN" --output-dir "./results" --pages-tree
```

### On failure

- Show the stderr JSON `error.message` and stop (unless a recovery step below applies).
- Do not fall back to vision-only reading of the document as a substitute for UniParser.
- For **long-running jobs**, **Token is duplicated**, or **502 on `--pdf-url`**, see the sections below before starting another `parse_document.py`.

### Long-running sync tasks

`parse_document.py` uses **`sync=true`**: the HTTP call to `trigger_file` / `trigger_url` / `trigger_snip` blocks until the server finishes (or errors). During that wait:

1. **There may be no terminal output for many minutes** (especially large scientific PDFs). This is normal—not proof that the process is broken.
2. **`--output-dir` is created only after success** (when Markdown is written). An empty or missing output folder while the command is still running does **not** mean parsing failed.
3. **Do not start a second `parse_document.py` for the same input** while one is already running. The server will reject duplicates (see **Token is duplicated** below).
4. **If the agent or user stops the local process** (timeout, background kill, IDE cancel) but parsing may still be running on the server:
   - Compute the token for the **same input string** you used (see **Token identity** above).
   - Run `fetch_by_token.py` with that token and the desired `--output-dir`.
   - If `get_formatted` returns `status: success`, deliver the `.md` to the user—do **not** re-upload with `parse_document.py`.
5. **If `fetch_by_token.py` returns `status: processing`**, wait and retry fetch later; still do **not** call `parse_document.py` again for that input.
6. **Tell the user the token** as soon as you know it (from stderr, stdout, `token.txt`, or `to_token(...)`), so they can recover even if the local command is interrupted.

Typical durations: small PDFs often finish in under a minute; large papers or URL-triggered downloads may take **10–20+ minutes** before any output appears.

### Common errors and recovery

| Error / situation | Meaning | What to do |
|-------------------|---------|------------|
| **No output for a long time** | Sync trigger still waiting on server | Wait; or compute token → `fetch_by_token.py`. Do **not** start another `parse_document.py` for the same input. |
| **`Token is duplicated`** | A job for this API key + input already exists (in progress or finished) | Do **not** call `parse_document.py` again. Compute the token for that **exact** path or URL → `fetch_by_token.py`. If `processing`, retry fetch later. |
| **`502 Bad Gateway`** (or HTML 502 in `error.message`, stage `trigger_url`) | Server failed while fetching/processing the remote PDF (common for **very large** PDF URLs) | See **URL 502 fallback** below. |
| **`PARSE_ERROR` after local process killed** | Local CLI stopped; server job may still have completed | Same as row 1: `fetch_by_token.py` with the correct token. |

#### URL 502 fallback (`--pdf-url` failed)

When `--pdf-url` fails with **502** (or similar gateway errors during `trigger_url`):

1. **If the user requires URL-only (no local download):**
   - Explain that the public URL trigger failed (often due to PDF size or server-side fetch limits).
   - Offer to **retry `--pdf-url` once** after a short wait, or to use **`fetch_by_token.py`** if a previous URL job already produced a token.
   - Do **not** download the file locally unless the user agrees.

2. **If the user allows a local file (default fallback when not forbidden):**
   - Download the PDF to a known path (e.g. under `--output-dir`).
   - Run `parse_document.py --file-path "/absolute/path/to/file.pdf" --output-dir "./results"`.
   - Use the **file-path token** for any later fetch (it will differ from the URL token).

3. **Always report** which method succeeded (URL vs file-path) and the final **token** and **markdown_path**.

### Performance and limits

- Sync mode waits until parsing finishes; large PDFs may take several minutes (see **Long-running sync tasks**).
- Public service concurrency is limited (see [references/notes.md](./references/notes.md)).
- `--pdf-url` must be a **publicly accessible** PDF link.
- Images must use `--image-path` (`trigger_snip`), not `--file-path`.

## Advanced

For callbacks, custom `ParseMode`, or SDK examples, see [references/patterns.md](./references/patterns.md) and [references/api-reference.md](./references/api-reference.md).

Optional MCP server setup is documented in the [UniParser-Tools GitHub repo](https://github.com/dptech-corp/UniParser-Tools); it is separate from this CLI workflow.

## Reference documents

| Topic | File |
|-------|------|
| API reference | [references/api-reference.md](./references/api-reference.md) |
| Common patterns | [references/patterns.md](./references/patterns.md) |
| Data classes | [references/data-classes.md](./references/data-classes.md) |
| Layout types | [references/layout-types.md](./references/layout-types.md) |
| Utilities | [references/utilities.md](./references/utilities.md) |
| Important notes | [references/notes.md](./references/notes.md) |
