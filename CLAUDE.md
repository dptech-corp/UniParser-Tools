# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## UniParser Skill

A comprehensive skill for using UniParser-Tools is available at `skills/UniParser-Tools/SKILL.md`.

To use it, reference the skill which includes:
- Quick start guide with initialization and basic parsing
- Complete API reference for `UniParserClient`
- Common patterns (PDF to Markdown, figure extraction, async callbacks, mixed formats)
- Data classes and layout types reference
- Error handling examples

## Build & Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install as editable package
pip install -e .

# Run linting
ruff check .

# Run playground notebooks for testing
# Jupyter notebooks are in playground/ directory
```

## Architecture Overview

**UniParser-Tools** is a document parser toolkit that provides Python API for parsing PDFs and images, extracting text, tables, equations, molecules, charts, and figures.

### Core Modules

- **`uniparser_tools/api/clients.py`** - HTTP client for UniParser API service. Main entry point: `UniParserClient` with methods:
  - `trigger_file()` / `trigger_snip()` / `trigger_url()` - Submit parsing tasks
  - `get_result()` - Get raw parsing results (content, objects, pages_dict, pages_tree)
  - `get_formatted()` - Get formatted output (Markdown, HTML, LaTeX, Plain)

- **`uniparser_tools/common/`** - Shared types and constants
  - `constant.py` - Enums: `ParseMode`, `ParseModeTextual`, `FormatFlag`, `LayoutType`, `SemanticType`, `Language`
  - `dataclass.py` - Data classes: `PDF`, `SemanticItem`, `LayoutItem`, `GroupedResult`, `TextualResult`, `TabularResult`, `EquationResult`, `MoleculeResult`, `ExpressionResult`

- **`uniparser_tools/utils/`** - Utility functions
  - `image.py` - Image encoding/decoding
  - `fileio.py` - File I/O operations
  - `bbox.py` - Bounding box operations
  - `processor.py` - Text processing and layout utilities

- **`uniparser_tools/order/`** - Reading order algorithms
  - `gap_tree.py` - Gap Tree algorithm (default)
  - `xy_cut.py` / `xy_cut_exp.py` - XY Cut algorithms
  - `structure_order.py` - Structure-based ordering

- **`uniparser_tools/tools/caption_extraction/`** - Figure-caption extraction tool

### Key Concepts

- **Token-based workflow**: Submit task → receive token → use token to fetch results
- **Async callbacks**: Set `sync=False` with `callback_url` for HTTP POST callbacks on completion
- **Output formats**: `content` (full text), `objects` (JSON blocks), `pages_dict` (page-organized), `pages_tree` (nested structure)
- **Format flags**: Each element type can have independent output format (`FormatFlag.Markdown`, `FormatFlag.Html`, `FormatFlag.Latex`, `FormatFlag.Plain`)

### Concurrency Limit

The public UniParser service allows maximum 5 concurrent requests.
