# API Reference

## UniParserClient Methods

| Method | Description |
|--------|-------------|
| `trigger_file(file_path, ...)` | Submit PDF file for parsing |
| `trigger_snip(snip_path, ...)` | Submit image for parsing |
| `trigger_url(pdf_url, ...)` | Submit PDF URL for parsing |
| `get_result(token, ...)` | Get raw parsing results |
| `get_formatted(token, ...)` | Get formatted output |

## Parse Modes

**For textual:**
- `ParseModeTextual.DigitalExported` - Extract embedded text (best quality)
- `ParseModeTextual.OCRFast` - Fast OCR
- `ParseModeTextual.OCRHighQuality` - High quality OCR
- `ParseModeTextual.Disable` - Skip textual extraction

**For table, molecule, chart, figure, expression, equation:**
- `ParseMode.OCRFast` - Fast OCR (default)
- `ParseMode.OCRHighQuality` - High quality OCR
- `ParseMode.DumpBase64` - Return raw image as base64
- `ParseMode.Disable` - Skip extraction

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
