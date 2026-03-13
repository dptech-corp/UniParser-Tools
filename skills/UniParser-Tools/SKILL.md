# UniParser-Tools Skill

Quick start guide for using UniParser-Tools to parse documents and extract structured content.

## Quick Start

### 1. Initialize Client

```python
import os
from uniparser_tools.api.clients import UniParserClient

api_key = os.getenv('UNIPARSER_API_KEY')
parser = UniParserClient(
    host="https://uniparser.dp.tech/",
    api_key=api_key
)
```

### 2. Parse a PDF File

```python
from uniparser_tools.common.constant import ParseMode, ParseModeTextual

result = parser.trigger_file(
    file_path="./document.pdf",
    textual=ParseModeTextual.DigitalExported,
    table=ParseMode.OCRFast,
    equation=ParseMode.OCRFast,
    figure=ParseMode.DumpBase64,
)

if result["status"] == "success":
    token = result["token"]
```

### 3. Get Results

```python
from uniparser_tools.common.constant import FormatFlag

# Option A: Get formatted content (Markdown/HTML/LaTeX)
result = parser.get_formatted(
    token,
    content=True,
    textual=FormatFlag.Markdown,
    table=FormatFlag.Markdown,
    equation=FormatFlag.Latex,
)
print(result["content"])

# Option B: Get structured data for programmatic access
from uniparser_tools.utils.convert import dict2obj

result = parser.get_result(token, pages_tree=True)
pages_tree = dict2obj(result["pages_tree"])

for page in pages_tree:
    for item in page:
        print(f"{item.type}: {item.format_as(FormatFlag.Markdown)}")
```

## Common Patterns

| Pattern | Description | Reference |
|---------|-------------|-----------|
| Simple PDF to Markdown | Extract full text as Markdown | [Patterns](./patterns.md#pattern-1-simple-pdf-to-markdown) |
| Extract Figures | Get figures with captions | [Patterns](./patterns.md#pattern-2-extract-figures-with-captions) |
| Async Callback | Background processing with webhook | [Patterns](./patterns.md#pattern-3-async-processing-with-callback) |
| Mixed Formats | Different output formats per element | [Patterns](./patterns.md#pattern-4-mixed-format-output) |
| Parse Image | Parse image/snippet files | [Patterns](./patterns.md#pattern-5-parse-image-snippet) |
| Parse URL | Parse PDF from URL | [Patterns](./patterns.md#pattern-6-parse-pdf-from-url) |

## Reference Documents

| Topic | File |
|-------|------|
| API Reference | [api-reference.md](./api-reference.md) |
| Common Patterns | [patterns.md](./patterns.md) |
| Data Classes | [data-classes.md](./data-classes.md) |
| Layout Types | [layout-types.md](./layout-types.md) |
| Utility Functions | [utilities.md](./utilities.md) |
| Important Notes | [notes.md](./notes.md) |

## Error Handling

```python
result = parser.trigger_file(file_path="./document.pdf")
if result["status"] != "success":
    print(f"Failed: {result.get('message')}")
    print(f"Details: {result.get('description')}")
```
