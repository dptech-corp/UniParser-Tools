# Common Patterns

## Pattern 1: Simple PDF to Markdown

```python
result = parser.trigger_file(file_path=pdf_path, textual=ParseModeTextual.DigitalExported)
token = result["token"]

result = parser.get_formatted(token, content=True, textual=FormatFlag.Markdown)
print(result["content"])
```

## Pattern 2: Extract Figures with Captions

```python
from uniparser_tools.utils.convert import dict2obj

result = parser.trigger_file(
    file_path=pdf_path,
    figure=ParseMode.DumpBase64,
    textual=ParseModeTextual.DigitalExported,
)
token = result["token"]

result = parser.get_result(token, pages_tree=True)
pages_tree = dict2obj(result["pages_tree"])

# Navigate tree to find figure groups
for page in pages_tree:
    for item in page:
        if item.type == "figuregroup":
            # item.items contains figure + caption
            print(item.format_as(FormatFlag.Markdown))
```

## Pattern 3: Async Processing with Callback

```python
result = parser.trigger_file(
    file_path=pdf_path,
    sync=False,  # Async mode
    callback_url="https://your-server.com/callback",
    callback_secret="your-secret-key",
)
# Service will POST to callback_url when done
```

## Pattern 4: Mixed Format Output

```python
result = parser.get_formatted(
    token,
    content=True,
    textual=FormatFlag.Markdown,   # Text as Markdown
    table=FormatFlag.Html,         # Tables as HTML
    equation=FormatFlag.Latex,     # Equations as LaTeX
    figure=FormatFlag.Markdown,    # Figures as Markdown img
)
```

## Pattern 5: Parse Image (Snippet)

```python
result = parser.trigger_snip(
    snip_path="./figure.png",
    textual=ParseModeTextual.OCRFast,
    equation=ParseMode.OCRFast,
)
token = result["token"]
```

## Pattern 6: Parse PDF from URL

```python
result = parser.trigger_url(
    pdf_url="https://example.com/paper.pdf",
    textual=ParseModeTextual.DigitalExported,
    proxy=None,  # Optional proxy
)
token = result["token"]
```
