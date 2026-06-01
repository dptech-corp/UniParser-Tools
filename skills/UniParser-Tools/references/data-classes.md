# Data Classes

## SemanticItem (Base Class)

All extracted elements inherit from `SemanticItem`:

- `type` - LayoutType (e.g., "paragraph", "table", "figure")
- `bbox` - Bounding box coordinates (has `.xyxy`, `.xywh`, `.area`, `.tl`, `.br` properties)
- `page` - Page number
- `source` - Image source (base64, path, or URL)
- `format_as(flag)` - Format as Markdown/HTML/LaTeX

### Format Properties

Available on all result types:

| Property | Description |
|----------|-------------|
| `.plain` | Plain text content |
| `.markdown` | Markdown formatted |
| `.html` | HTML formatted |
| `.latex` | LaTeX formatted |
| `.markup` | Tagged markup format |

## Specific Result Types

| Type | Key Fields | Description |
|------|------------|-------------|
| `TextualResult` | `text`, `contents`, `bboxes` | Text content with bounding boxes |
| `TabularResult` | `structure` (HTML), `placeholders`, `contents` | Table with HTML structure, has `.df` (DataFrame) |
| `EquationResult` | `latex_repr` | Math equation in LaTeX format |
| `MoleculeResult` | `smi`, `caption`, `markush`, `drawing` | Chemical structure (SMILES notation) |
| `ExpressionResult` | `reactions` | Chemical reaction with `reactants`, `conditions`, `products` |
| `ChartResult` | `data` | Chart data, has `.df` (DataFrame) |
| `FigureResult` | `desc` | Figure with description |
| `GroupedResult` | `items`, `level`, `method` | Container with nested `SemanticItem`s |

**Note:** `TabularResult`, `ExpressionResult`, and `ChartResult` have a `.df` property for pandas DataFrame access.

## Example: Accessing Result Properties

```python
from uniparser_tools.utils.convert import dict2obj

result = parser.get_result(token, pages_tree=True)
pages_tree = dict2obj(result["pages_tree"])

for page in pages_tree:
    for item in page:
        # Access format properties
        print(f"Plain: {item.plain}")
        print(f"Markdown: {item.markdown}")
        print(f"HTML: {item.html}")
        print(f"LaTeX: {item.latex}")

        # For tables, access DataFrame
        if hasattr(item, 'df'):
            print(f"DataFrame:\n{item.df}")
```
