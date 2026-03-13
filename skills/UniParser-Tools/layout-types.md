# Layout Types

## Text Types

| Type | Description |
|------|-------------|
| `paragraph` | Regular paragraph text |
| `title` | Section title |
| `documenttitle` | Document title |
| `caption` | Figure/table caption |
| `abstract` | Abstract section |
| `reference` | References/bibliography |
| `toc` | Table of contents |

## Content Types

| Type | Description |
|------|-------------|
| `table` | Table |
| `tablecaption` | Table title/caption |
| `tablefootnote` | Table footnote |
| `figure` / `image` | Figure/Image |
| `figurecaption` | Figure caption |
| `equation` | Math equation (display) |
| `equationinline` | Inline equation |
| `equationid` | Equation number |
| `molecule` | Chemical structure |
| `moleculeid` | Molecule index |
| `expression` | Chemical reaction |
| `chart` | Data chart |

## Layout Types

| Type | Description |
|------|-------------|
| `group` | Generic group container |
| `figuregroup` | Figure + caption group |
| `tablegroup` | Table + caption group |
| `pageheader` / `pagefooter` | Header/footer |
| `pagenumber` | Page number |

## Example: Checking Item Types

```python
from uniparser_tools.common.constant import LayoutType

for page in pages_tree:
    for item in page:
        # Check type as string
        if item.type == "figuregroup":
            print("Found figure group")

        # Or compare with LayoutType enum
        if item.type == LayoutType.Table:
            print("Found table")

        # Check multiple types
        if item.type in ["tablecaption", "tablefootnote"]:
            print("Found table annotation")
```
