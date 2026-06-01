# Utility Functions

## Convert & Navigation

```python
from uniparser_tools.utils.convert import dict2obj
from uniparser_tools.utils.processor import tree_repr, flat_layout, clean_scientific_text

# Convert dict to object for easier navigation
pages_tree = dict2obj(result["pages_tree"])

# Print tree structure (for debugging)
print(tree_repr(pages_tree[0]))

# Flatten nested groups into a list
flat_items = flat_layout(grouped_item)

# Clean scientific text (Unicode normalization, remove control chars)
clean_text = clean_scientific_text(text, strict=False)
```

## BBox Operations

```python
from uniparser_tools.utils.bbox import BBox, Point

# BBox properties
bbox.area        # Area
bbox.width       # Width
bbox.height      # Height
bbox.tl          # Top-left point
bbox.br          # Bottom-right point
bbox.ctr         # Center point
bbox.xyxy        # (x1, y1, x2, y2) tuple
bbox.xywh        # (x, y, w, h) tuple

# BBox operations
bbox.iou(other)           # Intersection over Union
bbox.iof(other)           # Intersection over Foreground
bbox.intersection(other)  # Intersection box
bbox.union(other)         # Union box
bbox.expand(pix, wh)      # Expand by pixels
bbox.shrink(pix, wh)      # Shrink by pixels
```

## Text Processing

```python
from uniparser_tools.utils.processor import (
    is_head_of_paragraph,
    is_tail_of_paragraph,
    find_figure_caption_kws,
    recursive_required_items,
    recursive_required_content,
)

# Check if text starts like a paragraph heading
is_head_of_paragraph("Figure 3 shows...")  # True

# Check if text ends like a paragraph
is_tail_of_paragraph("the results.")  # True

# Extract figure keywords from caption
kws = find_figure_caption_kws("See Figure 3 and Table S2")
# Returns: ["Fig3", "TableS2"]

# Recursively extract items of specific types
items = recursive_required_items(group, required_types=[LayoutType.Paragraph])
content = recursive_required_content(group, required_types=[...])
```

## tree_repr Function

Print a tree structure for debugging nested groups:

```python
from uniparser_tools.utils.processor import tree_repr

# Example output:
# group
# ├── figuregroup
# │   ├── figure
# │   └── figurecaption
# └─ tablegroup
#     ├── table
#     └─ tablecaption

print(tree_repr(item, verbose=False))  # verbose=True includes bbox
```
