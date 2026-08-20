from pathlib import Path

from PIL import Image

from uniparser_tools.common.constant import LayoutType
from uniparser_tools.common.dataclass import BBox, GroupedResult, TextualResult
from uniparser_tools.order.xy_cut_exp import sticky_items
from uniparser_tools.utils.image import dump_image_base64_str, load_image_base64_str
from uniparser_tools.utils.processor import tree_repr
from uniparser_tools.utils.visualize_together import plotly_pdf_results


def _text(block: int, text: str, bbox: BBox | None = None) -> TextualResult:
    return TextualResult(
        token="token",
        page=0,
        block=block,
        bbox=bbox or BBox(0.1, 0.1, 0.4, 0.2),
        conf=1.0,
        page_size=(100, 100),
        type=LayoutType.Paragraph,
        contents=[text],
        types=[LayoutType.Text],
    )


def test_sticky_items_keeps_valid_bbox_order() -> None:
    items = [
        _text(1, "first", BBox(0.10, 0.10, 0.21, 0.20)),
        _text(2, "second", BBox(0.12, 0.11, 0.23, 0.21)),
    ]

    snapped = sticky_items(items, offset=5)

    assert all(item.bbox.x1 <= item.bbox.x2 and item.bbox.y1 <= item.bbox.y2 for item in snapped)


def test_tree_repr_keeps_legacy_verbose_option() -> None:
    grouped = GroupedResult.clone(_text(1, "first"), items=[_text(2, "child")], method="synced")

    representation = tree_repr(grouped, verbose=True)
    assert "synced" in representation
    assert "[2]" in representation
    assert "BBox" in representation


def test_image_helpers_support_alpha() -> None:
    transparent = Image.new("RGBA", (4, 2), (255, 0, 0, 0))
    decoded = load_image_base64_str(dump_image_base64_str(transparent, quality=100)).convert("RGB")

    assert all(channel >= 245 for channel in decoded.getpixel((0, 0)))


def test_visualization_adds_page_thumbnail_without_server_assets(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (40, 60), "white").save(image_path)

    rendered = plotly_pdf_results([[]], str(image_path))

    assert "page-thumb-img" in rendered
    assert "data:image/jpeg;base64," in rendered
    assert "cdn.jsdelivr.net/npm/mathjax" in rendered
    assert "/static/images/parser-logo.png" not in rendered
