#!/usr/bin/env python3
# coding: utf-8

import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Union

from jinja2 import Environment, FileSystemLoader
from PIL import Image

from uniparser_tools.common.constant import Language, LayoutType
from uniparser_tools.common.dataclass import (
    PDF,
    BBox,
    ChartResult,
    DataclassJSONEncoder,
    EquationResult,
    ExpressionResult,
    LayoutItem,
    MoleculeResult,
    SemanticItem,
    TabularResult,
)
from uniparser_tools.order.structure_order import flatten_page
from uniparser_tools.utils import pdf_render as fitz  # PDFium-backed, permissive license (was PyMuPDF)
from uniparser_tools.utils.convert import dict2obj
from uniparser_tools.utils.image import dump_image_base64_str
from uniparser_tools.utils.log import get_root_logger


TYPE_COLORS = {
    "text": "#2ca02c",  # 文本 - 绿色，与段落颜色一致
    "image": "#bcbd22",  # 图像 - 黄绿色，与图像颜色一致
    "latex": "#cc6600",  # 公式 - 橙色，与公式颜色一致
    "html": "#8c564b",  # HTML表格 - 棕色，与表格颜色一致
    "molecule": "#ffb84d",  # 分子式 - 橙色，与分子式颜色一致
}

LABEL_COLORS = {
    # Top-level layout types
    LayoutType.DocumentTitle: "#1f77b4",  # 文档标题
    LayoutType.Title: "#ff7f0e",  # 章节标题
    LayoutType.Paragraph: "#2ca02c",  # 段落（文本）
    LayoutType.KeyValue: "#d62728",  # 字段代码（专利）
    LayoutType.Reference: "#9467bd",  # 参考文献
    LayoutType.Table: "#8c564b",  # 表格
    LayoutType.TableCaption: "#e377c2",  # 表格标题
    LayoutType.TableFootnote: "#7f7f7f",  # 表格脚注
    LayoutType.Image: "#bcbd22",  # 图像
    LayoutType.ImageCaption: "#17becf",  # 图像标题
    LayoutType.ImageFootnote: "#084f8f",  # 图像脚注
    LayoutType.Equation: "#cc6600",  # 公式（数学公式）
    LayoutType.EquationID: "#1b7a1b",  # 公式编号
    LayoutType.Algorithm: "#006bb3",  # 算法
    LayoutType.AlgorithmCaption: "#ff9933",  # 算法标题
    LayoutType.AlgorithmFootnote: "#40d040",  # 算法脚注
    LayoutType.TOC: "#cc6600",  # 目录
    LayoutType.Group: "#a61e1f",  # 组合
    LayoutType.HLine: "#6b4a8a",  # 分割
    LayoutType.PageHeader: "#634033",  # 页眉
    LayoutType.PageFooter: "#b35d9a",  # 页脚
    LayoutType.PageNumber: "#595959",  # 页码
    LayoutType.PageBar: "#858818",  # 页栏
    LayoutType.PageNote: "#108a97",  # 页注
    LayoutType.Watermark: "#3a9fdf",  # 水印（其他）
    # Bottom-level layout types
    LayoutType.Molecule: "#ffb84d",  # 内联分子
    LayoutType.MoleculeID: "#3fd03f",  # 内联分子索引
    LayoutType.MoleculeGroup: "#ff5c5d",  # 内联分子组
    LayoutType.Figure: "#b084d9",  # 内联图像
    LayoutType.Expression: "#b57b6c",  # 内联反应式
    LayoutType.Chart: "#f5a5d8",  # 内联图表
    LayoutType.Legend: "#a6a6a6",  # 内联子图图例
    LayoutType.FigureCaption: "#d4d52a",  # 内联子图标题
    LayoutType.FigureGroup: "#20e1f2",  # 内联图组
}

mapping_class = {
    LayoutType.Description: LayoutType.TableFootnote,
    LayoutType.Text: LayoutType.Paragraph,
    LayoutType.Token: LayoutType.MoleculeID,
    LayoutType.Caption: LayoutType.ImageCaption,
    LayoutType.Abandon: LayoutType.Watermark,
}
default_color = "#7B8185"
highlight_color = "#ff0000"

for t in LayoutType:
    if t not in LABEL_COLORS:
        if t in mapping_class:
            color = LABEL_COLORS.get(mapping_class[t], default_color)
        else:
            color = default_color
        LABEL_COLORS[t] = color


# 初始化 Jinja2 模板环境
_template_dir = Path(__file__).parent / "templates"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_template_dir)),
    autoescape=False,  # 因为我们需要输出 HTML 和 JavaScript
)
_template = _jinja_env.get_template("visualize_together.html")


def crop_image_to_base64(pil_image: Image.Image, poly: Union[List[float], BBox]) -> str:
    try:
        if not pil_image:
            return ""

        if hasattr(poly, "x1"):
            left, top, right, bottom = poly.x1, poly.y1, poly.x2, poly.y2
        elif isinstance(poly, list) and len(poly) >= 4:
            xs, ys = poly[0::2], poly[1::2]
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        else:
            return ""

        w, h = pil_image.size
        left, top = max(0, min(left, w - 1)), max(0, min(top, h - 1))
        right, bottom = max(left + 1, min(right, w)), max(top + 1, min(bottom, h))
        if right <= left or bottom <= top:
            return ""

        cropped = pil_image.crop((int(left), int(top), int(right), int(bottom)))
        return dump_image_base64_str(cropped)
    except Exception:
        return ""


def generate_svg_overlay(layout_items: List[SemanticItem], width: int, height: int) -> str:
    svg_parts = [f'<svg class="overlay-svg" viewBox="0 0 {width} {height}" preserveAspectRatio="none">']
    # 创建包含索引和item的列表，并按面积从大到小排序
    items_with_index = []
    for idx, item in enumerate(layout_items):
        bbox = item.bbox
        if not bbox:
            continue
        area = bbox.area
        items_with_index.append((idx, item, area))

    # 按面积从大到小排序，面积大的先绘制（在下面），面积小的后绘制（在上面），便于hover
    items_with_index.sort(key=lambda x: x[2], reverse=True)

    for idx, item, _ in items_with_index:
        bbox = item.bbox
        anno_id = str(idx)
        cat = item.type
        color = LABEL_COLORS.get(cat, highlight_color)
        points = [(bbox.x1, bbox.y1), (bbox.x2, bbox.y1), (bbox.x2, bbox.y2), (bbox.x1, bbox.y2)]

        try:
            points_str = " ".join([f"{p[0]},{p[1]}" for p in points])
            svg_parts.append(
                f'<polygon points="{points_str}" data-anno-id="{anno_id}" data-cat="{html.escape(str(cat))}" '
                f'stroke="{color}" onclick="highlightFromLeft(event, this)">'
                f"</polygon>"
            )
        except Exception:
            continue
    svg_parts.append("</svg>")
    return "".join(svg_parts)


def render_det_to_html(item: SemanticItem, order: int, pil_image: Image.Image) -> str:
    color = LABEL_COLORS.get(item.type, default_color)
    editable = "true"
    content_html = None
    edit_val = None
    copy_type = None

    if isinstance(item, LayoutItem) or not item.plain:
        editable = "false"
        source_b64 = crop_image_to_base64(pil_image, item.bbox)
        if source_b64:
            if not source_b64.startswith("data:"):
                source_b64 = f"data:image/png;base64,{source_b64}"
            content_html = f'<div class="rendered-box" data-type="image"><img src="{source_b64}" /></div>'
            edit_val = "(区域截图)"
            copy_type = "image"
        else:
            content_html = '<div class="rendered-box" data-type="text">(Empty)</div>'
            edit_val = ""
            copy_type = "text"
    # 公式 (Equation)
    elif isinstance(item, EquationResult):
        content_html = f'<div class="rendered-box" data-type="latex">$${item.latex}$$</div>'
        edit_val = html.escape(item.latex).strip()
        copy_type = "latex"
    # 表格 (Table)
    elif isinstance(item, (TabularResult, ExpressionResult, ChartResult)):
        content_html = f'<div class="rendered-box" data-type="html">{item.html}</div>'
        edit_val = html.escape(item.html).strip()
        copy_type = "html"
    # 分子式 (Molecule)
    elif isinstance(item, MoleculeResult):
        editable = "false"
        drawing = getattr(item, "drawing", "")
        if isinstance(drawing, str) and drawing.strip():
            drawing = drawing.replace("fill:#FFFFFF;", "fill:#FFFFFF00;")  # transparent background
            content_html = f'<div class="rendered-box" data-type="svg">{drawing}</div>'
        else:
            content_html = f'<div class="rendered-box" data-type="text">{html.escape(item.plain)}</div>'
        edit_val = item.plain
        copy_type = "molecule"
    else:
        content_html = f'<div class="rendered-box" data-type="text">{html.escape(item.plain)}</div>'
        edit_val = html.escape(item.plain).strip()
        copy_type = "text"

    block_val = item.block
    if content_html is None or copy_type is None or block_val is None:
        get_root_logger().error(f"Failed to render item: {item.token} {item.page} {item.block} {item.type}")
        return ""

    return f"""
    <div class="draggable" data-anno-id="{order}" style="border-left-color: {color}">
        <div class="controls">
            <div class="info-tag-group">
                <span class="info-order-index">No.{order}</span>
                <span class="info-order-tag" data-cat="{item.type}" style="background:{color}">{item.type}</span>
            </div>
            <div class="info-block-id">Block ID: {block_val} | Score: {item.conf:.4f}</div>
            <div class="btn-group">
                <span class="info-order-tag" data-cat="{copy_type}" style="color:{TYPE_COLORS[copy_type]}">{copy_type.capitalize()}</span>
                <button onclick="copyContent(this, '{copy_type}')">复制</button>
                <button onclick="moveItem(this, 'up')" title="上移">↑</button>
                <button onclick="moveItem(this, 'down')" title="下移">↓</button>
            </div>
        </div>
        <div class="editable" contenteditable="{editable}" spellcheck="false">{edit_val}</div>
        {content_html}
    </div>
    """


def plotly_pdf_results_interactive(pages_dict: List[List[Dict]], file_path: str, pages: List[int] = None) -> Dict:
    total_pages = len(pages_dict)
    if pages is None:
        pages = list(range(total_pages))

    page_idx = pages[0]
    if page_idx >= total_pages:
        page_idx = pages[-1]

    try:
        pdf = PDF("", 0, Language.Unknown, file_path)
        if pdf.is_pdf:
            doc = fitz.Document(pdf.path)
            assert total_pages == len(doc), f"Total pages mismatch: {total_pages} != {len(doc)}"
            page = doc[page_idx]
            rect: fitz.Rect = page.rect
            max_dpi = min(144, max(1, int(4096 * 72 / max(rect.width, rect.height))))  # max 4096 pixels
            pix: fitz.Pixmap = page.get_pixmap(dpi=max_dpi)
            pil_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        else:
            pil_image = Image.open(file_path)
    except Exception:
        get_root_logger().exception("[Viz Error] Load image failed:")
        pil_image = Image.new("RGB", (800, 1000), "white")

    page_objs = flatten_page(dict2obj([pages_dict[page_idx]])[0])

    for idx in range(len(page_objs)):
        page_objs[idx].bbox *= pil_image.size
        if hasattr(page_objs[idx], "bboxes"):
            for s_idx in range(len(page_objs[idx].bboxes)):
                page_objs[idx].bboxes[s_idx] *= pil_image.size

    return {
        "layout_items": page_objs,
        "pil_image": pil_image,
        "filename": str(file_path).split("/")[-1],
        "page_index": page_idx,
        "total_pages": total_pages,
    }


def create_interactive_html_from_data(vis_data: Dict) -> str:
    layout_items: List[SemanticItem] = vis_data["layout_items"]
    pil_image: Image.Image = vis_data["pil_image"]
    filename: str = vis_data["filename"]
    total_pages: int = vis_data["total_pages"]
    current_page: int = vis_data["page_index"]

    sidebar_parts = ['<div class="page-sidebar" id="page-sidebar">']
    for i in range(total_pages):
        active_cls = "active" if i == current_page else ""
        current_page_attr = ' data-current-page="true"' if i == current_page else ""
        sidebar_parts.append(
            f'<div class="page-thumb {active_cls}" onclick="goToPage({i})"{current_page_attr}>'
            f'<div class="page-thumb-num">{i + 1}</div>'
            f'<div class="page-thumb-label">PAGE</div>'
            f"</div>"
        )
    sidebar_parts.append("</div>")

    img_b64 = dump_image_base64_str(pil_image)
    svg_overlay = generate_svg_overlay(layout_items, pil_image.width, pil_image.height)
    content_image_block = f"""
    <div class="page-container">
        <img id="main-page-image" src="data:image/jpeg;base64,{img_b64}" alt="page" />
        {svg_overlay}
    </div>
    """

    content_list = "\n".join([render_det_to_html(item, idx, pil_image) for idx, item in enumerate(layout_items)])

    items_dicts = []
    for idx, item in enumerate(layout_items):
        d = asdict(item)
        d["anno_id"] = idx
        d["anno_type"] = item.type
        if isinstance(item, LayoutItem):
            # d["anno_text"] = item.plain
            pass
        elif isinstance(item, (TabularResult, ExpressionResult, ChartResult)):
            d["anno_html"] = item.html
        elif isinstance(item, MoleculeResult):
            d["anno_smiles"] = item.plain
            d["anno_svg"] = getattr(item, "drawing", "")
        elif isinstance(item, EquationResult):
            d["anno_latex"] = item.latex
        else:
            d["anno_text"] = item.plain
        items_dicts.append(d)

    json_data_str = json.dumps(items_dicts, ensure_ascii=False, cls=DataclassJSONEncoder)

    # 使用 Jinja2 模板渲染
    try:
        return _template.render(
            title=f"UniParser Viz - {filename}",
            base_name=filename,
            current_page_num=current_page + 1,
            total_pages=total_pages,
            page_sidebar_block="".join(sidebar_parts),
            content_image_block=content_image_block,
            content_list=content_list,
            raw_md_js="''",
            raw_json_js=json_data_str,
        )
    except Exception:
        # 如果模板加载失败，使用后备方案
        get_root_logger().exception("Failed to render Jinja2 template:")
        return ""


def plotly_pdf_results(pages_dict: List[List[Dict]], file_path: str, pages: List[int] = None):
    data = plotly_pdf_results_interactive(pages_dict, file_path, pages)
    return create_interactive_html_from_data(data)
