from typing import Dict, List

from uniparser_tools.common.constant import FormatFlag, LayoutType, ParseMode, to_semantic
from uniparser_tools.common.dataclass import (
    ChartResult,
    EquationResult,
    ExpressionResult,
    FigureResult,
    GroupedResult,
    LayoutItem,
    MoleculeResult,
    SemanticItem,
    TabularResult,
    TextualResult,
)


def build_item(block: Dict):
    if "pages" in block:
        block.pop("pages")
    if "reactions" in block:
        item = ExpressionResult(**block)
    elif "placeholders" in block:
        if "html" in block:
            block["structure"] = block.pop("html")
        if "text" in block:
            block.pop("text")
        item = TabularResult(**block)
    elif "markush" in block:
        item = MoleculeResult(**block)
    elif "data" in block:
        item = ChartResult(**block)
    elif "desc" in block:
        item = FigureResult(**block)
    elif "latex_repr" in block:
        item = EquationResult(**block)
    elif "text" in block:
        item = TextualResult(**block)
    elif "items" in block:
        items = [build_item(child) for child in block["items"]]
        item = GroupedResult.clone(GroupedResult(**block), items=items)
    else:
        item = LayoutItem(**block)
    return item


def dict2obj(pages_dict: List[List[Dict]]):
    objs: List[List[SemanticItem]] = []
    for page in pages_dict:
        items: List[SemanticItem] = []
        for i in range(len(page)):
            block = page[i]
            item = build_item(block)
            items.append(item)
        objs.append(items)
    return objs


def item2format(item: SemanticItem, data: Dict, status: Dict):
    if item.type == LayoutType.Section:
        return ""

    if not data.__dict__.get("marginalia", False):
        if item.type in [
            LayoutType.PageHeader,
            LayoutType.PageFooter,
            LayoutType.PageBar,
            LayoutType.PageNote,
            LayoutType.PageNumber,
            LayoutType.Watermark,
        ]:
            return ""

    if isinstance(item, LayoutItem):
        s = ""
    else:
        if not isinstance(item, GroupedResult):
            item_format = data.__dict__[to_semantic(item.type)]
            s = getattr(item, item_format)
            if not item.plain and getattr(item, "source", ""):
                if status["dict_cfg"][to_semantic(item.type)] == ParseMode.DumpBase64:
                    if item_format == FormatFlag.Markdown:
                        s += f"![{item.type}](data:image/png;base64,{item.source})"
                    elif item_format == FormatFlag.Html:
                        s += f"<img src='data:image/png;base64,{item.source}' alt='{item.type}'/>"
                elif status["dict_cfg"][to_semantic(item.type)] == ParseMode.DumpLocal:
                    if item_format == FormatFlag.Markdown:
                        s += f"![{item.type}]({item.source})"
                    elif item_format == FormatFlag.Html:
                        s += f"<img src='{item.source}' alt='{item.type}'/>"
                elif status["dict_cfg"][to_semantic(item.type)] == ParseMode.DumpHosting:
                    if item_format == FormatFlag.Markdown:
                        s += f"![{item.type}]({item.source})"
                    elif item_format == FormatFlag.Html:
                        s += f"<img src='{item.source}' alt='{item.type}'/>"
            if item_format == FormatFlag.Html:
                s += "<br>"
        else:
            s_ = []
            for item in item.items:
                s_.append(item2format(item, data, status))
            s = "\n".join(s_)
    return s
