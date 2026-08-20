from __future__ import annotations

import functools
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from html import escape
from io import StringIO
from typing import Any, Dict, List, Tuple, Union
from urllib.parse import quote

import pandas as pd
from pylatexenc.latexencode import unicode_to_latex

from uniparser_tools.common.constant import Direction, FileType, FormatFlag, Language, LayoutType, TableBBoxType
from uniparser_tools.utils.fileio import is_valid_image, read_html
from uniparser_tools.utils.log import get_root_logger


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

FLOAT_4 = Tuple[float, float, float, float]
INT_4 = Tuple[int, int, int, int]
INFO_DICT_TYPE = Dict[str, Any]


def item2format_(item: SemanticItem, item_format: FormatFlag):
    B64_RE = re.compile(r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")
    PATH_RE = re.compile(
        r"(?:[A-Za-z]:[/\\])?"  # 可选盘符
        r"(?:[\w\-+.]+[/\\])*[\w\-+.]+(?:\.\w+)?",  # 目录/文件名.扩展名
        re.IGNORECASE,
    )
    URL_RE = re.compile(
        r"https?://(?:[-\w.])+(?:\:[0-9]+)?"
        r"(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?",
        re.IGNORECASE,
    )

    if item.type == LayoutType.Section:
        return ""

    if isinstance(item, LayoutItem):
        s = ""
    else:
        if not isinstance(item, GroupedResult):
            s = getattr(item, item_format)
            if not item.plain and getattr(item, "source", ""):
                if B64_RE.fullmatch(item.source):
                    if item_format == FormatFlag.Markdown:
                        s += f"![{item.type}](data:image/png;base64,{item.source})"
                    elif item_format == FormatFlag.Html:
                        s += f"<img src='data:image/png;base64,{item.source}' alt='{item.type}'/>"
                    elif item_format == FormatFlag.Latex:
                        s += "\\includegraphics[width=0.5\\textwidth]{}"
                elif PATH_RE.match(item.source):
                    if item_format == FormatFlag.Markdown:
                        s += f"![{item.type}](<{item.source}>)"
                    elif item_format == FormatFlag.Html:
                        s += f"<img src='{item.source}' alt='{item.type}'/>"
                    elif item_format == FormatFlag.Latex:
                        s += f"\\includegraphics[width=0.5\\textwidth]{{{quote(item.source, safe='/:@&=?')}}}"
                elif URL_RE.match(item.source):
                    if item_format == FormatFlag.Markdown:
                        s += f"![{item.type}]({item.source})"
                    elif item_format == FormatFlag.Html:
                        s += f"<img src='{item.source}' alt='{item.type}'/>"
                    elif item_format == FormatFlag.Latex:
                        s += f"\\includegraphics[width=0.5\\textwidth]{{{quote(item.source, safe='/:@&=?')}}}"
            if item_format == FormatFlag.Markdown:
                s += "\n"
            elif item_format == FormatFlag.Html:
                s += "<br>"
            elif item_format == FormatFlag.Latex:
                s += "\n"
        else:
            s_ = []
            for item in item.items:
                s_.append(item2format_(item, item_format))
            s = "\n\n".join(s_)
    return s


class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


@dataclass
class DataClassGeneric:
    dump_dict = asdict


@dataclass
class PDF(DataClassGeneric):
    token: str
    pages: int
    lang: Language
    path: str
    exportable: bool = False
    file_type: FileType = FileType.Unknown

    def __post_init__(self):
        if self.file_type is FileType.Unknown:
            if self.path.lower().endswith(".pdf"):
                self.file_type = FileType.PDF
            elif is_valid_image(self.path):
                self.file_type = FileType.SNIP
            else:
                raise ValueError(f"Invalid file type: {self.path}")

    @property
    def is_pdf(self) -> bool:
        return self.file_type == FileType.PDF

    @property
    def is_snip(self) -> bool:
        return self.file_type == FileType.SNIP


@dataclass
class ProcessBar(DataClassGeneric):
    proc_page: int = 0
    total_page: int = 0

    proc_textual: int = 0
    total_textual: int = 0
    proc_mol: int = 0
    total_mol: int = 0
    proc_equa: int = 0
    total_equa: int = 0
    proc_figure: int = 0
    total_figure: int = 0
    proc_chart: int = 0
    total_chart: int = 0
    proc_expr: int = 0
    total_expr: int = 0
    proc_table: int = 0
    total_table: int = 0

    dict = asdict

    def total(self) -> Dict[str, int]:
        return {k.name: self.__dict__[k.name] for k in fields(self) if k.name.startswith("total")}

    def proc(self) -> Dict[str, int]:
        return {k.name: self.__dict__[k.name] for k in fields(self) if k.name.startswith("proc")}

    def __sub__(self, other: ProcessBar) -> ProcessBar:
        return ProcessBar(**{k.name: self.__dict__[k.name] - other.__dict__[k.name] for k in fields(self)})

    def __isub__(self, other: ProcessBar) -> ProcessBar:
        self = self.__sub__(other)
        return self

    def __add__(self, other: ProcessBar) -> ProcessBar:
        return ProcessBar(**{k.name: self.__dict__[k.name] + other.__dict__[k.name] for k in fields(self)})

    def __iadd__(self, other: ProcessBar) -> ProcessBar:
        self = self.__add__(other)
        return self

    def __getitem__(self, item: str) -> int:
        return self.__dict__[item]


@dataclass
class ServerConfig(DataClassGeneric):
    root: str
    parse_path: str
    version_path: str = "/version"
    health_path: str = "/health"

    @property
    def parse_url(self) -> str:
        return f"{self.root}{self.parse_path}" if self.root else ""

    @property
    def version_url(self) -> str:
        return f"{self.root}{self.version_path}" if self.root else ""

    @property
    def health_url(self) -> str:
        return f"{self.root}{self.health_path}" if self.root else ""


@dataclass
class Point(DataClassGeneric):
    x: float
    y: float

    def distance_to(self, other: Point, method: str = "euclidean") -> float:
        if method == "euclidean":
            return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
        elif method == "manhattan":
            return abs(self.x - other.x) + abs(self.y - other.y)
        raise NotImplementedError

    def __add__(self, other: Union[Point, Tuple[float, float]]) -> Point:
        if isinstance(other, (tuple, list)):
            return Point(self.x + other[0], self.y + other[1])
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Union[Point, Tuple[float, float]]) -> Point:
        if isinstance(other, (tuple, list)):
            return Point(self.x - other[0], self.y - other[1])
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other: Union[float, Tuple[float, float]]) -> Point:
        if isinstance(other, (tuple, list)):
            return Point(self.x * other[0], self.y * other[1])
        else:
            return Point(self.x * other, self.y * other)

    def __truediv__(self, other: Union[float, Tuple[float, float]]) -> Point:
        if isinstance(other, (tuple, list)):
            return Point(self.x / other[0], self.y / other[1])
        else:
            return Point(self.x / other, self.y / other)

    def __floordiv__(self, other: Union[float, Tuple[float, float]]) -> Point:
        if isinstance(other, (tuple, list)):
            return Point(self.x // other[0], self.y // other[1])
        else:
            return Point(self.x // other, self.y // other)

    def __eq__(self, other: Point) -> bool:
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __iter__(self):
        return iter((self.x, self.y))

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y)[index]

    def __len__(self) -> int:
        return 2

    @property
    def round(self) -> Point:
        return Point(*map(round, (self.x, self.y)))

    @property
    def int(self) -> Point:
        return Point(*map(int, (self.x, self.y)))

    @property
    def ceil(self) -> Point:
        return Point(*map(math.ceil, (self.x, self.y)))

    @property
    def floor(self) -> Point:
        return Point(*map(math.floor, (self.x, self.y)))

    @property
    def tuple(self) -> Tuple[float, float]:
        return self.x, self.y


@dataclass
class BBox(DataClassGeneric):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def wh(self) -> Tuple[float, float]:
        return self.width, self.height

    @property
    def tl(self) -> Point:
        return Point(self.x1, self.y1)

    @property
    def tr(self) -> Point:
        return Point(self.x2, self.y1)

    @property
    def bl(self) -> Point:
        return Point(self.x1, self.y2)

    @property
    def br(self) -> Point:
        return Point(self.x2, self.y2)

    @property
    def lc(self) -> Point:
        return Point(self.x1, (self.y1 + self.y2) / 2)

    @property
    def rc(self) -> Point:
        return Point(self.x2, (self.y1 + self.y2) / 2)

    @property
    def tc(self) -> Point:
        return Point((self.x1 + self.x2) / 2, self.y1)

    @property
    def bc(self) -> Point:
        return Point((self.x1 + self.x2) / 2, self.y2)

    @property
    def ctr(self) -> Point:
        return Point((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def xyxy(self) -> FLOAT_4:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def xywh(self) -> FLOAT_4:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2, self.width, self.height

    @property
    def xyxy_int(self) -> INT_4:
        return round(self.x1), round(self.y1), round(self.x2), round(self.y2)

    def expand(self, pix: int, wh: Tuple[int, int], axis=["x", "y"]) -> BBox:
        assert pix >= 0
        assert set(axis).issubset({"x", "y"}) and 2 >= len(axis) > 0
        if "x" in axis:
            x1 = max(self.x1 - pix, 0)
            x2 = min(self.x2 + pix, wh[0])
        else:
            x1 = self.x1
            x2 = self.x2
        if "y" in axis:
            y1 = max(self.y1 - pix, 0)
            y2 = min(self.y2 + pix, wh[1])
        else:
            y1 = self.y1
            y2 = self.y2
        return BBox(x1, y1, x2, y2)

    def shrink(self, pix: int, wh: Tuple[int, int], axis=["x", "y"]) -> BBox:
        assert pix >= 0
        assert set(axis).issubset({"x", "y"}) and 2 >= len(axis) > 0
        if "x" in axis:
            pix = max(0, min(pix, (self.x2 - self.x1 - 1) // 2))
            x1 = self.x1 + pix
            x2 = self.x2 - pix
        else:
            x1 = self.x1
            x2 = self.x2
        if "y" in axis:
            pix = max(0, min(pix, (self.y2 - self.y1 - 1) // 2))
            y1 = self.y1 + pix
            y2 = self.y2 - pix
        else:
            y1 = self.y1
            y2 = self.y2
        return BBox(x1, y1, x2, y2)

    def intersection(self, other: BBox) -> BBox:
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        return BBox(x1, y1, x2, y2) if x1 < x2 and y1 < y2 else BBox(0, 0, 0, 0)

    def union(self, other: BBox) -> BBox:
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)
        return BBox(x1, y1, x2, y2)

    def iou(self, other: BBox, axis=["x", "y"]) -> float:
        assert set(axis).issubset({"x", "y"}) and 2 >= len(axis) > 0
        if "x" in axis and "y" in axis:
            intersection = self.intersection(other).area
            union = self.area + other.area - intersection
            return intersection / union if union > 0 else 0
        else:
            union = self.union(other)
            if "x" in axis:
                x1 = max(self.x1, other.x1)
                x2 = min(self.x2, other.x2)
                return (x2 - x1) / union.width if x1 < x2 else 0
            else:
                y1 = max(self.y1, other.y1)
                y2 = min(self.y2, other.y2)
                return (y2 - y1) / union.height if y1 < y2 else 0

    def iof(self, other: BBox):
        intersection = self.intersection(other).area
        foreground = other.area
        return intersection / foreground if foreground > 0 else 0

    def __mul__(self, factor: Union[float, Tuple[float, float]]) -> BBox:
        if isinstance(factor, (float, int)):
            return BBox(self.x1 * factor, self.y1 * factor, self.x2 * factor, self.y2 * factor)
        else:
            return BBox(self.x1 * factor[0], self.y1 * factor[1], self.x2 * factor[0], self.y2 * factor[1])

    def __imul__(self, factor: Union[float, Tuple[float, float]]) -> BBox:
        if isinstance(factor, (float, int)):
            self.x1 *= factor
            self.y1 *= factor
            self.x2 *= factor
            self.y2 *= factor
        else:
            self.x1 *= factor[0]
            self.y1 *= factor[1]
            self.x2 *= factor[0]
            self.y2 *= factor[1]
        return self

    def __truediv__(self, factor: Union[float, Tuple[float, float]]) -> BBox:
        if isinstance(factor, (float, int)):
            return BBox(self.x1 / factor, self.y1 / factor, self.x2 / factor, self.y2 / factor)
        else:
            return BBox(self.x1 / factor[0], self.y1 / factor[1], self.x2 / factor[0], self.y2 / factor[1])

    def __itruediv__(self, factor: Union[float, Tuple[float, float]]) -> BBox:
        if isinstance(factor, (float, int)):
            self.x1 /= factor
            self.y1 /= factor
            self.x2 /= factor
            self.y2 /= factor
        else:
            self.x1 /= factor[0]
            self.y1 /= factor[1]
            self.x2 /= factor[0]
            self.y2 /= factor[1]
        return self

    def __eq__(self, other: BBox):
        return self.x1 == other.x1 and self.y1 == other.y1 and self.x2 == other.x2 and self.y2 == other.y2

    def __sub__(self, tl: Union[Point, Tuple[float, float]]) -> BBox:
        x1, y1 = tl
        return BBox(self.x1 - x1, self.y1 - y1, self.x2 - x1, self.y2 - y1)

    def __isub__(self, tl: Union[Point, Tuple[float, float]]) -> BBox:
        x1, y1 = tl
        self.x1 -= x1
        self.y1 -= y1
        self.x2 -= x1
        self.y2 -= y1
        return self

    def __add__(self, tl: Union[Point, Tuple[float, float]]) -> BBox:
        x1, y1 = tl
        return BBox(self.x1 + x1, self.y1 + y1, self.x2 + x1, self.y2 + y1)

    def __iadd__(self, tl: Union[Point, Tuple[float, float]]) -> BBox:
        x1, y1 = tl
        self.x1 += x1
        self.y1 += y1
        self.x2 += x1
        self.y2 += y1
        return self

    def __iter__(self):
        return iter([self.x1, self.y1, self.x2, self.y2])

    def __getitem__(self, key: int):
        return [self.x1, self.y1, self.x2, self.y2][key]

    def __len__(self):
        return 4

    def transpose(self, wh: Tuple[int, int], direction: Direction):
        if direction == Direction.Rotate_270:
            x1 = wh[0] - self.y2
            y1 = self.x1
            x2 = wh[0] - self.y1
            y2 = self.x2
        elif direction == Direction.Rotate_90:
            x1 = self.y1
            y1 = wh[1] - self.x2
            x2 = self.y2
            y2 = wh[1] - self.x1
        elif direction == Direction.Rotate_180:
            x1 = wh[0] - self.x2
            y1 = wh[1] - self.y2
            x2 = wh[0] - self.x1
            y2 = wh[1] - self.y1
        else:
            return
        assert x1 <= x2 and y1 <= y2, f"{x1} {y1} {x2} {y2}"
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        return self

    def itranspose(self, wh: Tuple[int, int], direction: Direction):
        if direction == Direction.Rotate_90:
            x1 = wh[1] - self.y2
            y1 = self.x1
            x2 = wh[1] - self.y1
            y2 = self.x2
        elif direction == Direction.Rotate_270:
            x1 = self.y1
            y1 = wh[0] - self.x2
            x2 = self.y2
            y2 = wh[0] - self.x1
        elif direction == Direction.Rotate_180:
            x1 = wh[0] - self.x2
            y1 = wh[1] - self.y2
            x2 = wh[0] - self.x1
            y2 = wh[1] - self.y1
        else:
            return
        assert x1 <= x2 and y1 <= y2, f"{x1} {y1} {x2} {y2}"
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        return self


@dataclass
class Item(DataClassGeneric):
    token: str
    page: int
    block: int
    bbox: BBox
    conf: float
    page_size: Tuple[int, int]
    type: LayoutType
    hidden: bool = False
    order: int = -1
    lang: Language = Language.Unknown
    direction: Direction = Direction.Unknown  # 需要如何旋转

    def __post_init__(self):
        if isinstance(self.bbox, list):
            self.bbox = BBox(*self.bbox)
        elif isinstance(self.bbox, dict):
            self.bbox = BBox(**self.bbox)
        if isinstance(self.type, str):
            self.type = LayoutType(self.type)
        if isinstance(self.lang, str):
            self.lang = Language(self.lang)
        if isinstance(self.direction, int):
            self.direction = Direction(self.direction)

    @staticmethod
    def _coerce_bbox(bbox):
        if isinstance(bbox, list):
            return BBox(*bbox)
        if isinstance(bbox, dict):
            return BBox(**bbox)
        return bbox

    @property
    def r_bbox(self):
        # return relative bbox, 0-1
        eps = 1 / min(self.page_size)
        if 0 <= self.bbox.x2 <= 1 + eps and 0 <= self.bbox.y2 <= 1 + eps:
            return self.bbox
        else:
            return self.bbox / self.page_size

    @property
    def p_bbox(self):
        # return absolute bbox, pixel
        eps = 1 / min(self.page_size)
        if 0 <= self.bbox.x2 <= 1 + eps and 0 <= self.bbox.y2 <= 1 + eps:
            return self.bbox * self.page_size
        else:
            return self.bbox

    @property
    def empty(self):
        # disable output
        return ""

    @property
    def plain(self):
        # palin text
        return ""

    def format_as(self, format_flag: FormatFlag):
        return item2format_(self, format_flag)

    @classmethod
    def clone(cls, item, **extra):
        assert is_dataclass(item)
        # skip key not in cls fields
        kwargs = {f.name: getattr(item, f.name) for f in fields(cls) if f.name in item.__dict__}
        kwargs.update({f.name: extra.get(f.name) for f in fields(cls) if f.name in extra})
        # clear source
        kwargs.pop("source", None)
        return cls(**kwargs)

    def transpose(self):
        self.bbox.transpose(self.page_size, self.direction)
        return self

    def itranspose(self):
        self.bbox.itranspose(self.page_size, self.direction)
        return self


@dataclass
class LayoutItem(Item):
    @property
    def plain(self):
        # palin text
        return f"<{self.type}>"

    @property
    def markup(self):
        # for sci-miner formater
        return self.plain

    @property
    def markdown(self):
        # for markdown formater
        return self.plain

    @property
    def latex(self):
        # for latex formater
        return self.plain

    @property
    def html(self):
        # for html formater
        return self.plain


@dataclass
class SemanticItem(Item):
    source: str = ""  # img_base64 / img_path / img_url

    @property
    def markup(self):
        # for sci-miner formater
        return self.plain

    @property
    def markdown(self):
        # for markdown formater
        return self.plain

    @property
    def latex(self):
        # for latex formater
        return self.plain

    @property
    def html(self):
        # for html formater
        return self.plain


@dataclass
class TextualResult(SemanticItem):
    type: LayoutType = LayoutType.Text
    bboxes: List[BBox] = field(default_factory=list)
    contents: List[str] = field(default_factory=list)
    text: str = ""  # Deprecated: legacy plain text field; use contents/types/bboxes instead.
    types: List[LayoutType] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if not self.contents and self.text:
            from uniparser_tools.utils.format_utils import parse_inline_text

            self.contents, self.types = parse_inline_text(self.text)
            self.bboxes = []
        elif self.contents and not self.types:
            self.types = [LayoutType.Text] * len(self.contents)
        if not self.text:
            self.text = "".join(self.contents)
        self.bboxes = [self._coerce_bbox(bbox) for bbox in self.bboxes]
        self.types = [LayoutType(t) if isinstance(t, str) else t for t in self.types]
        if len(self.contents) != len(self.types):
            raise ValueError("TextualResult contents/types length mismatch")
        if self.bboxes and len(self.bboxes) != len(self.contents):
            raise ValueError("TextualResult bboxes must be empty or match contents length")
        if any(bbox is None for bbox in self.bboxes):
            raise ValueError("TextualResult bboxes must not contain None")

    @staticmethod
    def _content_format(content: str, content_type: LayoutType, item_format: FormatFlag) -> str:
        if content_type in [LayoutType.Equation, LayoutType.EquationInline]:
            return EquationResult.format_latex_repr(content, item_format, inline=True)
        if content_type == LayoutType.Molecule:
            return MoleculeResult.format_molecule_text(content, item_format, inline=True)
        if item_format == FormatFlag.Html:
            return escape(content)
        if item_format == FormatFlag.Latex:
            try:
                return unicode_to_latex(content, unknown_char_policy="keep", unknown_char_warning=False)
            except Exception:
                get_root_logger().exception(f"unicode_to_latex failed: {content}")
                return content
        return content

    def _inline_text(self, item_format: FormatFlag) -> str:
        return "".join(
            self._content_format(content, content_type, item_format)
            for content, content_type in zip(self.contents, self.types)
        )

    @property
    def plain(self):
        return self._inline_text(FormatFlag.Plain)

    @property
    def markup(self):
        if self.type in [LayoutType.Text, LayoutType.Paragraph]:
            return self.plain
        else:
            return f"\\begin{{{self.type.value}}}\n{self.plain}\n\\end{{{self.type.value}}}"

    @property
    def markdown(self):
        if self.type == LayoutType.Algorithm:
            body = "".join(self.contents) if self.contents else self.text
            return f"```\n{body.rstrip()}\n```"

        plain = self._inline_text(FormatFlag.Markdown)
        if self.type in [LayoutType.Paragraph, LayoutType.Text, LayoutType.Description]:
            return plain
        elif self.type == LayoutType.Legend:
            return f"*{plain}*"
        elif self.type in [LayoutType.Token, LayoutType.EquationID, LayoutType.MoleculeID]:
            return f"{plain}"
        elif self.type in [
            LayoutType.Caption,
            LayoutType.TableCaption,
            LayoutType.FigureCaption,
            LayoutType.ImageCaption,
            LayoutType.AlgorithmCaption,
            LayoutType.ExpressionCaption,
        ]:
            return f"**{plain}**"
        elif self.type in [
            LayoutType.AlgorithmFootnote,
            LayoutType.ImageFootnote,
            LayoutType.TableFootnote,
            LayoutType.PageNote,
        ]:
            return f"*{plain}*"
        elif self.type == LayoutType.Title:
            return f"## {plain}"
        elif self.type == LayoutType.DocumentTitle:
            return f"# {plain}"
        else:
            return plain

    @property
    def latex(self):
        if self.contents:
            plain = self._inline_text(FormatFlag.Latex)
        else:
            try:
                # 对于无法映射到 LaTeX 的字符（例如中文“物”），使用 unknown_char_policy=\"keep\"
                # 以保留原始字符并避免 pylatexenc 打印告警。
                plain = unicode_to_latex(self.plain, unknown_char_policy="keep", unknown_char_warning=False)
            except Exception:
                get_root_logger().exception(f"unicode_to_latex failed: {self.plain}")
                plain = self.plain
        if self.type in [LayoutType.Paragraph, LayoutType.Text, LayoutType.Description]:
            return plain
        elif self.type == LayoutType.Legend:
            return f"\\textit{{{plain}}}"
        elif self.type in [LayoutType.Token, LayoutType.EquationID, LayoutType.MoleculeID]:
            return f"\\textit{{{plain}}}"
        elif self.type in [
            LayoutType.Caption,
            LayoutType.TableCaption,
            LayoutType.FigureCaption,
            LayoutType.ImageCaption,
            LayoutType.AlgorithmCaption,
            LayoutType.ExpressionCaption,
        ]:
            return f"\\textbf{{{plain}}}"
        elif self.type in [
            LayoutType.AlgorithmFootnote,
            LayoutType.ImageFootnote,
            LayoutType.TableFootnote,
            LayoutType.PageNote,
        ]:
            return f"\\textit{{{plain}}}"
        elif self.type == LayoutType.Algorithm:
            # verbatim needs raw source text, not latex-escaped body.
            body = "".join(self.contents) if self.contents else self.plain
            return f"\\begin{{verbatim}}\n{body.rstrip()}\n\\end{{verbatim}}"
        elif self.type == LayoutType.Title:
            return f"\\section{{{plain}}}"
        elif self.type == LayoutType.DocumentTitle:
            return f"\\title{{{plain}}}"
        else:
            return plain

    @property
    def html(self):
        plain = self._inline_text(FormatFlag.Html)
        if self.type in [LayoutType.Paragraph, LayoutType.Text, LayoutType.Description]:
            return f"<p>{plain}</p>"
        elif self.type == LayoutType.Legend:
            return f"<legend>{plain}</legend>"
        elif self.type in [LayoutType.Token, LayoutType.EquationID, LayoutType.MoleculeID]:
            return f"<em>{plain}</em>"
        elif self.type in [
            LayoutType.Caption,
            LayoutType.TableCaption,
            LayoutType.FigureCaption,
            LayoutType.ImageCaption,
            LayoutType.AlgorithmCaption,
            LayoutType.ExpressionCaption,
        ]:
            return f"<caption>{plain}</caption>"
        elif self.type in [
            LayoutType.AlgorithmFootnote,
            LayoutType.ImageFootnote,
            LayoutType.TableFootnote,
            LayoutType.PageNote,
        ]:
            return f"<aside><em>{plain}</em></aside>"
        elif self.type == LayoutType.Algorithm:
            return f"<pre><code>{plain}</code></pre>"
        elif self.type == LayoutType.Title:
            return f"<h2>{plain}</h2>"
        elif self.type == LayoutType.DocumentTitle:
            return f"<h1>{plain}</h1>"
        else:
            return f"<p>{plain}</p>"


@dataclass
class MoleculeResult(SemanticItem):
    """
    'beam_idx': 0,
    'caption': '*NC(=O)Nc1nc(*)c2c(*)n[nH]c2c1*<sep><a>0:R[2]</a><a>8:R[11]</a><a>11:R[1]</a><a>16:R[10]</a>',
    'drawing': '',
    'markush': True,
    'score': 0.9962224612367694,
    'smi': '*NC(=O)Nc1nc(*)c2c(*)n[nH]c2c1*',
    'sru': False
    """

    type: LayoutType = LayoutType.Molecule
    caption: str = ""
    markush: bool = False
    smi: str = ""
    sru: bool = False
    drawing: str = ""
    esmi: str = ""

    @property
    def plain(self):
        if self.markush or not self.smi:
            return self.caption
        else:
            return self.smi

    @property
    def markup(self):
        return f"\\begin{{{self.type.value}}}\n{self.plain}\n\\end{{{self.type.value}}}"

    @staticmethod
    def format_molecule_text(text: str, item_format: FormatFlag, inline: bool = False):
        if item_format == FormatFlag.Markdown:
            return f"`{text}`"
        if item_format == FormatFlag.Latex:
            return f"\\texttt{{{unicode_to_latex(text, unknown_char_policy='keep', unknown_char_warning=False)}}}"
        if item_format == FormatFlag.Html:
            return f"<code>{escape(text)}</code>"
        return text

    @property
    def markdown(self):
        # if self.drawing.strip():
        #     return f"<div><div>{self.drawing}</div><code>{self.plain}</code></div>"
        return self.format_molecule_text(self.plain, FormatFlag.Markdown)

    @property
    def latex(self):
        return self.format_molecule_text(self.plain, FormatFlag.Latex)

    @property
    def html(self):
        # if self.drawing.strip():
        #     return f"<div><div>{self.drawing}</div><code>{self.plain}</code></div>"
        return self.format_molecule_text(self.plain, FormatFlag.Html)


@dataclass
class ReactionComponent(DataClassGeneric):
    bbox: BBox
    category: str  # [Mol]  [Txt]  [Idt]
    category_id: int  # 1       2      3
    text: str = ""

    def __post_init__(self):
        if isinstance(self.bbox, list):
            self.bbox = BBox(*self.bbox)
        elif isinstance(self.bbox, dict):
            self.bbox = BBox(**self.bbox)

    @property
    def is_molecule(self):
        return self.category_id == 1 or self.category == "[Mol]"

    def format_text(self, item_format: FormatFlag) -> str:
        text = str(self.text or "").strip()
        if not text:
            return ""
        if self.is_molecule:
            return TextualResult._content_format(text, LayoutType.Molecule, item_format)

        from uniparser_tools.utils.format_utils import parse_inline_text

        contents, types = parse_inline_text(text)
        if not contents:
            contents, types = [text], [LayoutType.Text]
        content_types = types or [LayoutType.Text] * len(contents)
        return "".join(
            TextualResult._content_format(str(content), content_type, item_format)
            for content, content_type in zip(contents, content_types)
        )


@dataclass
class Reaction(DataClassGeneric):
    reactants: List[ReactionComponent] = field(default_factory=list)
    conditions: List[ReactionComponent] = field(default_factory=list)
    products: List[ReactionComponent] = field(default_factory=list)

    def __post_init__(self):
        if len(self.reactants) and isinstance(self.reactants[0], dict):
            self.reactants = [ReactionComponent(**r) for r in self.reactants]
        if len(self.conditions) and isinstance(self.conditions[0], dict):
            self.conditions = [ReactionComponent(**c) for c in self.conditions]
        if len(self.products) and isinstance(self.products[0], dict):
            self.products = [ReactionComponent(**p) for p in self.products]

    def __iter__(self):
        return iter(self.reactants + self.conditions + self.products)

    def __len__(self):
        return len(self.reactants) + len(self.conditions) + len(self.products)

    def __getitem__(self, n: int):
        if n < len(self.reactants):
            return self.reactants[n]
        elif n < len(self.reactants) + len(self.conditions):
            return self.conditions[n - len(self.reactants)]
        else:
            return self.products[n - len(self.reactants) - len(self.conditions)]

    @property
    def dict(self):
        return dict(
            prev_mols=[r.text for r in self.reactants],
            post_mols=[pr.text for pr in self.products],
            condition=dict(
                text="".join([c.text for c in self.conditions if c.category_id != 1]),
                catalyst=[c.text for c in self.conditions if c.category_id == 1],
            ),
        )


@dataclass
class ExpressionResult(SemanticItem):
    type: LayoutType = LayoutType.Expression
    reactions: List[Reaction] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if len(self.reactions) and isinstance(self.reactions[0], dict):
            self.reactions = [Reaction(**r) for r in self.reactions]

    @staticmethod
    def _format_components(components, item_format: FormatFlag, sep: str = ",") -> str:
        texts = []
        for component in components or []:
            if isinstance(component, dict):
                component = ReactionComponent(**component)
            text = component.format_text(item_format)
            if text:
                texts.append(text)
        return sep.join(texts)

    def _df_for_format(self, item_format: FormatFlag) -> pd.DataFrame:
        try:
            reactions = []
            for r in self.reactions:
                reactions.append(
                    dict(
                        reactants=self._format_components(r.reactants, item_format),
                        products=self._format_components(r.products, item_format),
                        conditions=self._format_components(r.conditions, item_format),
                    )
                )
            if not len(reactions):
                return pd.DataFrame()
            df = pd.DataFrame(reactions, columns=["reactants", "products", "conditions"])
            df.index.rename("No.", inplace=True)
            df.reset_index(inplace=True)
            return df
        except Exception:
            get_root_logger().warning(
                f"{self.token} {self.page} {self.block} {self.type} convert expression to dataframe error: {self.reactions}"
            )
            return pd.DataFrame()

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        return self._df_for_format(FormatFlag.Plain)

    @property
    def plain(self):
        if self.df.shape[-1] == 0:
            return ""
        return self.df.to_markdown(index=False, disable_numparse=True)

    @property
    def markup(self):
        return f"\\begin{{reaction}}\n{[r.dict for r in self.reactions]}\n\\end{{reaction}}"

    @property
    def markdown(self):
        df = self._df_for_format(FormatFlag.Markdown)
        if df.shape[-1] == 0:
            return ""
        return df.to_markdown(index=False, disable_numparse=True)

    @property
    def latex(self):
        df = self._df_for_format(FormatFlag.Latex)
        if df.shape[-1] == 0:
            return ""
        return df.style.hide(axis="index").to_latex()

    @property
    def html(self):
        def component_html(components):
            return self._format_components(components, FormatFlag.Html, sep="<br>")

        rows = []
        for reaction in self.reactions or []:
            reactants = (
                reaction.get("reactants", []) if isinstance(reaction, dict) else getattr(reaction, "reactants", [])
            )
            products = reaction.get("products", []) if isinstance(reaction, dict) else getattr(reaction, "products", [])
            conditions = (
                reaction.get("conditions", []) if isinstance(reaction, dict) else getattr(reaction, "conditions", [])
            )
            rows.append(
                "<tr>"
                f"<td>{component_html(reactants)}</td>"
                f"<td>{component_html(conditions)}</td>"
                f"<td>{component_html(products)}</td>"
                "</tr>"
            )
        if not rows:
            return ""
        return "<table><tr><th>reactants</th><th>conditions</th><th>products</th></tr>" + "".join(rows) + "</table>"


@dataclass
class TabularResult(SemanticItem):
    type: LayoutType = LayoutType.Table
    bboxes: List[BBox] = field(default_factory=list)
    labels: List[TableBBoxType] = field(default_factory=list)
    types: List[LayoutType] = field(default_factory=list)
    placeholders: List[str] = field(default_factory=list)
    contents: List[str] = field(default_factory=list)
    structure: str = ""

    def __post_init__(self):
        super().__post_init__()
        assert len(self.placeholders) == len(self.contents)
        if self.bboxes:
            assert len(self.bboxes) == len(self.placeholders)
        if self.labels:
            assert len(self.labels) == len(self.placeholders)
        if self.types:  # 兼容旧版本
            assert len(self.types) == len(self.placeholders)
        if len(self.bboxes) and isinstance(self.bboxes[0], dict):
            self.bboxes = [BBox(**b) for b in self.bboxes]
        if len(self.labels) and isinstance(self.labels[0], int):
            self.labels = [TableBBoxType(i) for i in self.labels]
        if len(self.types):
            self.types = [LayoutType(i) if isinstance(i, (int, str)) else i for i in self.types]

    def _table_html_for_format(self, item_format: FormatFlag) -> str:
        html = self.structure
        content_types = self.types or [LayoutType.Text] * len(self.contents)
        for placeholder, content, content_type in zip(
            self.placeholders[::-1], self.contents[::-1], content_types[::-1]
        ):
            formatted = TextualResult._content_format(str(content), content_type, item_format)
            if item_format != FormatFlag.Html:
                # Preserve table structure while letting the cell text carry
                # markdown/latex/plain inline formatting literally.
                formatted = escape(formatted)
            html = html.replace(placeholder, formatted)
        return html

    def _table_html(self) -> str:
        html = self._table_html_for_format(FormatFlag.Html)
        html = "\n" + html + "\n"
        return html

    @functools.cached_property
    def full_html(self) -> str:
        return self._table_html_for_format(FormatFlag.Html)

    def _df_for_format(self, item_format: FormatFlag) -> pd.DataFrame:
        df = read_html(StringIO(self._table_html_for_format(item_format)))[0]
        df = df.dropna(how="all")  # drop empty 'NaN' rows
        df = df[df.astype(bool).sum(axis=1) > 0]  # remove empty 'blacksapce' rows
        return df

    def _safe_df_for_format(self, item_format: FormatFlag) -> pd.DataFrame:
        try:
            return self._df_for_format(item_format)
        except Exception:
            get_root_logger().warning(
                f"{self.token} {self.page} {self.block} {self.type} "
                f"convert html to {item_format} dataframe error: {repr(self.structure)}"
            )
            return pd.DataFrame()

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        try:
            return self._df_for_format(FormatFlag.Plain)
        except Exception:
            get_root_logger().warning(
                f"{self.token} {self.page} {self.block} {self.type} convert html to dataframe error: {repr(self.structure)}"
            )
            return pd.DataFrame()

    @property
    def plain(self):
        if self.df.shape[-1] == 0:
            return ""
        return self.df.to_markdown(index=False, disable_numparse=True)

    @property
    def markup(self):
        return f"\\begin{{{self.type.value}}}\n{self.plain}\n\\end{{{self.type.value}}}"

    @property
    def markdown(self):
        df = self._safe_df_for_format(FormatFlag.Markdown)
        if df.shape[-1] == 0:
            return ""
        return df.to_markdown(index=False, disable_numparse=True)

    @property
    def latex(self):
        df = self._safe_df_for_format(FormatFlag.Latex)
        if df.shape[-1] == 0:
            return ""
        return df.style.hide(axis="index").to_latex()

    @property
    def html(self):
        try:
            return self.full_html
        except Exception:
            get_root_logger().warning(
                f"{self.token} {self.page} {self.block} {self.type} convert html to dataframe error: {repr(self.full_html)}"
            )
            pass
        return self.df.to_html(index=False)

    def transpose(self):
        super().transpose()
        for b_idx in range(len(self.bboxes)):
            self.bboxes[b_idx].transpose(self.page_size, self.direction)
        return self

    def itranspose(self):
        super().itranspose()
        for b_idx in range(len(self.bboxes)):
            self.bboxes[b_idx].itranspose(self.page_size, self.direction)
        return self


@dataclass
class ChartResult(SemanticItem):
    type: LayoutType = LayoutType.Chart
    data: str = ""

    @functools.cached_property
    def df(self):
        try:
            # chart is not standard markdown table, so we convert it to markdown table
            underlying_df = pd.DataFrame(
                [[col.strip() for col in row.strip("|").split("|")] for row in self.data.split("\n")]
            )
            underlying_df = underlying_df.replace(r"^\s*$", None, regex=True).dropna(how="all")
            if underlying_df.empty:
                return pd.DataFrame()

            underlying_df.columns = underlying_df.iloc[0]
            underlying_df = underlying_df[1:]
            if underlying_df.empty:
                return pd.DataFrame()

            return underlying_df
        except Exception:
            get_root_logger().warning(
                f"{self.token} {self.page} {self.block} {self.type} convert chart to dataframe error: {repr(self.data)}"
            )
            return pd.DataFrame()

    @property
    def plain(self):
        if self.df.shape[-1] == 0:
            return ""
        return self.df.to_markdown(index=False, disable_numparse=True)

    @property
    def markup(self):
        return f"\\begin{{{self.type.value}}}\n{self.plain}\n\\end{{{self.type.value}}}"

    @property
    def markdown(self):
        return self.df.to_markdown(index=False, disable_numparse=True)

    @property
    def latex(self):
        if self.df.shape[-1] == 0:
            return ""
        return self.df.style.hide(axis="index").format(escape="latex").to_latex()

    @property
    def html(self):
        return self.df.to_html(index=False)


@dataclass
class FigureResult(SemanticItem):
    type: LayoutType = LayoutType.Figure
    desc: str = ""

    @property
    def plain(self):
        return self.desc

    @property
    def markup(self):
        return f"\\begin{{{self.type.value}}}\n{self.plain}\n\\end{{{self.type.value}}}"

    @property
    def markdown(self):
        return self.desc

    @property
    def latex(self):
        return self.desc

    @property
    def html(self):
        return self.desc


@dataclass
class EquationResult(SemanticItem):
    type: LayoutType = LayoutType.Equation
    latex_repr: str = ""

    @property
    def plain(self):
        return self.latex_repr

    @property
    def markup(self):
        return f"\\begin{{equation}}\n{self.latex_repr}\n\\end{{equation}}"

    @staticmethod
    def format_latex_repr(latex_repr: str, item_format: FormatFlag, inline: bool = False):
        if item_format == FormatFlag.Plain:
            # only for textual result, not for equation result
            return f"\\({latex_repr}\\)" if inline else f"\\[\n{latex_repr}\n\\]"
        if item_format == FormatFlag.Markdown:
            return f"${latex_repr}$" if inline else f"$$\n{latex_repr}\n$$"
        if item_format == FormatFlag.Latex:
            return f"\\({latex_repr}\\)" if inline else f"\\[\n{latex_repr}\n\\]"
        if item_format == FormatFlag.Html:
            return f"\\({latex_repr}\\)" if inline else f"\\[\n{latex_repr}\n\\]"
        return latex_repr

    @property
    def markdown(self):
        return self.format_latex_repr(self.latex_repr, FormatFlag.Markdown)

    @property
    def latex(self):
        return self.format_latex_repr(self.latex_repr, FormatFlag.Latex)

    @property
    def html(self):
        return self.format_latex_repr(self.latex_repr, FormatFlag.Html)


@dataclass
class GroupedResult(SemanticItem):
    type: LayoutType = LayoutType.Group
    level: int = 1
    method: str = "default"
    items: List[SemanticItem] = field(default_factory=list)

    @property
    def prefix(self):
        # return "--" * 5 * self.level + f"<{self.type}>" + "--" * 5 * self.level + "\n"
        return ""

    @property
    def suffix(self):
        # return "\n" + "--" * 5 * self.level + f"</{self.type}>" + "--" * 5 * self.level
        return ""

    @property
    def plain(self):
        return f"{self.prefix}" + "\n\n".join([item.plain for item in self.items]) + f"{self.suffix}"

    @property
    def markup(self):
        return f"{self.prefix}" + "\n\n".join([item.markup for item in self.items]) + f"{self.suffix}"

    @property
    def markdown(self):
        return f"{self.prefix}" + "\n\n".join([item.markdown for item in self.items]) + f"{self.suffix}"

    @property
    def latex(self):
        return f"{self.prefix}" + "\n\n".join([item.latex for item in self.items]) + f"{self.suffix}"

    @property
    def html(self):
        return f"{self.prefix}" + "\n\n".join([item.html for item in self.items]) + f"{self.suffix}"

    @classmethod
    def clone(cls, item, **extra):
        extra.setdefault("method", "default-clone")
        return super().clone(item, **extra)


if __name__ == "__main__":
    a = LayoutItem(
        token="None",
        lang=Language.English,
        page=1,
        block=1,
        type=LayoutType.Table,
        page_size=[0, 0],
        bbox=BBox(x1=1, y1=2, x2=3, y2=4),
        conf=0.9,
    )
    print(json.dumps(a, indent=4, cls=DataclassJSONEncoder))
    print(a.bbox.xywh)

    b = TextualResult.clone(a)
    b.text = "Hello World"
    c = TextualResult.clone(a, text="Hi")
    d = TextualResult.clone(b, text="Hi again")
    print(f"{a = }")
    print(f"{b = }")
    print(f"{c = }")
    print(f"{d = }")

    e = EquationResult.clone(a, latex_repr="a - b")
    print(e)

    assert BBox(0, 0, 1, 1) + (1, 1) == BBox(1, 1, 2, 2)
    assert BBox(0, 0, 1, 1).intersection(BBox(0, 0, 1, 1)) == BBox(0, 0, 1, 1)
    assert BBox(0, 0, 1, 1).intersection(BBox(1, 1, 2, 2)) == BBox(0, 0, 0, 0)
    assert BBox(0, 0, 1, 1).intersection(BBox(0.5, 0.5, 1.5, 1.5)) == BBox(0.5, 0.5, 1, 1)

    assert BBox(0, 0, 1, 1).iou(BBox(0, 0, 1, 1)) == 1
    assert BBox(0, 0, 1, 1).iou(BBox(1, 1, 2, 2)) == 0
    assert BBox(0, 0, 1, 1).iou(BBox(0.5, 0.5, 1.5, 1.5)) == 0.25 / (1 + 1 - 0.25)

    assert BBox(0, 0, 1, 1).iof(BBox(0, 0, 1, 1)) == 1
    assert BBox(0, 0, 1, 1).iof(BBox(1, 1, 2, 2)) == 0
    assert BBox(0, 0, 1, 1).iof(BBox(0.5, 0.5, 1.5, 1.5)) == 0.25 / 1

    assert BBox(1, 1, 2, 2) * 4 == BBox(4, 4, 8, 8)
    assert BBox(1, 1, 2, 2) / 2 == BBox(0.5, 0.5, 1, 1)
    assert BBox(1, 1, 2, 2) * [2, 4] == BBox(2, 4, 4, 8)
    assert BBox(1, 1, 2, 2) / [2, 4] == BBox(0.5, 0.25, 1, 0.5)
    assert BBox(1, 1, 2, 2) * (2, 4) == BBox(2, 4, 4, 8)
    assert BBox(1, 1, 2, 2) / (2, 4) == BBox(0.5, 0.25, 1, 0.5)
    assert BBox(1, 1, 2, 2).ctr.tuple == (1.5, 1.5)
    assert BBox(1, 1, 2, 2).ctr.round.tuple == (2, 2)

    assert len(BBox(1, 1, 2, 2)) == 4
    assert BBox(1, 1, 2, 2)[0] == 1
    assert BBox(1, 1, 2, 2)[2] == 2

    assert Point(1, 1) + Point(1, 1) == Point(2, 2)
    assert Point(1, 1) - Point(1, 1) == Point(0, 0)
    assert Point(1, 1) * 2 == Point(2, 2)
    assert Point(1, 1) / 2 == Point(0.5, 0.5)

    assert len(Point(1, 1)) == 2
    assert Point(1, 2)[0] == 1
    assert Point(1, 2)[1] == 2

    assert BBox(0, 0, 4, 4) - Point(1, 1) == BBox(-1, -1, 3, 3)
    assert BBox(0, 0, 4, 4) + Point(1, 1) == BBox(1, 1, 5, 5)
    assert BBox(0, 0, 4, 4) - (1, 1) == BBox(-1, -1, 3, 3)
    assert BBox(0, 0, 4, 4) + (1, 1) == BBox(1, 1, 5, 5)
    assert (BBox(0, 0, 4, 4) + (1, 1)).ctr.int.tuple == (3, 3)

    b = BBox(1, 1, 2, 2) / (2, 4)
    b.xyxy

    # output from expr server
    reaction = {
        "conditions": [
            {
                "bbox": [0.26163081540770383, 0.5762757922170962, 0.27963981990995496, 0.6404893804927154],
                "category": "[Idt]",
                "category_id": 3,
                "text": "a,b",
            }
        ],
        "products": [
            {
                "bbox": [0.33366683341670833, 0.047748565640845106, 0.5302651325662832, 0.7458655253552702],
                "category": "[Mol]",
                "category_id": 1,
                "text": "Brc1cc2[nH]cnc2cn1",
            }
        ],
        "reactants": [
            {
                "bbox": [0.2351175587793897, 0.011525515844341923, 0.3371685842921461, 0.49889018583365746],
                "category": "[Mol]",
                "category_id": 1,
                "text": "c1(C)c(C)cc(C)cc1C",
            }
        ],
    }
    reaction = Reaction(**reaction)
    print(reaction)

    for i in range(len(reaction)):
        print(reaction[i])

    for c in reaction:
        print(c)

    expr = ExpressionResult.clone(a)
    expr.reactions = [reaction] * 2
    print(expr.plain)
    # output
    [
        {
            "prev_mols": ["Cc1nc(-c2cccc3[nH]ccc23)cc2nccn12"],
            "post_mols": ["C[C@@H]1COCCN1c1nc(-c2nncc3[nH]ccc23)cc2c1ncn2C"],
            "condition": {"text": "", "catalyst": []},
        }
    ]
    [
        {
            "prev_mols": ["c1(C)c(C)cc(C)cc1C"],
            "post_mols": ["Brc1cc2[nH]cnc2cn1"],
            "condition": {"text": "a,b", "catalyst": []},
        },
        {
            "prev_mols": ["Brc1cc2[nH]cnc2cn1"],
            "post_mols": ["C[C@@H]1COCCN1c1nc(Br)cc2[nH]cnc12"],
            "condition": {"text": "", "catalyst": []},
        },
        {
            "prev_mols": ["C[C@@H]1COCCN1c1nc(Br)cc2[nH]cnc12"],
            "post_mols": ["C[C@@H]1COCCN1c1nc(Br)cc2c1ncn2C"],
            "condition": {"text": "", "catalyst": []},
        },
        {
            "prev_mols": ["C[C@@H]1COCCN1c1nc(Br)cc2c1ncn2C"],
            "post_mols": ["Cn1cnc2cc(-c3nncc4c3ccn4S(=O)(=O)c3ccc(C)cc3)ncc21"],
            "condition": {"text": "e,f", "catalyst": []},
        },
        {
            "prev_mols": ["Cn1cnc2cc(-c3nncc4c3ccn4S(=O)(=O)c3ccc(C)cc3)ncc21"],
            "post_mols": ["CC(C)N(C)c1cn(C)c2cc(-c3nncc4[nH]ccc34)nnc12"],
            "condition": {"text": "", "catalyst": []},
        },
    ]
    # expr input
    {
        "bbox": [0.9394697348674337, 0.540451023516746, 0.9939969984992496, 0.6319119659580414],
        "category": "[Idt]",
        "category_id": 3,
        "text": "",
        "smiles": "",
    }

    bar = ProcessBar()
    print({**bar.dict()})

    bar.total_chart += 1
    bar.proc_chart += 1
    print({**bar.dict()})
    print(bar.total())
    print(bar.proc())

    bar += bar
    bar = bar + bar
    print({**bar.dict()})
    print(bar.total())
    print(bar.proc())

    print(bar["total_chart"])
