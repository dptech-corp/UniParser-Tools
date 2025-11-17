from __future__ import print_function

import collections
import io
import os
import re
import struct
import unicodedata
from copy import deepcopy
from io import StringIO
from typing import Dict, List, Union

from uniparser_tools.common.constant import LayoutType
from uniparser_tools.common.dataclass import GroupedResult, SemanticItem, TextualResult
from uniparser_tools.utils.log import get_root_logger


FILE_UNKNOWN = "Sorry, don't know how to get size for this file."


def flat_layout(item: Union[Dict, SemanticItem]) -> List[Union[Dict, SemanticItem]]:
    item = deepcopy(item)
    if isinstance(item, dict) and "items" in item:
        items = item.pop("items")
        item["items"] = []
        return [item] + [ii for i in items for ii in flat_layout(i)]
    elif isinstance(item, GroupedResult):
        items = item.items
        item.items = []  # inplace clear
        return [item] + [ii for i in items for ii in flat_layout(i)]
    else:
        return [item]


def truncate_string(s, front=20, back=10):
    if len(s) <= front + back:
        return s
    else:
        return s[:front] + "..." + s[-back:]


def is_head_of_paragraph(text: str):
    # 10-[A-Z]
    # [a-z] [A-Z]
    LINE_HEAD_FLAG = tuple(chr(i) for i in range(ord("A"), ord("Z") + 1)) + ("#",)
    t = 0
    while text and text[0] in ["(", "（", "[", "【"]:
        t += 1
        if text[0] == "(":
            s, e = "\(", "\)"
        elif text[0] == "（":
            s, e = "（", "）"
        elif text[0] == "[":
            s, e = "\[", "\]"
        elif text[0] == "【":
            s, e = "【", "】"
        else:
            break
        if t > 10:
            break
        text = re.sub(rf"^{s}[^{e}]{{0,64}}{e}", "", text)
    t = 0
    while True:
        t += 1
        for s, e in [
            ("\(", "\)"),
            ("（", "）"),
            ("\[", "\]"),
            ("【", "】"),
        ]:
            if e.strip("\\") in text[:5]:
                text = re.sub(rf"^[^{e}]{{0,64}}{e}", "", text)
        if t > 10:
            break
    text = re.sub(r"^[\d]{1,4}[-]?", "", text)
    text = re.sub(r"^[a-z#*†‡§¶⁰¹²³⁴⁵⁶⁷⁸⁹\s]{0,2}([A-Z])", "\g<1>", text.lstrip(" \n\r\t"))
    return text.startswith(LINE_HEAD_FLAG)


def is_tail_of_paragraph(text: str):
    LINE_TAIL_FLAG = (".", "!", "?", "。", "！", "？", '"', "”", ":", "：", ";", "；")
    t = 0
    while text and text[-1] in [")", "）", "]", "】"]:
        t += 1
        if text[-1] == ")":
            s, e = "\(", "\)"
        elif text[-1] == "）":
            s, e = "（", "）"
        elif text[-1] == "]":
            s, e = "\[", "\]"
        elif text[-1] == "】":
            s, e = "【", "】"
        else:
            break
        if t > 10:
            break
        text = re.sub(rf"{s}[^{e}]{{0,64}}{e}$", "", text)
    text = text.strip(" \n\r\t")
    return text.endswith(LINE_TAIL_FLAG)


def clean_scientific_text(text: str, strict: bool = False) -> str:
    """
    文献文本清洗 pipeline:
    1. Unicode 标准化 (NFKC)
    2. 删除不可见/控制字符
    3. 替换常见符号
    4. 严格模式下仅保留 ASCII/中文/常见标点
    """
    if not text:
        return ""

    # 1. Unicode 标准化 (连字 -> 普通字母，全角 -> 半角)
    text = unicodedata.normalize("NFKC", text)

    # 2. 删除不可见字符（零宽空格、BOM、控制符）
    # 附录1 字符类型
    # [Cc] Other, Control
    # [Cf] Other, Format
    # [Cn] Other, Not Assigned (no characters in the file have this property)
    # [Co] Other, Private Use
    # [Cs] Other, Surrogate
    # [LC] Letter, Cased
    # [Ll] Letter, Lowercase
    # [Lm] Letter, Modifier
    # [Lo] Letter, Other
    # [Lt] Letter, Titlecase
    # [Lu] Letter, Uppercase
    # [Mc] Mark, Spacing Combining
    # [Me] Mark, Enclosing
    # [Mn] Mark, Nonspacing
    # [Nd] Number, Decimal Digit
    # [Nl] Number, Letter
    # [No] Number, Other
    # [Pc] Punctuation, Connector
    # [Pd] Punctuation, Dash
    # [Pe] Punctuation, Close
    # [Pf] Punctuation, Final quote (may behave like Ps or Pe depending on usage)
    # [Pi] Punctuation, Initial quote (may behave like Ps or Pe depending on usage)
    # [Po] Punctuation, Other
    # [Ps] Punctuation, Open
    # [Sc] Symbol, Currency
    # [Sk] Symbol, Modifier
    # [Sm] Symbol, Math
    # [So] Symbol, Other
    # [Zl] Separator, Line
    # [Zp] Separator, Paragraph
    # [Zs] Separator, Space

    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[\u2000-\u200f\u202a-\u202f\u205f\u3000]", " ", text)

    # 3. 替换常见符号
    replacements = {
        "\u2013": "-",  # en-dash
        "\u2014": "-",  # em-dash
        "\u2015": "-",  # horizontal bar
        "\u2212": "-",  # minus sign
        "\u00a0": " ",  # non-breaking space
        "\u2026": "...",  # ellipsis
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # 4. 严格模式：仅保留常用字符
    if strict:
        text = "".join(
            ch
            for ch in text
            if (
                0x20 <= ord(ch) <= 0x7E  # ASCII
                or 0x4E00 <= ord(ch) <= 0x9FFF  # 中文
                or 0x3000 <= ord(ch) <= 0x303F  # 常见中文标点
            )
        )

    # 去掉多余空格
    text = re.sub(r"\s+", " ", text).strip()

    return text


# 替换 "Figures 3 and 4" 为 "Figure 3 and figure 4"
def replace_figures(
    text: str,
    pattern=r"(figures|figs|\$figures)[\s\.]+([S]?)(\d+)[A-Za-z]?\s*(and|,|or|to|-|~|&)\s*([S]?)(\d+)",
):
    def replace_figures_(match):
        # 提取数字
        fig: str = match.group(1)[:-1]
        suf1: str = match.group(2)
        num1: str = match.group(3)
        conj: str = match.group(4)
        suf2: str = match.group(5)
        num2: str = match.group(6)

        if fig.startswith("$"):
            suf1 = "S"
            suf2 = "S"

        try:
            if conj in ["to", "-", "~"] and int(num1) < int(num2) - 1:
                return f"{fig} {suf1}" + f" and {fig.lower()} {suf1}".join(
                    [str(num) for num in range(int(num1), int(num2) + 1)]
                )
        except Exception:
            get_root_logger().exception("replace_figures error")
        return f"{fig} {suf1}{num1} {conj} {fig.lower()}s {suf2}{num2}"

    return re.sub(pattern, replace_figures_, text, flags=re.I)


def find_figure_caption_kws(captions: Union[str, List[str]]) -> List[str]:
    # 提取关键词
    # “补充图”或“补充材料”编号风格，常见于论文的 Supplementary Figures
    # Figure S1, Figure S2A
    # 图2B, 2G和附图S2A

    if isinstance(captions, str):
        captions = [captions]

    kws: List[str] = []
    for cap in captions:
        # 连接断掉的单词
        cap = re.sub(r"\((\d+[a-z]*?)\)", r"\1", cap, flags=re.I)  # (1a) -> 1a
        cap = re.sub(r"(\w)-[\s](\w)", r"\1\2", cap)
        # TODO: Supplementary Figure 4
        # cap = re.sub(r"supplementary figure([s])?[\s\.]*(\d+)", r"figure\1 S\2", cap, flags=re.I)
        cap = re.sub(r"supplementary figure", "$figure", cap, flags=re.I)
        # 进行两次操作，展开figures 4-5 ｜ figures 4 and 5 ｜ figures 4 and 5 and 6
        cap = replace_figures(cap)
        cap = replace_figures(cap)
        cap = cap.replace("figures", "figure").replace("figs", "fig")
        # 替换单独的Supplementary Figure 4
        cap = re.sub(r"\$figure[\s\.]+(\d+)", r"figure S\1", cap, flags=re.I)
        cap = cap.replace("$figures", "figure")
        # "Figure1 shows the results.",
        # "Fig1 shows the results.",
        # "Figure 1 shows the results.",
        # "Fig 1 shows the results.",
        # "Figure.1 shows the results.",
        # "Fig.1 shows the results.",
        # "Figure-1 shows the results.",
        # "Fig-1 shows the results.",
        # "Figure 1a shows the results.",
        matches = re.findall(r"(fig(?:ure)?[\.\-\s]*[S]?[\d]+)", cap, flags=re.I)
        kws.extend(matches)
        matches = re.findall(r"(sche(?:me)?[\.\-\s]*[S]?[\d]+)", cap, flags=re.I)
        kws.extend(matches)
    unique_kws = []
    for kw in kws:  # keep in order
        kw = re.sub(r"(\s)S(\d+)", r"\1$\2", kw)  # 替换 "S1" 为 "$1"
        kw = kw.replace(" ", "").replace("-", "").replace(".", "")
        kw = kw.lower().replace("figure", "fig").capitalize()
        if kw not in unique_kws:
            unique_kws.append(kw)
    return unique_kws


def recursive_required_items(
    item: SemanticItem,
    required_types: List[LayoutType] = [
        LayoutType.DocumentTitle,
        LayoutType.Title,
        LayoutType.Paragraph,
        LayoutType.ImageCaption,
        LayoutType.ImageFootnote,
        LayoutType.TableCaption,
        LayoutType.TableFootnote,
    ],
) -> List[SemanticItem]:
    if isinstance(item, GroupedResult):
        items = [item for it in item.items for item in recursive_required_items(it, required_types)]
        return items
    elif isinstance(item, TextualResult):
        if item.type in required_types:
            return [item]
        else:
            return []
    else:
        return []


def recursive_required_content(
    item: SemanticItem,
    required_types: List[LayoutType] = [
        LayoutType.DocumentTitle,
        LayoutType.Title,
        LayoutType.Paragraph,
        LayoutType.ImageCaption,
        LayoutType.ImageFootnote,
        LayoutType.TableCaption,
        LayoutType.TableFootnote,
    ],
    clean: bool = True,
) -> str:
    content = " ".join([i.plain.strip() for i in recursive_required_items(item, required_types)])
    if clean:
        return clean_scientific_text(content)
    else:
        return content


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

get_image_size.py
====================

    :Name:        get_image_size
    :Purpose:     extract image dimensions given a file path

    :Author:      Paulo Scardine (based on code from Emmanuel VAÏSSE)

    :Created:     26/09/2013
    :Copyright:   (c) Paulo Scardine 2013
    :Licence:     MIT

"""


class UnknownImageFormat(Exception):
    pass


types = collections.OrderedDict()
BMP = types["BMP"] = "BMP"
GIF = types["GIF"] = "GIF"
ICO = types["ICO"] = "ICO"
JPEG = types["JPEG"] = "JPEG"
PNG = types["PNG"] = "PNG"
TIFF = types["TIFF"] = "TIFF"

image_fields = ["path", "type", "file_size", "width", "height"]


def get_image_size(file_path):
    """
    Return an `Image` object for a given img file content - no external
    dependencies except the os and struct builtin modules

    Args:
        file_path (str): path to an image file

    Returns:
        Image: (width, height)
    """
    size = os.path.getsize(file_path)

    # be explicit with open arguments - we need binary mode
    with io.open(file_path, "rb") as input:
        """
        Return an `Image` object for a given img file content - no external
        dependencies except the os and struct builtin modules

        Args:
            input (io.IOBase): io object support read & seek
            size (int): size of buffer in byte
            file_path (str): path to an image file

        Returns:
            Image: (width, height)
        """
        height = -1
        width = -1
        data = input.read(26)
        msg = " raised while trying to decode as JPEG."

        if (size >= 10) and data[:6] in (b"GIF87a", b"GIF89a"):
            # GIFs
            imgtype = GIF
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif (size >= 24) and data.startswith(b"\211PNG\r\n\032\n") and (data[12:16] == b"IHDR"):
            # PNGs
            imgtype = PNG
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith(b"\211PNG\r\n\032\n"):
            # older PNGs
            imgtype = PNG
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith(b"\377\330"):
            # JPEG
            imgtype = JPEG
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while b and ord(b) != 0xDA:
                    while ord(b) != 0xFF:
                        b = input.read(1)
                    while ord(b) == 0xFF:
                        b = input.read(1)
                    if ord(b) >= 0xC0 and ord(b) <= 0xC3:
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0]) - 2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        elif (size >= 26) and data.startswith(b"BM"):
            # BMP
            imgtype = "BMP"
            headersize = struct.unpack("<I", data[14:18])[0]
            if headersize == 12:
                w, h = struct.unpack("<HH", data[18:22])
                width = int(w)
                height = int(h)
            elif headersize >= 40:
                w, h = struct.unpack("<ii", data[18:26])
                width = int(w)
                # as h is negative when stored upside down
                height = abs(int(h))
            else:
                raise UnknownImageFormat("Unkown DIB header size:" + str(headersize))
        elif (size >= 8) and data[:4] in (b"II\052\000", b"MM\000\052"):
            # Standard TIFF, big- or little-endian
            # BigTIFF and other different but TIFF-like formats are not
            # supported currently
            imgtype = TIFF
            byteOrder = data[:2]
            boChar = ">" if byteOrder == "MM" else "<"
            # maps TIFF type id to size (in bytes)
            # and python format char for struct
            tiffTypes = {
                1: (1, boChar + "B"),  # BYTE
                2: (1, boChar + "c"),  # ASCII
                3: (2, boChar + "H"),  # SHORT
                4: (4, boChar + "L"),  # LONG
                5: (8, boChar + "LL"),  # RATIONAL
                6: (1, boChar + "b"),  # SBYTE
                7: (1, boChar + "c"),  # UNDEFINED
                8: (2, boChar + "h"),  # SSHORT
                9: (4, boChar + "l"),  # SLONG
                10: (8, boChar + "ll"),  # SRATIONAL
                11: (4, boChar + "f"),  # FLOAT
                12: (8, boChar + "d"),  # DOUBLE
            }
            ifdOffset = struct.unpack(boChar + "L", data[4:8])[0]
            try:
                countSize = 2
                input.seek(ifdOffset)
                ec = input.read(countSize)
                ifdEntryCount = struct.unpack(boChar + "H", ec)[0]
                # 2 bytes: TagId + 2 bytes: type + 4 bytes: count of values + 4
                # bytes: value offset
                ifdEntrySize = 12
                for i in range(ifdEntryCount):
                    entryOffset = ifdOffset + countSize + i * ifdEntrySize
                    input.seek(entryOffset)
                    tag = input.read(2)
                    tag = struct.unpack(boChar + "H", tag)[0]
                    if tag == 256 or tag == 257:
                        # if type indicates that value fits into 4 bytes, value
                        # offset is not an offset but value itself
                        type = input.read(2)
                        type = struct.unpack(boChar + "H", type)[0]
                        if type not in tiffTypes:
                            raise UnknownImageFormat("Unkown TIFF field type:" + str(type))
                        typeSize = tiffTypes[type][0]
                        typeChar = tiffTypes[type][1]
                        input.seek(entryOffset + 8)
                        value = input.read(typeSize)
                        value = int(struct.unpack(typeChar, value)[0])
                        if tag == 256:
                            width = value
                        else:
                            height = value
                    if width > -1 and height > -1:
                        break
            except Exception as e:
                raise UnknownImageFormat(str(e))
        elif size >= 2:
            # see http://en.wikipedia.org/wiki/ICO_(file_format)
            imgtype = "ICO"  # noqa
            input.seek(0)
            reserved = input.read(2)
            if 0 != struct.unpack("<H", reserved)[0]:
                raise UnknownImageFormat(FILE_UNKNOWN)
            format = input.read(2)
            assert 1 == struct.unpack("<H", format)[0]
            num = input.read(2)
            num = struct.unpack("<H", num)[0]
            if num > 1:
                import warnings

                warnings.warn("ICO File contains more than one image")
            # http://msdn.microsoft.com/en-us/library/ms997538.aspx
            w = input.read(1)
            h = input.read(1)
            width = ord(w)
            height = ord(h)
        else:
            raise UnknownImageFormat(FILE_UNKNOWN)

        return width, height


def tree_repr(item: GroupedResult, output_str: StringIO = None, prefix: str = "") -> str:
    # ├─ │ └─ 绘制树，树名为其type，子节点为items
    # group
    # ├── image
    # │   ├── figure group
    # │   │   ├── figure
    # │   │   ├── chart
    # │   │   └── legend
    # │   ├── figure group
    # │   │   ├── figure
    # │   │   └── legend
    # │   └── figure group
    # │       ├── expression
    # │       └── legend
    # └── image caption

    # 打印当前节点
    if output_str is None:
        output_str = StringIO()
    if not prefix:
        output_str.write(f"{prefix}{item.type}\n")
    if isinstance(item, GroupedResult) and item.items:
        last_idx = len(item.items) - 1
        for idx, child in enumerate(item.items):
            if idx == last_idx:
                branch = "└─ "
                next_prefix = prefix + "   "
            else:
                branch = "├─ "
                next_prefix = prefix + "│  "
            if isinstance(child, GroupedResult) and child.items:
                output_str.write(f"{prefix}{branch}{child.type}\n")
                tree_repr(child, output_str=output_str, prefix=next_prefix)
            else:
                output_str.write(f"{prefix}{branch}{child.type}\n")
    return output_str.getvalue()


if __name__ == "__main__":
    from uniparser_tools.utils.convert import dict2obj  # noqa

    haedcases = [
        # ["Figs. 3 and 4", ["Fig3", "Fig4"]],  # ✅
        # ["Figs. 13 and 14 ", ["Fig13", "Fig14"]],  # ✅
        # ["Figs. 20–22", ["Fig20", "Fig21", "Fig22"]],  # ✅
        # ["FIG. 14或FIG. 22", ["Fig14", "Fig22"]],  # ✅
        # ["Figs. 1 and 2", ["Fig1", "Fig2"]],  # ✅
        # ["Figs 1,2,4", ["Fig1", "Fig2", "Fig4"]],  # ✅
        # ["Figs 1,2 and 4", ["Fig1", "Fig2", "Fig4"]],  # ✅
        # ["Figs 1,2 & 4", ["Fig1", "Fig2", "Fig4"]],  # ✅
        # ["FIG 7捕获不到Figs. 5 and 7", ["Fig7", "Fig5"]],  # ✅
        # ["Figure 5 捕获不到 Figs. 4–7", ["Fig5", "Fig4", "Fig6", "Fig7"]],  # ✅
        # ["换行变成了Fig- ure 4", ["Fig4"]],  # ✅
        # ["Figures 13 to 15", ["Fig13", "Fig14", "Fig15"]],  # ✅
        # ["Figures 5 or 6", ["Fig5", "Fig6"]],  # ✅
        # ["Figures 7 and 8 or 9.", ["Fig7", "Fig8", "Fig9"]],  # ✅
        # ["Figures 13-15.", ["Fig13", "Fig14", "Fig15"]],  # ✅
        # ["Figs. 3 and 4.", ["Fig3", "Fig4"]],  # ✅
        # ["Figs. 13 and 14.", ["Fig13", "Fig14"]],  # ✅
        # ["Figs. 20–22", ["Fig20", "Fig21", "Fig22"]],  # ✅
        # ["Figs. 5 and 6", ["Fig5", "Fig6"]],  # ✅
        # ["Figures 3 and 4 provide an overview of sample processing and endpoints.", ["Fig3", "Fig4"]],
        # ["Figure 12捕获不到Figs. S11 to S13", ["Fig12", "Fig$11", "Fig$12", "Fig$13"]],  # ✅
        # ["Figs. 5 and 6 and Figs. S11 to S13", ["Fig5", "Fig6", "Fig$11", "Fig$12", "Fig$13"]],  # ✅
        # ["Figs. S11 to S13", ["Fig$11", "Fig$12", "Fig$13"]],  # ✅
        # ["Figure S1. Related to Figure 1A-D", ["Fig$1", "Fig1"]],  # ✅
        # ["Figure S2A", ["Fig$2"]],  # ✅
        # ["Figures S2A and 2A", ["Fig$2", "Fig2"]],  # ✅
        # ["Figures S2A and 2B", ["Fig$2", "Fig2"]],  # ✅
        # ["Figures S2A and 2C", ["Fig$2", "Fig2"]],  # ✅
        # ["figures S4 and 5", ["Fig$4", "Fig5"]],  # ✅
        # ["Supplementary Figure 4", ["Fig$4"]],  # ✅
        # ["Supplementary Figures 4 to 6", ["Fig$4", "Fig$5", "Fig$6"]],  #  ✅
        # ["Supplementary Figures 4 and 5", ["Fig$4", "Fig$5"]],  # ✅
        # ["Supplementary Figures 4,5,6", ["Fig$4", "Fig$5", "Fig$6"]],  # ✅
        [
            "In Fig. (6) we show quenching of the NVs’ fluorescence when we put a layer of BHQ3 on the diamond’s surface. We attribute this quenching to FRET coupling between NVs and BHQ3 molecules. However, quenching can be caused by other effects.In Fig. (12) we show optical spectrum of our shallow NVs with and without the BHQ3 layer. We point out that the spectra of the NVs in both cases have the same profile and there is no visible trace of NV0 spectrum. Moreover, since both spectra are those of NV−, we know that brightness measured in Fig. (6a) is coming from the NV layer and not from contamination on the diamonds surface.",
            ["Fig6", "Fig12"],
        ],
    ]

    for cap, gt in haedcases:
        ans = find_figure_caption_kws([clean_scientific_text(cap)])
        if ans != gt:
            print(f"cap: {cap}")
            print(f"ans: {ans}")
            print(f"gt: {gt}")
            print(f"rcap: {replace_figures(clean_scientific_text(cap))}")
            print("===")
