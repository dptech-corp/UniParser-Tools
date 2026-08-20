"""Formatting helpers shared by release/v1.3 result-model conversion."""

import html
import re
from typing import Any, Dict, List, Tuple

from bs4 import BeautifulSoup

from uniparser_tools.common.constant import LayoutType, TableBBoxType


def parse_inline_text(text: str) -> Tuple[List[str], List[LayoutType]]:
    """Split rich text into plain-text, molecule, and inline-equation spans."""
    if not text:
        return [], []

    latex_marker = re.compile(r"\\[A-Za-z]+|[\^_{}=]|[A-Za-z]\s*[+\-*/=<>]\s*[A-Za-z0-9\\]")
    pattern = re.compile(
        r"(?P<molecule>\*\*\*(?P<molecule_text>.+?)\*\*\*)|"
        r"(?P<equation>"
        r"\\\[(?P<eq_display_text>.+?)\\\]|"
        r"\\\((?P<eq_inline_text>.+?)\\\)|"
        r"\$\$(?P<eq_dollar_display_text>.+?)\$\$|"
        r"\$(?P<eq_dollar_inline_text>(?:\\.|[^$\\\n])+?)\$"
        r")",
        re.DOTALL,
    )
    contents: List[str] = []
    types: List[LayoutType] = []
    pos = 0
    for match in pattern.finditer(text):
        if match.start() > pos:
            contents.append(text[pos : match.start()])
            types.append(LayoutType.Text)

        if match.group("molecule") is not None:
            contents.append(match.group("molecule_text").strip())
            types.append(LayoutType.Molecule)
        else:
            equation_text = (
                match.group("eq_inline_text")
                or match.group("eq_display_text")
                or match.group("eq_dollar_inline_text")
                or match.group("eq_dollar_display_text")
                or ""
            )
            if (
                match.group("eq_dollar_inline_text") or match.group("eq_dollar_display_text")
            ) and not latex_marker.search(equation_text):
                contents.append(match.group(0))
                types.append(LayoutType.Text)
                pos = match.end()
                continue
            contents.append(equation_text.strip())
            types.append(LayoutType.EquationInline)
        pos = match.end()

    if pos < len(text):
        contents.append(text[pos:])
        types.append(LayoutType.Text)

    merged_contents: List[str] = []
    merged_types: List[LayoutType] = []
    for content, content_type in zip(contents, types):
        if merged_types and merged_types[-1] == content_type == LayoutType.Text:
            merged_contents[-1] += content
        else:
            merged_contents.append(content)
            merged_types.append(content_type)
    return merged_contents, merged_types


def parse_table_full_html(html_text: str) -> Dict[str, Any]:
    """Convert a full HTML table into release/v1.3 placeholder spans."""
    soup = BeautifulSoup(html_text, "html.parser")
    placeholders: List[str] = []
    contents: List[str] = []
    types: List[LayoutType] = []

    for cell_idx, cell in enumerate(soup.find_all(["td", "th"])):
        span_contents, span_types = parse_inline_text(cell.decode_contents())
        if not span_contents:
            continue

        cell_placeholders = []
        for span_idx, (span_content, span_type) in enumerate(zip(span_contents, span_types)):
            placeholder = f"[[VL_TABLE_{cell_idx}_{span_idx}]]"
            placeholders.append(placeholder)
            if span_type == LayoutType.Text:
                span_content = BeautifulSoup(span_content, "html.parser").get_text("", strip=False)
            else:
                span_content = html.unescape(span_content)
            contents.append(span_content)
            types.append(span_type)
            cell_placeholders.append(placeholder)
        cell.clear()
        cell.append("".join(cell_placeholders))

    return {
        "bboxes": [],
        "labels": [TableBBoxType.Content] * len(placeholders),
        "types": types,
        "placeholders": placeholders,
        "contents": contents,
        "structure": str(soup),
    }
