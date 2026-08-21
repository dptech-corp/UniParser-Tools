"""Scientific-paper parse preset (matches uniparser_tools.cli.core.parse_options)."""

from __future__ import annotations


SCIENTIFIC_PAPER_TRIGGER: dict[str, object] = {
    "lang": "unknown",
    "sync": True,
    "timeout": 1800,
    "padding_snip": True,
    "inplace_update": False,
    "preset_layout": "",
    "textual": 2,
    "equation": 2,
    "table": 2,
    "chart": -1,
    "figure": -1,
    "expression": -1,
    "molecule": 1,
    "ordering_method": "xy_cut_exp",
}
