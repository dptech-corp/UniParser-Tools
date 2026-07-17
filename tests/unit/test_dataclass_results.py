"""Tests for the semantic result dataclasses.

These are the primary consumers of pandas (TabularResult/ExpressionResult/ChartResult)
and therefore the most likely place where the pandas 1.5.3 -> 2.3.3 upgrade
could introduce regressions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import make_chart_data, make_reaction_dict, make_tabular_payload
from uniparser_tools.common.constant import LayoutType
from uniparser_tools.common.dataclass import (
    BBox,
    ChartResult,
    EquationResult,
    ExpressionResult,
    FigureResult,
    GroupedResult,
    MoleculeResult,
    Reaction,
    TabularResult,
    TextualResult,
)


ITEM_KWARGS = dict(
    token="tok",
    page=0,
    block=0,
    conf=1.0,
    bbox=BBox(0.0, 0.0, 1.0, 1.0),
    page_size=(100, 100),
    type=LayoutType.Text,
)


class TestTextualResult:
    def test_plain_is_text_field(self) -> None:
        r = TextualResult(
            **ITEM_KWARGS,
            bboxes=[BBox(0, 0, 1, 1)],
            contents=["hello"],
            text="hello",
        )
        assert r.plain == "hello"

    def test_bboxes_from_dict_are_coerced(self) -> None:
        r = TextualResult(
            **ITEM_KWARGS,
            bboxes=[{"x1": 0, "y1": 0, "x2": 1, "y2": 1}],
            contents=["x"],
            text="x",
        )
        assert isinstance(r.bboxes[0], BBox)

    def test_title_markdown_gets_heading(self) -> None:
        kw = {**ITEM_KWARGS, "type": LayoutType.Title}
        r = TextualResult(**kw, bboxes=[], contents=[], text="Chapter 1")
        assert r.markdown == "# Chapter 1"

    def test_title_html_uses_h2(self) -> None:
        kw = {**ITEM_KWARGS, "type": LayoutType.Title}
        r = TextualResult(**kw, bboxes=[], contents=[], text="Chapter 1")
        assert r.html == "<h2>Chapter 1</h2>"

    def test_document_title_latex_uses_title_macro(self) -> None:
        kw = {**ITEM_KWARGS, "type": LayoutType.DocumentTitle}
        r = TextualResult(**kw, bboxes=[], contents=[], text="Doc")
        assert r.latex == "\\title{Doc}"


class TestTabularResult:
    @pytest.fixture()
    def simple_tabular(self) -> TabularResult:
        structure = (
            "<table>"
            "<thead><tr><th>name</th><th>value</th></tr></thead>"
            "<tbody>"
            "<tr><td>##P0##</td><td>##P1##</td></tr>"
            "<tr><td>##P2##</td><td>##P3##</td></tr>"
            "</tbody>"
            "</table>"
        )
        payload = make_tabular_payload(
            structure=structure,
            placeholders=["##P0##", "##P1##", "##P2##", "##P3##"],
            contents=["alpha", "1", "beta", "2"],
        )
        return TabularResult(**payload)

    def test_df_is_dataframe_and_contains_content(self, simple_tabular: TabularResult) -> None:
        df = simple_tabular.df
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["name", "value"]
        assert set(df["name"].astype(str)) == {"alpha", "beta"}

    def test_full_html_substitutes_placeholders(self, simple_tabular: TabularResult) -> None:
        html = simple_tabular.full_html
        assert "##P0##" not in html
        assert "alpha" in html and "beta" in html

    def test_markdown_uses_tabulate_pipe_syntax(self, simple_tabular: TabularResult) -> None:
        md = simple_tabular.markdown
        assert md.startswith("|")
        assert "alpha" in md and "beta" in md
        assert "name" in md

    def test_plain_is_empty_when_df_has_no_columns(self) -> None:
        payload = make_tabular_payload(structure="<table></table>", placeholders=[], contents=[])
        r = TabularResult(**payload)
        assert r.plain == ""

    def test_latex_returns_tabular_env(self, simple_tabular: TabularResult) -> None:
        latex = simple_tabular.latex
        assert "\\begin{tabular}" in latex
        assert "\\end{tabular}" in latex

    def test_html_returns_full_html(self, simple_tabular: TabularResult) -> None:
        assert "<table>" in simple_tabular.html
        assert "alpha" in simple_tabular.html

    def test_labels_are_coerced_from_int(self) -> None:
        payload = make_tabular_payload(
            structure="<table><tr><td>##P0##</td></tr></table>",
            placeholders=["##P0##"],
            contents=["x"],
        )
        r = TabularResult(**payload)
        from uniparser_tools.common.constant import TableBBoxType

        assert all(isinstance(lb, TableBBoxType) for lb in r.labels)


class TestExpressionResult:
    @pytest.fixture()
    def expr_single(self) -> ExpressionResult:
        return ExpressionResult(
            **{**ITEM_KWARGS, "type": LayoutType.Expression},
            reactions=[make_reaction_dict(("A",), ("B",), ("cat",))],
        )

    def test_reactions_coerced_from_dict(self, expr_single: ExpressionResult) -> None:
        assert isinstance(expr_single.reactions[0], Reaction)

    def test_df_columns(self, expr_single: ExpressionResult) -> None:
        df = expr_single.df
        assert list(df.columns) == ["No.", "reactants", "products", "conditions"]
        assert df.shape[0] == 1

    def test_empty_reactions_yields_empty_df(self) -> None:
        r = ExpressionResult(
            **{**ITEM_KWARGS, "type": LayoutType.Expression},
            reactions=[],
        )
        assert r.df.shape == (0, 0)
        assert r.plain == ""

    def test_markdown_latex_html_non_empty(self, expr_single: ExpressionResult) -> None:
        assert expr_single.markdown.startswith("|")
        assert "\\begin{tabular}" in expr_single.latex
        assert "<table" in expr_single.html


class TestChartResult:
    def test_df_parses_pipe_rows(self) -> None:
        data = make_chart_data([["x", "y"], ["1", "2"], ["3", "4"]])
        r = ChartResult(**{**ITEM_KWARGS, "type": LayoutType.Chart}, data=data)
        df = r.df
        assert "x" in df.columns and "y" in df.columns
        assert df.shape[0] == 2

    def test_markdown_and_latex_non_empty_with_valid_data(self) -> None:
        data = make_chart_data([["x", "y"], ["1", "2"]])
        r = ChartResult(**{**ITEM_KWARGS, "type": LayoutType.Chart}, data=data)
        assert "|" in r.markdown
        assert "\\begin{tabular}" in r.latex
        assert "<table" in r.html

    def test_empty_data_falls_back_to_empty_df(self) -> None:
        r = ChartResult(**{**ITEM_KWARGS, "type": LayoutType.Chart}, data="")
        df = r.df
        assert df.shape[1] == 0 or df.shape[0] == 0


class TestMoleculeResult:
    def test_plain_prefers_smi(self) -> None:
        r = MoleculeResult(
            **{**ITEM_KWARGS, "type": LayoutType.Molecule},
            caption="benzene",
            smi="c1ccccc1",
        )
        assert r.plain == "c1ccccc1"

    def test_plain_falls_back_to_caption_for_markush(self) -> None:
        r = MoleculeResult(
            **{**ITEM_KWARGS, "type": LayoutType.Molecule},
            caption="*NC(=O)*",
            smi="*NC(=O)*",
            markush=True,
        )
        assert r.plain == "*NC(=O)*"

    def test_accepts_esmi(self) -> None:
        r = MoleculeResult(
            **{**ITEM_KWARGS, "type": LayoutType.Molecule},
            smi="CCO",
            esmi="CCO",
        )
        assert r.esmi == "CCO"

    def test_markdown_wraps_in_bold_italic(self) -> None:
        r = MoleculeResult(
            **{**ITEM_KWARGS, "type": LayoutType.Molecule},
            smi="CCO",
        )
        assert r.markdown == "***CCO***"


class TestEquationResult:
    def test_latex_identity(self) -> None:
        r = EquationResult(**{**ITEM_KWARGS, "type": LayoutType.Equation}, latex_repr="a+b")
        assert r.latex == "a+b"
        assert r.markdown == "$$\na+b\n$$"

    def test_html_returns_mathml_like(self) -> None:
        r = EquationResult(**{**ITEM_KWARGS, "type": LayoutType.Equation}, latex_repr="a+b")
        assert "<math" in r.html

    def test_html_fallback_on_unconvertible(self) -> None:
        r = EquationResult(**{**ITEM_KWARGS, "type": LayoutType.Equation}, latex_repr="\\unknown_cmd{xx}")
        assert "<math" in r.html


class TestFigureResult:
    def test_plain_is_desc(self) -> None:
        r = FigureResult(**{**ITEM_KWARGS, "type": LayoutType.Figure}, desc="cat on a mat")
        assert r.plain == "cat on a mat"
        assert r.markdown == "cat on a mat"
        assert r.html == "cat on a mat"


class TestGroupedResult:
    def test_groups_items_with_blank_line_join(self) -> None:
        def _text(txt: str) -> TextualResult:
            return TextualResult(
                **{**ITEM_KWARGS, "type": LayoutType.Paragraph},
                bboxes=[],
                contents=[],
                text=txt,
            )

        g = GroupedResult(
            **{**ITEM_KWARGS, "type": LayoutType.Group},
            items=[_text("first"), _text("second")],
        )
        assert g.plain == "first\n\nsecond"
        assert g.markdown == "first\n\nsecond"
