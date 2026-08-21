"""Register MCP tools."""

from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from uniparser_mcp.client import get_client
from uniparser_mcp.errors import config_error, input_error
from uniparser_mcp.pipeline.parse import run_parse
from uniparser_mcp.schemas import (
    ParseModeChoice,
    ParseRequest,
    ParseResult,
    TextualChoice,
)


_PARSE_DESCRIPTION = (
    "Parse a local PDF, local image snippet, or public PDF URL with UniParser. "
    "Uploads content to the configured UniParser service. "
    "Returns saved file paths, a short markdown preview, and metadata. "
    "Use absolute paths for local files."
)


def register_tools(mcp: FastMCP) -> None:
    @mcp.tool(
        title="Parse document",
        description=_PARSE_DESCRIPTION,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "openWorldHint": True,
        },
    )
    async def uniparser_parse(
        file_path: Annotated[str | None, Field(description="Absolute path to a local PDF file.")] = None,
        image_path: Annotated[
            str | None,
            Field(description="Absolute path to a local image (.png, .jpg, etc.)."),
        ] = None,
        pdf_url: Annotated[str | None, Field(description="Publicly accessible PDF URL.")] = None,
        output_dir: Annotated[
            str | None,
            Field(
                description=(
                    "Preferred directory for saved results. If occupied, a suffixed sibling is used. "
                    "Default: ~/Uni-Parser-Skill/<stem>/."
                )
            ),
        ] = None,
        async_mode: Annotated[
            bool,
            Field(description="Submit with sync=false and poll until completion."),
        ] = False,
        textual: Annotated[TextualChoice, Field(description="Textual parse mode.")] = TextualChoice.ocr_hq,
        equation: Annotated[ParseModeChoice, Field(description="Equation parse mode.")] = ParseModeChoice.ocr_hq,
        table: Annotated[ParseModeChoice, Field(description="Table parse mode.")] = ParseModeChoice.ocr_hq,
        chart: Annotated[ParseModeChoice, Field(description="Chart parse mode.")] = ParseModeChoice.base64,
        figure: Annotated[ParseModeChoice, Field(description="Figure parse mode.")] = ParseModeChoice.base64,
        expression: Annotated[
            ParseModeChoice,
            Field(description="Expression parse mode."),
        ] = ParseModeChoice.base64,
        molecule: Annotated[ParseModeChoice, Field(description="Molecule parse mode.")] = ParseModeChoice.ocr_fast,
        ctx: Context = None,
    ) -> ParseResult:
        try:
            client = get_client()
        except ValueError as exc:
            return config_error(str(exc))

        try:
            try:
                req = ParseRequest(
                    file_path=file_path,
                    image_path=image_path,
                    pdf_url=pdf_url,
                    output_dir=output_dir,
                    async_mode=async_mode,
                    textual=textual,
                    equation=equation,
                    table=table,
                    chart=chart,
                    figure=figure,
                    expression=expression,
                    molecule=molecule,
                )
            except ValueError as exc:
                return input_error(str(exc))

            return await run_parse(client, req, ctx)
        finally:
            client.close()
