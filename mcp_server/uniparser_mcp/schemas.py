"""Pydantic models for the uniparser_parse tool."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TextualChoice(str, Enum):
    disable = "disable"
    ocr_fast = "ocr-fast"
    ocr_hq = "ocr-hq"
    digital = "digital"
    base64 = "base64"


class ParseModeChoice(str, Enum):
    disable = "disable"
    ocr_fast = "ocr-fast"
    ocr_hq = "ocr-hq"
    base64 = "base64"


class ParseRequest(BaseModel):
    file_path: str | None = Field(default=None, description="Absolute path to a local PDF file.")
    image_path: str | None = Field(
        default=None,
        description="Absolute path to a local image snippet (.png, .jpg, etc.).",
    )
    pdf_url: str | None = Field(default=None, description="Publicly accessible PDF URL.")
    output_dir: str | None = Field(
        default=None,
        description=(
            "Preferred output directory. If occupied, an available suffixed sibling is created. "
            "Default: ~/Uni-Parser-Skill/<source_stem>/."
        ),
    )
    async_mode: bool = Field(
        default=False,
        description="Submit with sync=false, then poll until the job completes.",
    )
    textual: TextualChoice = Field(default=TextualChoice.ocr_hq)
    equation: ParseModeChoice = Field(default=ParseModeChoice.ocr_hq)
    table: ParseModeChoice = Field(default=ParseModeChoice.ocr_hq)
    chart: ParseModeChoice = Field(default=ParseModeChoice.base64)
    figure: ParseModeChoice = Field(default=ParseModeChoice.base64)
    expression: ParseModeChoice = Field(default=ParseModeChoice.base64)
    molecule: ParseModeChoice = Field(default=ParseModeChoice.ocr_fast)

    @model_validator(mode="after")
    def exactly_one_input(self) -> ParseRequest:
        provided = [
            name
            for name, value in (
                ("file_path", self.file_path),
                ("image_path", self.image_path),
                ("pdf_url", self.pdf_url),
            )
            if value
        ]
        if len(provided) != 1:
            raise ValueError("Provide exactly one of file_path, image_path, or pdf_url.")
        return self


class ErrorDetail(BaseModel):
    code: str
    message: str
    stage: str | None = None
    output_dir: str | None = None
    token: str | None = None


class ParseSuccess(BaseModel):
    ok: Literal[True] = True
    status: Literal["success"] = "success"
    output_dir: str
    markdown_path: str
    pages_tree_path: str
    formatted_meta_path: str
    trigger_meta_path: str | None = None
    token: str
    input_type: Literal["file", "image", "url"]
    content_chars: int
    content_preview: str
    message: str


class ErrorResult(BaseModel):
    ok: Literal[False] = False
    error: ErrorDetail


ParseResult = ParseSuccess | ErrorResult
