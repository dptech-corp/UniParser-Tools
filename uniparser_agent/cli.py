from __future__ import annotations

import json
from typing import Optional

import typer

from uniparser_agent.llm import LLMConfig, resolve_llm_config
from uniparser_agent.parse.service import parse_document
from uniparser_agent.pdf2vqa.pipeline import run_vqa_pipeline


app = typer.Typer(
    name="uniparser-agent",
    help="UniParser document parsing and exam VQA extraction.",
    no_args_is_help=True,
)


@app.command("parse")
def parse_cmd(
    input_path: str = typer.Argument(..., help="Local PDF/image path or public PDF URL."),
    output_dir: Optional[str] = typer.Option(
        None,
        "-o",
        "--output-dir",
        help="Preferred output directory; a suffixed sibling is used if occupied.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Parse a document with UniParser scientific-paper defaults."""
    result = parse_document(input_path, output_dir=output_dir)
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    typer.echo(f"Token: {result.get('token', '')}")
    typer.echo(f"Pages tree: {result['pages_tree_path']}")
    typer.echo(f"Markdown: {result['markdown_path']}")
    typer.echo(f"Output directory: {result['output_dir']}")


@app.command("vqa")
def vqa_cmd(
    input_path: Optional[str] = typer.Argument(
        None,
        help="Local PDF/image path or public PDF URL. Omit when using --pages-tree.",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "-o",
        "--output-dir",
        help="Preferred VQA output directory; a suffixed sibling is used if occupied.",
    ),
    answer_pdf: Optional[str] = typer.Option(
        None,
        "--answer-pdf",
        help="Answer booklet PDF. Merged after the question booklet (local PDFs only).",
    ),
    pages_tree: Optional[str] = typer.Option(
        None,
        "--pages-tree",
        help="Skip UniParser parse and use an existing pages_tree.json.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="LLM API key (overrides OPENAI_API_KEY).",
        envvar=[],
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help="LLM base URL (overrides OPENAI_BASE_URL).",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="LLM model name (overrides OPENAI_MODEL).",
    ),
    enable_thinking: bool = typer.Option(
        False,
        "--enable-thinking/--no-enable-thinking",
        help="Pass chat_template_kwargs.enable_thinking for Qwen-compatible servers.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Parse with UniParser (unless --pages-tree), then extract VQA pairs via LLM."""
    if answer_pdf and pages_tree:
        raise typer.BadParameter("Use either --answer-pdf or --pages-tree, not both.")
    if answer_pdf and not input_path:
        raise typer.BadParameter("--answer-pdf requires the question booklet as INPUT.")
    if not input_path and not pages_tree:
        raise typer.BadParameter("Provide INPUT (pdf/url/image) or --pages-tree.")
    if input_path and pages_tree:
        raise typer.BadParameter("Use either INPUT or --pages-tree, not both.")

    llm_config = _build_llm_config(
        api_key=api_key,
        base_url=base_url,
        model=model,
        enable_thinking=enable_thinking,
    )
    result = run_vqa_pipeline(
        input_path=input_path,
        answer_pdf=answer_pdf,
        pages_tree_path=pages_tree,
        output_dir=output_dir,
        llm_config=llm_config,
    )
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return

    paths = result["paths"]
    if paths.get("merged_pdf"):
        typer.echo(f"Merged PDF: {paths['merged_pdf']}")
    typer.echo(f"Pages tree: {paths['pages_tree']}")
    typer.echo(f"Content list items: {result['n_content_items']}")
    typer.echo(f"VQA images: {result.get('n_vqa_images', 0)} -> {paths.get('vqa_images', '')}")
    typer.echo(f"Merged VQA pairs: {result['n_merged_vqa']}")
    typer.echo(f"JSONL: {paths['merged_vqa_pairs_jsonl']}")
    typer.echo(f"Markdown: {paths['merged_vqa_pairs_md']}")
    if paths.get("vqa_sharegpt"):
        typer.echo(f"ShareGPT: {paths['vqa_sharegpt']}")
    typer.echo(f"Output directory: {paths['output_dir']}")


def _build_llm_config(
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    enable_thinking: bool,
) -> LLMConfig:
    try:
        return resolve_llm_config(
            api_key=api_key,
            base_url=base_url,
            model=model,
            enable_thinking=enable_thinking,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()
