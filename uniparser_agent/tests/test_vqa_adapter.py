"""Unit tests for UniParser → LLM content-list adapter."""

from __future__ import annotations

from pathlib import Path

from uniparser_agent.pdf2vqa.layout_adapter import pages_tree_to_content_list


def test_adapter_skips_noise_and_numbers_ids():
    data = {
        "pages_tree": [
            [
                {"type": "hline", "order": 0, "text": ""},
                {"type": "pageheader", "order": 1, "text": "header"},
                {"type": "paragraph", "order": 2, "text": "Q1 text"},
                {
                    "type": "equation",
                    "order": 3,
                    "latex_repr": "x^2=1",
                    "text": "",
                },
                {"type": "pagenumber", "order": 4, "text": "1"},
            ]
        ]
    }
    content = pages_tree_to_content_list(data)
    assert [c["id"] for c in content] == [0, 1]
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Q1 text"
    assert content[1]["type"] == "equation"
    assert "x^2=1" in content[1]["text"]


def test_adapter_supports_v13_inline_spans_and_table_structure():
    data = {
        "pages_tree": [
            [
                {
                    "type": "paragraph",
                    "order": 0,
                    "contents": ["Energy ", "E=mc^2", " in ", "CCO"],
                    "types": ["text", "equationinline", "text", "molecule"],
                },
                {
                    "type": "table",
                    "order": 1,
                    "structure": "<table><tr><td>[[VL_TABLE_0_0]]</td></tr></table>",
                    "placeholders": ["[[VL_TABLE_0_0]]"],
                    "contents": ["x^2"],
                    "types": ["equationinline"],
                },
                {
                    "type": "molecule",
                    "order": 2,
                    "esmi": "C1=CC=CC=C1",
                },
            ]
        ]
    }

    content = pages_tree_to_content_list(data)

    assert content[0] == {
        "id": 0,
        "type": "text",
        "text": "Energy $E=mc^2$ in `CCO`",
    }
    assert content[1]["type"] == "table"
    assert "<td>$x^2$</td>" in content[1]["table_body"]
    assert content[2] == {
        "id": 2,
        "type": "text",
        "text": "`C1=CC=CC=C1`",
    }


def test_adapter_uses_v13_caption_contents(tmp_path: Path):
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"image")
    data = {
        "pages_tree": [
            [
                {
                    "type": "figuregroup",
                    "order": 0,
                    "items": [
                        {
                            "type": "figure",
                            "page": 1,
                            "block": 1,
                            "source": "image-data",
                        },
                        {
                            "type": "figurecaption",
                            "contents": ["Figure ", "x=1"],
                            "types": ["text", "equationinline"],
                        },
                    ],
                }
            ]
        ]
    }

    content = pages_tree_to_content_list(
        data,
        image_path_map={(1, 1): image_path},
    )

    assert content == [
        {
            "id": 0,
            "type": "image",
            "img_path": "vqa_images/figure.png",
            "image_caption": ["Figure $x=1$"],
        }
    ]
