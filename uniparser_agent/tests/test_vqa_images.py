"""Tests for VQA image export, adapter image items, and ShareGPT formatting."""

from __future__ import annotations

import json
from pathlib import Path

from uniparser_agent.pdf2vqa.image_export import export_images_from_pages_tree
from uniparser_agent.pdf2vqa.layout_adapter import adapt_pages_tree_file, pages_tree_to_content_list
from uniparser_agent.pdf2vqa.output_parser import parse_llm_response
from uniparser_agent.pdf2vqa.vqa_formatter import convert_vqa_pair_to_sharegpt, write_sharegpt


# Minimal valid 1x1 PNG (preferred — magic bytes detect format)
_PNG_1X1_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def _tiny_image_b64() -> str:
    return _PNG_1X1_B64


def test_export_and_adapt_figuregroup(tmp_path: Path):
    b64 = _tiny_image_b64()
    tree = {
        "pages_tree": [
            [
                {
                    "type": "figuregroup",
                    "page": 0,
                    "block": 10,
                    "order": 0,
                    "source": "",
                    "items": [
                        {
                            "type": "figure",
                            "page": 0,
                            "block": 11,
                            "order": 0,
                            "source": b64,
                            "desc": "",
                        },
                        {
                            "type": "figurecaption",
                            "page": 0,
                            "block": 12,
                            "order": 1,
                            "text": "Fig. 1 Energy diagram",
                        },
                    ],
                },
                {"type": "paragraph", "page": 0, "block": 13, "order": 1, "text": "Q1 text"},
            ]
        ]
    }
    images_dir = tmp_path / "vqa_images"
    path_map = export_images_from_pages_tree(tree, images_dir)
    assert path_map
    assert list(images_dir.iterdir())

    content = pages_tree_to_content_list(tree, image_path_map=path_map)
    image_items = [c for c in content if c.get("type") == "image"]
    assert len(image_items) == 1
    assert image_items[0]["img_path"].startswith("vqa_images/")
    assert "Fig. 1" in " ".join(image_items[0].get("image_caption") or [])
    assert any(c.get("text") == "Q1 text" for c in content)


def test_export_deeply_nested_figure(tmp_path: Path):
    """UniParser often nests figure under group → image → figuregroup."""
    b64 = _tiny_image_b64()
    tree = {
        "pages_tree": [
            [
                {
                    "type": "group",
                    "page": 0,
                    "block": 1,
                    "order": 0,
                    "source": "",
                    "items": [
                        {
                            "type": "image",
                            "page": 0,
                            "block": 2,
                            "order": 0,
                            "source": "",
                            "items": [
                                {
                                    "type": "figuregroup",
                                    "page": 0,
                                    "block": 3,
                                    "order": 0,
                                    "source": "",
                                    "items": [
                                        {
                                            "type": "figure",
                                            "page": 0,
                                            "block": 4,
                                            "order": 0,
                                            "source": b64,
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        ]
    }
    images_dir = tmp_path / "vqa_images"
    path_map = export_images_from_pages_tree(tree, images_dir)
    assert len(path_map) == 1
    assert list(images_dir.iterdir())
    content = pages_tree_to_content_list(tree, image_path_map=path_map)
    assert sum(1 for c in content if c.get("type") == "image") == 1


def test_adapt_pages_tree_file_writes_images(tmp_path: Path):
    b64 = _tiny_image_b64()
    tree = {
        "pages_tree": [
            [
                {
                    "type": "molecule",
                    "page": 1,
                    "block": 2,
                    "order": 0,
                    "source": b64,
                    "text": "",
                }
            ]
        ]
    }
    tree_path = tmp_path / "pages_tree.json"
    tree_path.write_text(json.dumps(tree), encoding="utf-8")
    content_path = tmp_path / "llm_content_list.json"
    content = adapt_pages_tree_file(tree_path, content_path, images_dir=tmp_path / "vqa_images")
    assert (tmp_path / "vqa_images").is_dir()
    assert any(c.get("type") == "image" for c in content)


def test_parser_and_sharegpt_image_placeholders(tmp_path: Path):
    images_dir = tmp_path / "vqa_images"
    images_dir.mkdir()
    img = images_dir / "abc123.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 64)

    content = [
        {"id": 0, "type": "text", "text": "What is shown?"},
        {
            "id": 1,
            "type": "image",
            "img_path": "vqa_images/abc123.jpg",
            "image_caption": ["diagram"],
        },
        {"id": 2, "type": "text", "text": "It is A."},
    ]
    response = (
        "<chapter><title></title>"
        "<vqa_pair><label>1</label><question>0,1</question>"
        "<answer>A</answer><solution>2</solution></vqa_pair>"
        "</chapter>"
    )
    extracted = parse_llm_response(response, content)
    assert len(extracted) == 1
    assert "![diagram](vqa_images/abc123.jpg)" in extracted[0]["question"]

    merged = [
        {
            "label": 1,
            "question": extracted[0]["question"],
            "answer": "A",
            "solution": extracted[0]["solution"],
            "question_chapter_title": "",
            "answer_chapter_title": "",
        }
    ]
    out = write_sharegpt(merged, images_dir, tmp_path / "vqa_sharegpt.json", base_dir=tmp_path)
    records = json.loads(out.read_text(encoding="utf-8"))
    assert len(records) == 1
    user = records[0]["messages"][0]["content"]
    assert user.startswith("<image>")
    assert len(records[0]["images"]) == 1
    assert user.count("<image>") == len(records[0]["images"])
    assert "What is shown?" in user
    assert "![" not in user


def test_sharegpt_no_images_ok():
    item = convert_vqa_pair_to_sharegpt(
        {"question": "2+2?", "answer": "4", "solution": ""},
        image_index={},
        base_dir=Path("."),
    )
    assert item is not None
    assert item["images"] == []
    assert "<image>" not in item["messages"][0]["content"]
