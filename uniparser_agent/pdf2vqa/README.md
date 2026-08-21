# 习题 VQA 抽取（`uniparser-agent vqa`）

从习题、试卷、题册或答案册中提取结构化问答数据，包括题号、题干、短答案和详细解析，并生成适合题库整理、人工检查或模型训练的 JSONL、Markdown 和 ShareGPT 数据。

支持本地 PDF、图片和公开 PDF URL；题册与答案册分开时，也可以同时输入两个本地 PDF 自动配对。

## 核心能力

- 从习题或试卷中提取题号、题干、答案和解析
- 支持题目与答案位于同一文档
- 支持“题册 + 答案册”两个独立 PDF，并自动配对相同题号
- 保留题干和解析中的公式
- 导出题目相关图片，生成图文结合的 VQA 样本
- 输出 JSONL 和 Markdown，方便数据处理与人工检查
- 输出 ShareGPT 格式，可用于多模态或纯文本模型训练
- 支持复用已有 `pages_tree.json`，方便更换模型后重新抽取

## 安装

运行要求：

- Python 3.11+
- 已安装 `uniparser-agent`
- 输入原始文档时，需要 UniParser API Key
- OpenAI 兼容的 LLM 服务

安装项目依赖：

```bash
cd uniparser_agent
uv sync
```

或在已有虚拟环境中安装：

```bash
cd uniparser_agent
uv pip install -e ".[dev]"
```



## 配置

设置 UniParser 和 LLM 环境变量：

```bash
export UNIPARSER_API_KEY="your-uniparser-key"
export OPENAI_API_KEY="your-llm-key"
export OPENAI_BASE_URL="https://example.com/v1"
export OPENAI_MODEL="your-model"
```


| 变量                   | 是否必填               | 用途                 |
| -------------------- | ------------------ | ------------------ |
| `UNIPARSER_API_KEY`  | 输入 PDF、图片或 URL 时必填 | 解析原始文档             |
| `OPENAI_API_KEY`     | 必填                 | 调用问答抽取模型           |
| `OPENAI_BASE_URL`    | 必填                 | OpenAI 兼容服务地址      |
| `OPENAI_MODEL`       | 必填                 | 问答抽取模型名称           |
| `UNIPARSER_BASE_URL` | 可选                 | 自定义 UniParser 服务地址 |


使用已有 `pages_tree.json` 时不需要 `UNIPARSER_API_KEY`。

也可以通过 `--api-key`、`--base-url` 和 `--model` 在命令行中覆盖 LLM 配置。

不要把真实 API Key 写入源码或提交到仓库。

## 快速开始



### 从一个 PDF 中抽取

适用于题目、答案和解析位于同一文档：

```bash
uv run uniparser-agent vqa /path/to/exam.pdf \
  -o ./vqa_out
```

完成后主要查看：

```text
vqa_out/merged_vqa_pairs.jsonl
vqa_out/merged_vqa_pairs.md
```



## 使用指南



### 题册与答案册分别存放

题册和答案册都必须是本地 PDF：

```bash
uv run uniparser-agent vqa /path/to/questions.pdf \
  --answer-pdf /path/to/answers.pdf \
  -o ./vqa_out
```

程序会按“题册在前、答案册在后”的顺序处理，并根据章节和题号配对题干、答案与解析。

`--answer-pdf` 不能与图片、URL 或 `--pages-tree` 同时使用。

### 输入图片

```bash
uv run uniparser-agent vqa /path/to/page.png \
  -o ./vqa_out
```

适合单页试题、截图或扫描图片。

### 输入公开 PDF URL

```bash
uv run uniparser-agent vqa "https://example.com/exam.pdf" \
  -o ./vqa_out
```

URL 必须能够公开访问，并直接返回 PDF。

### 使用已有解析结果

如果文档已完成 UniParser 解析：

```bash
uv run uniparser-agent vqa \
  --pages-tree /path/to/pages_tree.json \
  -o ./vqa_out
```

该方式适合更换模型或重新抽取，不会再次消耗 UniParser 解析额度。

## 常用参数

```text
uniparser-agent vqa [OPTIONS] [INPUT_PATH]
```


| 参数                    | 用途                     |
| --------------------- | ---------------------- |
| `INPUT_PATH`          | 本地 PDF、图片或公开 PDF URL   |
| `-o` / `--output-dir` | 首选输出目录；默认 `./vqa_out`；已存在时自动使用同级后缀目录 |
| `--answer-pdf`        | 输入独立答案册 PDF            |
| `--pages-tree`        | 复用已有 `pages_tree.json` |
| `--json`              | 在终端输出机器可读的运行摘要         |


输入规则：

- `INPUT_PATH` 与 `--pages-tree` 二选一
- `--answer-pdf` 必须与题册 PDF 一起使用
- `--answer-pdf` 与 `--pages-tree` 不能同时使用

查看全部参数：

```bash
uv run uniparser-agent vqa --help
```



## 输出结果

默认输出目录如下：

```text
vqa_out/
├── merged_vqa_pairs.jsonl
├── merged_vqa_pairs.md
├── vqa_sharegpt.json
├── vqa_images/
├── run_meta.json
├── parse/
│   └── pages_tree.json
├── merge/
│   └── merged.pdf
├── extracted_vqa.jsonl
├── llm_content_list.json
└── llm_raw_response.txt
```

`merge/merged.pdf` 只在使用“题册 + 答案册”模式时生成。`vqa_images/` 中是否有图片取决于原始解析结果。

### 主要结果


| 文件                       | 用途                     |
| ------------------------ | ---------------------- |
| `merged_vqa_pairs.jsonl` | **结构化主结果**，每行一条完整问答对   |
| `merged_vqa_pairs.md`    | 便于人工阅读和检查的 Markdown 版本 |
| `vqa_sharegpt.json`      | ShareGPT 格式的模型训练数据     |
| `vqa_images/`            | 与题目或解析相关的图片            |
| `run_meta.json`          | 本次运行的模型、题量、图片数、耗时和文件路径 |




### 辅助结果


| 文件                      | 用途                     |
| ----------------------- | ---------------------- |
| `parse/pages_tree.json` | UniParser 解析结果，可用于重新抽取 |
| `merge/merged.pdf`      | 合并后的题册与答案册，仅双 PDF 模式生成 |
| `extracted_vqa.jsonl`   | 合并前的题目、答案和解析片段         |
| `llm_content_list.json` | 送入问答抽取阶段的文档内容          |
| `llm_raw_response.txt`  | 模型原始返回，主要用于问题排查        |




### JSONL 数据格式

`merged_vqa_pairs.jsonl` 每行是一道题：

```json
{
  "question_chapter_title": "第一章",
  "answer_chapter_title": "第一章答案",
  "label": 1,
  "question": "题干内容",
  "answer": "A",
  "solution": "详细解析"
}
```

字段说明：


| 字段                       | 含义                |
| ------------------------ | ----------------- |
| `question_chapter_title` | 题目所在章节或栏目         |
| `answer_chapter_title`   | 答案所在章节或栏目         |
| `label`                  | 题号                |
| `question`               | 题干，可包含公式和图片引用     |
| `answer`                 | 短答案，如选项字母、数值或填空结果 |
| `solution`               | 解题过程或详细解析         |


题目没有章节标题时，对应字段可能为空。

### ShareGPT 数据格式

`vqa_sharegpt.json` 是 JSON 数组，每条数据包含：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>题干内容"
    },
    {
      "role": "assistant",
      "content": "答案\n\n详细解析"
    }
  ],
  "images": [
    "/absolute/path/to/vqa_images/question_1.png"
  ]
}
```

- 有图片时，user 内容包含对应数量的 `<image>` 标记
- `images` 保存相关图片路径
- 无图片时，`images` 为空数组，可作为纯文本问答使用
- 没有题干或没有答案/解析的数据不会进入 ShareGPT 主结果



## 结果检查

1. 打开 `merged_vqa_pairs.md`，快速检查题目、答案和解析是否配对。
2. 使用 `merged_vqa_pairs.jsonl` 进行数据处理或导入题库。
3. 训练模型前检查 `vqa_sharegpt.json` 中的图片路径是否在目标环境可访问。
4. 如果题目数量明显偏少，查看 `extracted_vqa.jsonl` 判断是抽取不足还是配对失败。
5. 如果公式、图片或章节识别不正确，结合 `parse/pages_tree.json` 检查原始解析质量。



## 常见问题



### 为什么最终题目数量比原文少？

主结果只保留能够识别题号并完成题目与答案配对的数据。题号不清晰、章节不一致、答案缺失或模型未正确识别都可能导致数量减少。

### 为什么题目图片没有导出？

只有原始解析结果中实际包含图片数据的内容才能写入 `vqa_images/`。PDF 中看得到图片不代表解析结果一定包含可导出的图片。

### 题册和答案册可以使用 URL 吗？

不可以。双文档模式要求题册和答案册都是本地 PDF。单文档模式支持公开 PDF URL。

### 输出目录已存在时会怎样？

程序不会报错、复用或删除旧目录，而是自动创建第一个可用的同级目录。例如 `vqa_out`
已存在时使用 `vqa_out_1`，两者都存在时使用 `vqa_out_2`。请以运行结果中的
`output_dir` 为准；失败任务的部分产物也保留在本次新目录中，便于排查。

> 出于安全原因，根目录、HOME、当前工作目录及 Git 元数据目录不能作为首选输出目录。



### 如何更换模型后重新抽取？

保留首次运行生成的 `parse/pages_tree.json`，然后使用 `--pages-tree` 重新运行。

### ShareGPT 中为什么使用绝对图片路径？

结果会记录当前运行环境中的图片绝对路径。将数据迁移到其他机器或训练环境后，需要同步图片并按需要更新路径。

## 当前限制

- 双 PDF 模式只支持本地 PDF
- 题号需要能够识别为正整数
- 题目和答案主要依靠章节与题号配对
- 跨页题目、复杂多栏版式或严重粘连内容可能配对不准确
- 图片导出取决于 UniParser 是否提供可用图片数据
- 公式和题目结构的准确性依赖原始文档解析质量
- 抽取结果适合批量整理，但正式训练或入库前仍建议抽样检查
