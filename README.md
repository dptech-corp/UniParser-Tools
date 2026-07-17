# UniParser-Tools

UniParser Tools 是一个强大的文档解析工具包，支持对 PDF 文件和图片进行智能解析，提取文本、表格、图片、公式、分子结构等多种语义元素。

本工具包提供了完整的 Python API，方便开发者快速集成文档解析能力到自己的项目中。

## 主要功能

### 核心解析能力

- **文本提取 textual**：支持数字导出和 OCR 快速识别
- **表格识别 table**：自动识别表格结构并提取内容
- **公式识别 equation**：识别数学公式和化学表达式
- **分子识别 molecule**：提取化学分子式及其索引
- **反应式识别 expression**：提取化学反应式
- **图表识别 chart**：识别简单图表元素
- **图片识别 figure**：提取图片

### 输出格式

支持多种输出格式，满足不同场景需求：

- **Content (End2End)**：全文文本内容，适用于 LLM 等场景
- **Objects**：JSON 格式的语义块，适用于语义分析
- **Pages dict**：原始解析格式，按页面组织的详细语义块
- **Pages tree**：带嵌套关系的树结构，支持复杂语义分析

### 格式化输出

支持多种格式化输出方式：

- **Markdown**：Markdown 格式输出 ⭐️
- **HTML**：HTML 格式输出 ⭐️
- **LaTeX**：LaTeX 格式输出
- **Plain**：纯文本格式输出
- **Markup**：标记文本格式输出

### 高级功能

- **图文对提取**：自动提取图片及其对应的标题、图注
- **表格结构化提取**：提取表格及其表题、表注
- **分子索引关联**：提取分子结构及其索引信息
- **公式索引关联**：提取公式及其编号信息

## 安装

```bash
pip install -r requirements.txt
```

或者安装为可编辑包：

```bash
pip install -e .
```

## API-Key 配置

所有请求都通过 `X-API-Key` 请求头认证，`UniParserClient` 会自动注入。

- **获取方式**：在 UniParser 服务首页（如 `https://uniparser.dp.tech/`）注册访客账号，或向运维/业务方申请长期 API-Key。
- **推荐存储**：使用环境变量 `UNIPARSER_API_KEY`，避免在代码中硬编码。
- **错误处理**：Key 缺失/过期、限流等情况都会被客户端统一包装成 `{"status": "error", ...}` 返回，详见下方 [错误处理](#错误处理)。

```python
import os
parser = UniParserClient(
    host="https://uniparser.dp.tech/",
    api_key=os.getenv("UNIPARSER_API_KEY"),
)
```

## 解析配置：7 个语义类 + 2 个枚举

提交解析任务时（`trigger_file` / `trigger_snip` / `trigger_url`），可分别设置 7 类语义元素的处理模式：

| 字段 | 含义 | 枚举类型 |
|------|------|---------|
| `textual` | 普通文本（段落、标题等） | `ParseModeTextual` |
| `equation` | 数学公式 | `ParseMode` |
| `table` | 表格 | `ParseMode` |
| `chart` | 图表 | `ParseMode` |
| `figure` | 图片 / 插图 | `ParseMode` |
| `expression` | 化学反应式 | `ParseMode` |
| `molecule` | 化学分子结构 | `ParseMode` |

### `ParseMode`（除 textual 外都用这个）

| 取值 | 名称 | 含义 |
|------|------|------|
| `-3` / `-2` | `DumpHosting` / `DumpLocal` | 保留接口，默认关闭 |
| `-1` | `DumpBase64` | 禁用解析，输出原始图像 Base64 |
| `0`  | `Disable`   | 禁用解析，不输出 |
| `1`  | `OCRFast`   | 快速 OCR（默认） |
| `2`  | `OCRHighQuality` | 高质 OCR |

### `ParseModeTextual`（仅用于 `textual`）

| 取值 | 名称 | 含义 |
|------|------|------|
| `-1` | `DumpBase64` | 输出原始图像 Base64 |
| `0`  | `Disable`   | 不解析、不输出 |
| `1`  | `OCRFast`   | 快速 OCR |
| `2`  | `OCRHighQuality` | 高质 OCR，支持行内公式 |
| `3`  | `DigitalExported` | 从数字原生 PDF 直接抽取文字 |

## 快速开始

> ‼️‼️‼️ 以下仅为代码功能示例，具体运行代码请参考 `playground/*.ipynb` ‼️‼️‼️

### 1. 初始化客户端

```python
import os
from uniparser_tools.api.clients import UniParserClient

# 设置 API 密钥
api_key = os.getenv('UNIPARSER_API_KEY')

# 初始化客户端
parser = UniParserClient(
    host="https://uniparser.dp.tech/",
    api_key=api_key
)
```

### 2. 解析 PDF 文件（科学文献推荐默认）

```python
from uniparser_tools.common.constant import ParseMode, ParseModeTextual

# 科学文献解析模式（推荐默认值）
result = parser.trigger_file(
    file_path="./example.pdf",
    textual=ParseModeTextual.OCRHighQuality,  # high quality
    equation=ParseMode.OCRHighQuality,        # high quality
    table=ParseMode.OCRHighQuality,           # high quality
    chart=ParseMode.DumpBase64,               # original image base64
    figure=ParseMode.DumpBase64,              # original image base64
    expression=ParseMode.DumpBase64,          # original image base64
    molecule=ParseMode.OCRFast,               # fast
)

if result["status"] == "success":
    token = result["token"]
    print(f"解析成功，token: {token}")
```

### 3. 获取解析结果

#### 输出配置（`get_result` / `get_formatted` 通用开关）

| 开关 | 默认 | 说明 |
|------|------|------|
| `content` | `True` | 返回全文纯/富文本，适合 LLM |
| `objects` | `False` | JSON 语义块列表，适合语义分析 |
| `pages_dict` | `False` | 按页组织的原始解析布局 |
| `pages_tree` | `False` | 带父子关系的嵌套树，适合复杂分析 |
| `return_half` | `False` | 解析进行中即取已完成部分 |
| `molecule_source` | `False` | 返回分子原始源（SMILES/mol 等） |

同一 token 可复用，多次获取不同组合不会重复计费。

#### 输出格式（`FormatFlag`，仅作用于 `content` / `objects` 中的文本字段）

| 取值 | 适用场景 |
|------|----------|
| `FormatFlag.Plain` | 纯文本，适合检索 |
| `FormatFlag.Markup` | 默认标记文本 |
| `FormatFlag.Markdown` | ⭐ 推荐给 LLM |
| `FormatFlag.Latex` | LaTeX，适合公式 |
| `FormatFlag.Html` | HTML，适合表格 |

```python
from uniparser_tools.common.constant import FormatFlag

# 获取 Markdown 格式的全文内容
result = parser.get_formatted(
    token,
    content=True,
    textual=FormatFlag.Markdown,
    table=FormatFlag.Markdown,
    equation=FormatFlag.Markdown,
)

if result["status"] == "success":
    print(result["content"])
```

### 4. 使用异步回调 (Callbacks)

UniParser 支持在异步任务完成后通过 HTTP POST 回调结果到指定地址。这对于长耗时任务非常有用，无需轮询结果。

```python
# 提交带回调地址的异步解析任务
result = parser.trigger_file(
    file_path="./example.pdf",
    sync=False,  # 必须为 False 才能触发异步回调
    callback_url="https://your-server.com/api/callback",
    callback_secret="your-shared-secret",  # 用于校验回调内容的签名
    textual=ParseModeTextual.OCRHighQuality,
    equation=ParseMode.OCRHighQuality,
    table=ParseMode.OCRHighQuality,
    chart=ParseMode.DumpBase64,
    figure=ParseMode.DumpBase64,
    expression=ParseMode.DumpBase64,
    molecule=ParseMode.OCRFast,
)

if result["status"] == "success":
    token = result["token"]
    print(f"异步任务已提交，完成后将回调到指定地址。Token: {token}")
```

回调结果以 JSON 请求体发送，签名位于 `X-UniParser-Signature: sha256=<digest>` 请求头中。接收端必须使用 `callback_secret` 对原始请求体字节计算 HMAC-SHA256，再使用常量时间比较校验签名。

### 5. 解析图片文件

```python
from uniparser_tools.common.constant import ParseMode, ParseModeTextual

# 提交图片解析任务
result = parser.trigger_snip(
    snip_path="./example.png",
    textual=ParseModeTextual.OCRFast,
    table=ParseMode.OCRFast,
    molecule=ParseMode.OCRFast,
)

if result["status"] == "success":
    token = result["token"]
    # 使用 token 获取解析结果
```

### 6. 解析 URL 或对象存储文件

`trigger_url` 支持 `http(s)://`、`s3://`、`oss://`、`tos://` 和指向绝对路径的 `file://` URL；来源可以是 PDF 或常见图片格式。

```python
result = parser.trigger_url(
    pdf_url="tos://bucket/path/document.pdf",
    textual=ParseModeTextual.OCRHighQuality,
)
```

### 7. 获取第三方格式输出

```python
from uniparser_tools.common.constant import ThirdPartyFormatter

result = parser.get_third_party_output(token, formatter=ThirdPartyFormatter.MinerU)
```

## 使用示例

### 图文对提取

详细示例请参考 `playground/app.caption_extraction.ipynb`：

```python
import json
import os
from uniparser_tools.tools.caption_extraction.main import main

# 设置文件路径和保存目录
pdf_path = "./example.pdf"
save_dir = "./outputs/caption_extraction"
os.makedirs(save_dir, exist_ok=True)

# 首先需要提交解析任务并获取 token（参考前面的步骤）
# result = parser.trigger_file(pdf_path, ...)
# token = result["token"]

# 获取解析结果
result = parser.get_result(token, pages_dict=True)
json_path = f"{save_dir}/{token}.json"
json.dump(result["pages_dict"], open(json_path, "w"), indent=4)

# 提取图文对
results = main(
    token=token,
    pdf_path=pdf_path,
    json_path=json_path,
    save_dir=save_dir,
    dpi=300,
    log_level="INFO",
)

# 处理提取结果
if results:
    extracted = results["extracted"]
    for k, item in extracted.items():
        # item.main_image: 主图片
        # item.caption_image: 图题图片
        # item.group_image: 组合图片
        # item.captions: 图题文本列表
        # item.keywords: 关键词列表
        pass
```

### 多种格式输出

可以在同一次格式化输出中设置不同语义元素的输出模式：

```python
from uniparser_tools.common.constant import FormatFlag

# token 和 parser 需要从前面的步骤获取
result = parser.get_formatted(
    token,
    content=True,
    textual=FormatFlag.Markdown,    # 文本使用 Markdown
    table=FormatFlag.Html,          # 表格使用 HTML
    equation=FormatFlag.Latex,       # 公式使用 LaTeX
)

if result["status"] == "success":
    print(result["content"])
```

## 错误处理

`UniParserClient` 的所有方法**都返回 `dict`，不会抛 `requests`/HTTP 异常**。网络错误、鉴权失败、限流、业务校验失败等都被统一包装到返回值里，调用方只需判断 `status` 字段即可，不需要关心底层 HTTP 细节。

```python
result = parser.trigger_file(file_path="./paper.pdf")
if result["status"] != "success":
    # 统一错误入口
    print(result.get("description") or result.get("message"))
    raise RuntimeError(f"trigger failed: {result}")

token = result["token"]
```

返回体字段约定：

| 字段 | 出现场景 | 说明 |
|------|----------|------|
| `status` | 始终存在 | `"success"` / `"error"`（见 `StatusFlag`） |
| `token` | 触发/查询类接口 | 本次任务的 token，出错也会带上以便追溯 |
| `description` | 错误时 | 业务层错误原因，通常取自 `ErrorFlag`（如 `Token_Invalid`、`File_Size_Exceeded`、`Domain_Not_Allowed`…）或本地 traceback |
| `message` | 错误时 | 服务端返回的原始报文（非 JSON 时才填充） |

> 直接调用 REST API（curl / 自研客户端）时才需要关注 `401/403/429/…` 等原始 HTTP 状态码，详见各部署实例 `<host>/api` 上的 Authentication 章节。

## 面向 AI Agent

本仓库提供 **Agent Skill**（`skills/UniParser-Tools/`），让 Cursor、Claude Code 等助手在对话中自动完成 PDF / 图片 / 支持的文件 URL → 结构化 Markdown 与版面 JSON 的解析。用户只需安装 Skill、配置 API Key，并用自然语言或下方触发词发起任务；具体执行步骤由 Skill 内的 `SKILL.md` 指导 Agent，无需手动敲命令。

### 快速使用 Skill

**1. 为 Agent 安装 Skill**

将本仓库中的 `skills/UniParser-Tools/` 整个目录发送给 Agent，并让 Agent 安装该 Skill。

安装后重启 Agent，确保 Skill 列表中出现 **uniparser-tools**。

**2. 配置 API Key**

在 [https://uniparser.dp.tech/](https://uniparser.dp.tech/) 注册并申请 API Key，写入环境变量（Agent 终端需能读到）：

```bash
export UNIPARSER_API_KEY="your-api-key"
```

**3. 在 Agent 中使用 Skill**

在 Agent 对话中上传文件、粘贴支持的文件链接，或使用类似表述即可触发 Skill，例如：

- 中文：`解析这个 PDF`、`PDF 转 Markdown`、`提取论文`、`文档解析`、`表格提取`、`公式识别`、`化学分子`
- 英文：`parse this PDF`、`extract this paper`、`PDF to markdown`、`UniParser`、`scientific paper`

支持的输入：**本地 PDF**、**本地图片**（png / jpg 等）、**可公网访问的 PDF URL**。

**4. 使用效果与结果位置**

解析成功后，Agent 会在回复中给出 **Markdown 文件路径**（以及需要时的 **版面结构文件路径**），并可将正文摘要或全文交付给你。典型效果包括：

- 按阅读顺序输出的 **Markdown 全文**（`{源文件主名}.md`，如 `paper.pdf` → `paper.md`）
- **表格** 转为 Markdown 表格
- **公式** 转为 LaTeX
- **图片 / 图表** 等以 base64 等形式出现在结果中（视文档类型而定）
- **版面结构树** `pages_tree.json`，便于需要章节、块级布局时使用
- 面向科技文献的默认识别策略（高质量 OCR 等，由 Skill 配置）

默认将结果写入用户主目录下：

`~/Uni-Parser-Skill/<源文件主名>/`

例如解析 `paper.pdf` 时，默认目录为 `~/Uni-Parser-Skill/paper/`。该目录在**解析成功完成后**才会写入文件，通常包含：

| 文件 | 说明 |
|------|------|
| `{源文件主名}.md` | 解析得到的完整 Markdown |
| `pages_tree.json` | 结构化版面树（页面与语义块层次） |
| `formatted_meta.json` | 元数据（不含全文 `content`） |

若你在对话中指定了输出目录，Agent 也可将结果保存到你提供的路径。若目标目录已存在，Agent 会先征求你是否覆盖后再继续。大文档解析可能耗时数分钟至十余分钟；重复提交同一文件时 Agent 会按 Skill 说明从已有任务恢复，而不会重复上传。

Agent 实现细节、错误恢复与 SDK 安装说明见 `skills/UniParser-Tools/SKILL.md`。

### 参考文档

- `skills/UniParser-Tools/references/api-reference.md`
- `skills/UniParser-Tools/references/patterns.md`
- `skills/UniParser-Tools/references/data-classes.md`
- `skills/UniParser-Tools/references/layout-types.md`
- `skills/UniParser-Tools/references/utilities.md`
- `skills/UniParser-Tools/references/notes.md`

## MCP Server

UniParser 提供了基于 [Model Context Protocol](https://modelcontextprotocol.io/) 的 MCP 服务，位于 `mcp_server/` 目录，支持通过 MCP 工具调用 UniParser HTTP API。

### 可用工具

| 工具 | 说明 |
|------|------|
| `uniparser_health` | 检查服务健康状态 |
| `uniparser_version` | 获取服务版本信息 |
| `uniparser_parse_file` | 解析本机 PDF（传入绝对路径），返回 `content` 文本 |
| `uniparser_parse_url` | 解析支持的文件 URL，返回 `content` 文本 |

### 快速启动

```bash
cd mcp_server
uv sync                          # 安装依赖（与主项目隔离）
uv run python -m uniparser_mcp   # 启动 MCP 服务（stdio 模式）
```

运行时必须设置以下环境变量：

| 变量 | 说明 |
|------|------|
| `UNIPARSER_BASE_URL` | UniParser 用户服务根 URL，例如 `http://127.0.0.1:40001` |
| `UNIPARSER_API_KEY` | API 密钥，对应请求头 `X-API-Key` |

默认解析参数和输出格式见 `mcp_server/config.yaml`。

### 接入 Cursor / Claude Code

在 MCP 配置文件中添加（将路径替换为本机实际路径）：

```json
{
  "mcpServers": {
    "uniparser": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/UniParser-Tools/mcp_server",
        "python",
        "-m",
        "uniparser_mcp"
      ],
      "env": {
        "UNIPARSER_BASE_URL": "http://127.0.0.1:40001",
        "UNIPARSER_API_KEY": "your-api-key"
      }
    }
  }
}
```

传输模式默认为 `stdio`，可通过 `UNIPARSER_MCP_TRANSPORT` 环境变量切换为 `sse` 或 `streamable-http`。

详细文档见 [`mcp_server/README.md`](./mcp_server/README.md)。

## 项目结构

```
uniparser_tools/
├── api/              # API 客户端
├── common/           # 通用常量和数据类
├── tools/            # 工具模块
│   └── caption_extraction/  # 图文对提取工具
├── utils/            # 工具函数
└── order/            # 排序算法

mcp_server/           # MCP 服务（独立子项目）
├── uniparser_mcp/    # MCP server 实现
├── config.yaml       # 默认解析参数配置
└── pyproject.toml    # 独立依赖管理

playground/
├── 01.quick_start.ipynb          # 快速开始教程
├── 02.advance.ipynb              # 高级用法教程
├── 04.use_callbacks.py           # 异步回调功能演示
├── app.caption_extraction.ipynb  # 图文对提取示例
└── app.molecule_extracrtion.ipynb # 分子提取示例
```

## 详细文档

项目提供了丰富的示例和教程，位于 `playground/` 目录下：

- **快速开始**：`playground/01.quick_start.ipynb` - 基础用法教程，包括 PDF 和图片解析、多种格式输出
- **高级用法**：`playground/02.advance.ipynb` - 高级功能教程，包括图片+图题+图注、表格+表题+表注、分子+分子索引、公式+公式索引的提取
- **异步回调**：`playground/04.use_callbacks.py` - 异步回调演示，用于在异步解析任务完成后自动接收通知和结果
- **图文对提取**：`playground/app.caption_extraction.ipynb` - 图文对提取完整示例
- **分子提取**：`playground/app.molecule_extracrtion.ipynb` - 分子结构提取示例

## 注意事项

1. **并发限制**：公开 UniParser 服务最高仅允许 5 并发，使用时请注意控制并发数量
2. **API 密钥**：需要配置有效的 API 密钥才能使用服务，可通过环境变量 `UNIPARSER_API_KEY` 设置
3. **服务端点**：不同 host 对应功能不完全相同，解析质量也不一样，具体请在售后群中咨询
4. **图文对提取**：必须使用特定端口（30001）进行解析，其他接口不支持提取图文对
5. **Token 复用**：解析任务提交后会返回一个 token，可以持有该 token 多次获取不同格式的结果

## 依赖要求

主要依赖包请参考 `requirements.txt`，包括：

- PyMuPDF
- pandas
- pillow
- opencv-python
- numpy
- scipy
- lxml
- 等

## 许可证

[待补充]

## 联系方式

如有问题，请在售后群中咨询。
