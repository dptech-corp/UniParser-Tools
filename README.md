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

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

使用 **`uniparser` 命令行工具**时，还需将本仓库安装为可编辑包（入口在 `pyproject.toml` 中注册）：

```bash
pip install -e .
```

> **说明：** 仅执行 `pip install -r requirements.txt` **不会**注册 `uniparser` 命令；SDK 开发用前者即可，CLI 必须执行 `pip install -e .`。

开发和运行测试时，请安装测试依赖：

```bash
pip install -e ".[test]"
```

当 `.env` 中配置了 `UNIPARSER_API_KEY` 时，完整测试会连接真实服务，并分别
通过本地 PDF、公开 HTTPS PDF 和图片 snip 提交 3 次计费解析：

```bash
python -m pytest -q
```

若 `.env` 不在当前仓库，可通过 `UNIPARSER_DOTENV_PATH=/path/to/.env`
显式指定。

安装后验证：

```bash
uniparser --help
```

CLI 完整说明见 [`uniparser_tools/cli/README.md`](./uniparser_tools/cli/README.md)。

## 使用风险与结果保留

- 高质解析依赖生成式模型，文本、表格、公式、图表、图片、反应式和分子等结果都可能出现遗漏、误识别、错误关联或模型补全。关键字段、数字、公式、名称和结构必须对照原文核验，不应将解析结果作为科研、合规、医疗或财务等高风险判断的唯一依据。
- 高质表格解析侧重语义和结构，不提供每个单元格在原页面中的精确位置。需要坐标、回标或区域高亮时，应选择明确提供位置数据的处理方式，并先验证输出结构。
- 图表解析可能无法准确恢复标签、图例、坐标轴、数据点和趋势；精确数值及趋势判断必须对照原图复核。
- 高质模式通常更慢。长文档或批量任务建议使用异步提交，并合理配置轮询、回调和超时。
- 在线服务的解析结果仅保留 **24 小时**。任务完成后应立即下载并保存所需结果，不能将 token 当作长期存储引用。

## CLI 命令行

`uniparser` 提供 `auth`、`parse`、`fetch`、`health`、`version` 等子命令。

| 命令 | 说明 |
|------|------|
| `uniparser auth` | 交互式配置 API Key（写入 `~/.uniparser/config.yaml`） |
| `uniparser parse INPUT` | 解析本地 PDF/图片或公网 PDF URL |
| `uniparser fetch --token TOKEN` | 用已有 token 轮询并下载结果 |
| `uniparser health` | 检查服务健康状态（需要 API Key） |
| `uniparser version` | 查看本地包版本（无 API Key 时跳过远端查询） |

**API Key 优先级：** `--api-key` > `UNIPARSER_API_KEY` > `~/.uniparser/config.yaml`

**`--json` 须写在子命令之前：** `uniparser --json parse paper.pdf`

参数、输出文件、错误码等详见 [`uniparser_tools/cli/README.md`](./uniparser_tools/cli/README.md)。

首次使用：

```bash
uniparser auth
uniparser parse report.pdf
```

## API-Key 配置

所有请求都通过 `X-API-Key` 请求头认证，`UniParserClient` 会自动注入。

- **获取方式**：在 UniParser 服务首页（如 `https://uniparser.dp.tech/`）注册访客账号，或向运维/业务方申请长期 API-Key。
- **推荐存储**：运行 `uniparser auth`，或设置环境变量 `UNIPARSER_API_KEY`，避免在代码中硬编码。
- **错误处理**：Key 缺失/过期、限流等情况都会被客户端统一包装成 `{"status": "error", ...}` 返回，详见下方 [错误处理](#错误处理)。

```python
import os

parser = UniParserClient(
    host="https://uniparser.dp.tech/",
    api_key=os.getenv("UNIPARSER_API_KEY"),
    request_timeout=(10, 60),  # 普通请求：连接/读取超时
    sync_request_timeout=(10, 1860),  # 同步解析请求：连接/读取超时
    upload_request_timeout=(60, 300),  # TOS 上传：socket/响应超时
)
```

`request_timeout` 用于健康检查、结果获取和异步任务提交；`sync_request_timeout`
只用于 `sync=True` 的解析请求；`upload_request_timeout` 用于 TOS 文件内容上传。
它们是客户端 HTTP 超时，不等同于服务端解析预算。
客户端可作为上下文管理器使用，以及时关闭连接池：

```python
with UniParserClient(host=host, api_key=api_key) as parser:
    result = parser.version()
```

`version()` 会原样返回 `release/v1.3` 的模型路由信息，包括
`default_version`、`backend_versions`，以及后端声明的 `capabilities`；
可据此选择 `trigger_*()` 的 `model_version`。`get_constants()` 返回服务端
当前的 `LayoutType`、解析/格式枚举和 token 规则。`health()`、`version()`
和 `get_constants()` 也都支持单次 `http_timeout=`。

```python
service = parser.version()
default_model = service["default_version"]
capabilities = service["backend_versions"][default_model].get("capabilities", {})
constants = parser.get_constants()
```

## 解析配置：7 个语义类 + 2 个枚举

提交解析任务时（`trigger_file` / `trigger_snip` / `trigger_url`），可分别设置 7 类语义元素的处理模式：

| 字段 | 含义 | 枚举类型 |
|------|------|------|
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
| `0` | `Disable` | 禁用解析，不输出 |
| `1` | `OCRFast` | 快速 OCR（默认） |
| `2` | `OCRHighQuality` | 高质 OCR |

### `ParseModeTextual`（仅用于 `textual`）

| 取值 | 名称 | 含义 |
|------|------|------|
| `-1` | `DumpBase64` | 输出原始图像 Base64 |
| `0` | `Disable` | 不解析、不输出 |
| `1` | `OCRFast` | 快速 OCR |
| `2` | `OCRHighQuality` | 高质 OCR，支持行内公式 |
| `3` | `DigitalExported` | 从数字原生 PDF 直接抽取文字 |

### 提交任务的通用参数

三个提交入口已与 `release/v1.3` 对齐。`trigger_file`、`trigger_snip` 和
`trigger_url` 都支持以下参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `timeout` | `1800` | 服务端解析预算（秒），不是 HTTP 超时 |
| `http_timeout` | `None` | 仅覆盖本次请求的客户端 HTTP 超时 |
| `inplace_update` | `False` | 是否允许更新同 token 的已有任务 |
| `preset_layout` | `None` | 预设版面；可传 JSON 字符串或 Python 列表 |
| `model_version` | `None` | 指定服务端 `/version` 返回的模型版本 |
| `server_generated_token` | `False` | token 为空时交给服务端生成；默认保留历史确定性 token |
| `callback_url` / `callback_secret` | `None` | 异步任务完成回调及其验证密钥 |

`padding_snip` 只适用于文件和图片入口；`proxy` 只适用于 URL 入口。URL
入口支持服务端接受的 HTTP(S) 以及 S3、OSS、TOS 对象地址。`preset_layout`
在三个入口中都会按服务端契约编码成 JSON 字符串。

```python
result = parser.trigger_url(
    "tos://bucket/document.pdf",
    sync=False,
    model_version="v1.3",
    preset_layout=[[{"type": "textual", "bbox": [0, 0, 100, 30]}]],
    server_generated_token=True,
)
token = result["token"]
```

### TOS 预签名上传

本地文件可以先上传到 TOS，再把返回的 `source_url` 交给 `trigger_url`。
上传与解析刻意分成两步，调用上传助手不会自动启动计费解析：

```python
uploaded = parser.upload_files_to_tos(["./large-document.pdf"])
source_url = uploaded["files"][0]["source_url"]
result = parser.trigger_url(source_url, server_generated_token=True)
```

如需自行执行上传，可调用 `request_tos_upload_links()` 获取预签名 `PUT`
地址。该地址是短期 bearer credential，不应记录到日志或转发给其他服务。
客户端向预签名地址上传时不会携带 UniParser API Key；高层
`upload_files_to_tos()` 完成上传后也不会在返回值中保留 `upload_url`。
可通过客户端的 `upload_request_timeout=` 或单次调用的 `http_timeout=`
调整上传超时。

## 快速开始

> ‼️‼️‼️ 以下仅为代码功能示例，具体运行代码请参考 `playground/*.ipynb` ‼️‼️‼️

### 1. 初始化客户端

```python
import os
from uniparser_tools.api.clients import UniParserClient

# 设置 API 密钥
api_key = os.getenv("UNIPARSER_API_KEY")

# 初始化客户端
parser = UniParserClient(host="https://uniparser.dp.tech/", api_key=api_key)
```

### 2. 解析 PDF 文件（科学文献推荐默认）

```python
from uniparser_tools.common.constant import ParseMode, ParseModeTextual

# 科学文献解析模式（推荐默认值）
result = parser.trigger_file(
    file_path="./example.pdf",
    textual=ParseModeTextual.OCRHighQuality,  # high quality
    equation=ParseMode.OCRHighQuality,  # high quality
    table=ParseMode.OCRHighQuality,  # high quality
    chart=ParseMode.DumpBase64,  # original image base64
    figure=ParseMode.DumpBase64,  # original image base64
    expression=ParseMode.DumpBase64,  # original image base64
    molecule=ParseMode.OCRFast,  # fast
)

if result["status"] == "success":
    token = result["token"]
    print(f"解析成功，token: {token}")
```

### 3. 获取解析结果

#### 输出配置（`get_result` / `get_formatted` 通用开关）

| 开关 | 默认 | 说明 |
|------|------|------|
| `content` | `False` | 返回全文纯/富文本，适合 LLM |
| `objects` | `False` | JSON 语义块列表，适合语义分析 |
| `pages_dict` | `False` | 按页组织的原始解析布局 |
| `pages_tree` | `False` | 带父子关系的嵌套树，适合复杂分析 |
| `molecule_source` | `False` | 返回分子原始源（SMILES/mol 等） |

同一 token 可复用，多次获取不同组合不会重复计费。
两个结果接口都可用 `http_timeout=` 覆盖单次读取超时，适合包含大量对象或
Base64 源的大文档。

#### 输出格式（`FormatFlag`，仅作用于 `content` / `objects` 中的文本字段）

| 取值 | 适用场景 |
|------|------|
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

如需 MinerU 兼容结构，可直接调用第三方格式结果接口：

```python
from uniparser_tools.common.constant import ThirdPartyFormatter

result = parser.get_third_party_output(
    token,
    formatter=ThirdPartyFormatter.MinerU,
)
```

`dict2obj()` / `build_item()` 已对齐 `release/v1.3` 的结果模型，包括
文本块的 `contents + types` 行内公式/分子表示、分子的 `esmi` 字段，以及
完整 HTML 表格的 span 升级。服务端未来增加未知字段时，转换器会忽略未知
字段，而不是让已有客户代码因构造参数不匹配而崩溃。

### 账户与用量（只读）

解析客户端提供 `account` 命名空间，使用同一连接池和 API Key：

```python
profile = parser.account.get_current_user()
balance = parser.account.get_balance()
summary = parser.account.get_usage_summary(period="current_month")
usage = parser.account.list_usage_records(page=1, size=20)
transactions = parser.account.list_balance_transactions(page=1, size=20)
```

也可以独立创建只读账户客户端：

```python
from uniparser_tools.api.account import UniParserAccountClient

with UniParserAccountClient(host=host, api_key=api_key) as account:
    print(account.get_balance())
```

该封装不提供注册、资料更新、API Key 管理、充值或管理员写操作。用量明细
遵循服务端当前契约，只返回最近 14 天并分页；`size` 的服务端上限为 100。

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

`release/v1.3` 的回调 body 是原始 JSON 结果，不再包成
`{"checksum": ..., "content": ...}`。服务端对实际收到的 body bytes 使用
`callback_secret` 计算 HMAC-SHA256，并在
`X-UniParser-Signature: sha256=<hex>` 中发送签名；接收方必须在解析 JSON
之前，对原始 body bytes 验签。`Idempotency-Key` 可用于去重，
`X-UniParser-Callback-Attempt` 表示当前重试次数。

`callback_url` 仅允许搭配 `sync=False` 使用，且必须与 `callback_secret`
同时提供；部署方还可能对回调 host 配置 allowlist。

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
    textual=FormatFlag.Markdown,  # 文本使用 Markdown
    table=FormatFlag.Html,  # 表格使用 HTML
    equation=FormatFlag.Latex,  # 公式使用 LaTeX
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
|------|------|------|
| `status` | 任务响应或错误响应 | `"success"` / `"error"`（见 `StatusFlag`）；`version` 等信息接口不保证该字段 |
| `token` | 触发/查询类接口 | 本次任务的 token，出错也会带上以便追溯 |
| `description` | 错误时 | 服务端业务错误，或不含本地 traceback 的网络错误摘要 |
| `message` | 错误时 | 客户端请求阶段说明；非 JSON 响应会额外保留在 `body` |
| `http_status` | HTTP 4xx/5xx 时 | 原始 HTTP 状态码，同时保留服务端 JSON 错误体 |

> 直接调用 REST API（curl / 自研客户端）时才需要关注 `401/403/429/…` 等原始 HTTP 状态码，详见各部署实例 `<host>/api` 上的 Authentication 章节。

## 面向 AI Agent

本仓库提供 **Agent Skill**（[skills/UniParser-Tools/](./skills/UniParser-Tools/)），让 Cursor、Claude Code 等助手自动完成 PDF / 图片 / 公网 PDF 链接 → Markdown 与版面 JSON 的解析。用户只需要安装 Skill、准备 API Key，然后在对话里提出解析需求；CLI 安装与具体执行步骤由 Agent 按 [SKILL.md](./skills/UniParser-Tools/SKILL.md) 自动完成。

### 快速使用 Skill

**1. 安装 Skill**

使用 `skills` 命令安装：

```bash
npx skills add dptech-corp/UniParser-Tools
```

也可以手动安装：将 [skills/UniParser-Tools/](./skills/UniParser-Tools/) 整个目录发送给 Agent，并让 Agent 安装该 Skill。安装后重启 Agent，确保 Skill 列表中出现 **uniparser-tools**。

**2. 准备 API Key**

在 [https://uniparser.dp.tech/](https://uniparser.dp.tech/) 注册并申请 API Key。你可以让 Agent 按 Skill 指引配置，也可以提前设置环境变量：

```bash
export UNIPARSER_API_KEY="your-api-key"
```

不要把 API Key 直接粘贴到公开对话或代码仓库中。

**3. 在 Agent 中使用 Skill**

在 Agent 对话中上传文件、粘贴公网 PDF 链接，或使用类似表述即可触发 Skill，例如：

- 中文：`解析这个 PDF`、`PDF 转 Markdown`、`提取论文`、`文档解析`、`表格提取`、`公式识别`、`化学分子`
- 英文：`parse this PDF`、`extract this paper`、`PDF to markdown`、`UniParser`、`scientific paper`

支持的输入：**本地 PDF**、**本地图片**（png / jpg 等）、**可公网访问的 PDF URL**。

**4. 查看输出结果**

默认输出到 `~/Uni-Parser-Skill/<源文件主名>/`，通常包含：

| 文件 | 说明 |
|------|------|
| `{源文件主名}.md` | 解析得到的完整 Markdown |
| `pages_tree.json` | 结构化版面树（页面与语义块层次） |
| `formatted_meta.json` | 元数据（不含全文 `content`） |
| `trigger_meta.json` | 任务 token 与解析参数（供 `uniparser fetch` 中断恢复） |

Agent 会回复 Markdown 路径，并在需要版面结构时提供 `pages_tree.json`。大文档可能耗时数分钟至十余分钟；中断或重复任务可按 Skill 说明用 `trigger_meta.json` 中的 token 执行 `uniparser fetch`。

Agent 实现细节、CLI 命令与错误恢复见 [SKILL.md](./skills/UniParser-Tools/SKILL.md)。

## MCP Server

UniParser 提供了基于 [Model Context Protocol](https://modelcontextprotocol.io/) 的 MCP 服务，位于 `mcp_server/` 目录，支持通过 MCP 工具调用 UniParser HTTP API。

### 可用工具


| 工具                | 说明                                                               |
| ----------------- | ---------------------------------------------------------------- |
| `uniparser_parse` | 解析本地 PDF / 图片或公网 PDF URL；落盘 Markdown 与 `pages_tree.json`；返回路径与预览 |


健康检查、版本查询、按 token 恢复请使用 CLI（`uniparser health` / `version` / `fetch`）。详见 [mcp_server/README.md](./mcp_server/README.md)。

### 快速启动

```bash
cd mcp_server
uv sync
uv run python -m uniparser_mcp
```


| 变量                   | 说明                                |
| -------------------- | --------------------------------- |
| `UNIPARSER_API_KEY`  | 必填                                |
| `UNIPARSER_BASE_URL` | 可选，默认 `https://uniparser.dp.tech` |




### 接入 Cursor / Claude Code

先克隆本仓库并在 `mcp_server/` 下执行 `uv sync`，再在 MCP 配置中增加如下内容。**必须**将两处占位符改成你的本机值，否则 MCP 无法启动：

1. `"--directory"` 后的路径：把 `/path/to/UniParser-Tools/mcp_server` 替换为克隆到本机后的 `mcp_server` **绝对路径**（例如 macOS：`/Users/<you>/UniParser-Tools/mcp_server`）。
2. `UNIPARSER_API_KEY`：把 `your-api-key` 替换为你在 [https://uniparser.dp.tech/](https://uniparser.dp.tech/) 申请的真实 API Key。

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
        "UNIPARSER_API_KEY": "your-api-key"
      }
    }
  }
}
```

传输模式默认为 `stdio`，可通过 `UNIPARSER_MCP_TRANSPORT` 切换为 `sse` 或 `streamable-http`。

详细文档见 [mcp_server/README.md](./mcp_server/README.md)。

## 项目结构

```
uniparser_tools/
├── cli/              # uniparser 命令行工具
│   ├── commands/     # auth, parse, fetch, health, version
│   └── core/         # 配置、凭证、pipeline、输出
├── api/              # API 客户端
├── common/           # 通用常量和数据类
├── tools/            # 工具模块
│   └── caption_extraction/  # 图文对提取工具
├── utils/            # 工具函数
└── order/            # 排序算法

mcp_server/           # MCP 服务（独立子项目，仅 uniparser_parse tool）
├── uniparser_mcp/
└── pyproject.toml

playground/
├── 01.quick_start.ipynb          # 快速开始教程
├── 02.advance.ipynb              # 高级用法教程
├── 04.use_callbacks.py           # 异步回调功能演示
├── app.caption_extraction.ipynb  # 图文对提取示例
└── app.molecule_extracrtion.ipynb # 分子提取示例
```

## 详细文档

项目提供了丰富的示例和教程，位于 `playground/` 目录下：

- **CLI 命令行**：[`uniparser_tools/cli/README.md`](./uniparser_tools/cli/README.md) - `uniparser` 安装、子命令与参数说明
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
