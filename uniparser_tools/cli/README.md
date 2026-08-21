# UniParser CLI 使用说明

`uniparser` 是 UniParser 的命令行工具，用于把 PDF、图片或公网 PDF 链接解析成 Markdown，并保存到本地目录。

官网与 API Key 申请：[https://uniparser.dp.tech/](https://uniparser.dp.tech/)

---

## 安装

需要 **Python 3.11+**。在运行 `uniparser` 的同一 Python 环境中执行：

```bash
pip install "git+https://github.com/dptech-corp/UniParser-Tools.git"
```

> PyPI 尚未发布，请使用上述 git 安装。只运行 `pip install -r requirements.txt` **不会**出现 `uniparser` 命令。

在本仓库内开发时，可在仓库根目录执行：

```bash
cd /path/to/UniParser-Tools
pip install -e .
```

检查是否安装成功：

```bash
uniparser --help
```

---

## 快速开始

**第一次使用：配置 API Key**

```bash
uniparser auth
```

按提示在 [https://uniparser.dp.tech/](https://uniparser.dp.tech/) 获取 Key 并粘贴保存。配置会写入 `~/.uniparser/config.yaml`，下次无需重复输入。

**解析一个 PDF**

```bash
uniparser parse /path/to/report.pdf
```

本地 PDF 使用 multipart 直传，上传窗口为 60 秒，并要求服务端生成 token。只有服务端成功接收文件并创建任务后，CLI 才会记录可恢复 token。

默认把结果保存到：

```text
~/Uni-Parser-Skill/report/
├── report.md           # 解析后的 Markdown
├── pages_tree.json     # 文档版面结构（可选查阅）
├── formatted_meta.json # 格式化元数据
└── trigger_meta.json   # 任务信息（含 token，便于中断后恢复）
```

终端会显示 `Parsing... report.pdf`，完成后打印 Markdown 与输出目录路径。

---

## 命令一览

| 命令 | 作用 |
|------|------|
| `uniparser auth` | 配置 / 查看 API Key |
| `uniparser parse INPUT` | 解析文件或 PDF 链接 |
| `uniparser fetch --token TOKEN` | 用已有任务 token 重新下载结果 |
| `uniparser health` | 检查服务是否可用 |
| `uniparser version` | 查看工具版本 |

---

## 配置 API Key

### 交互式配置（推荐）

```bash
uniparser auth
```

已有 Key 时直接回车可保留当前配置。

### 其他方式

也可任选其一（优先级从高到低）：

1. 命令行临时指定：`uniparser --api-key YOUR_KEY parse file.pdf`
2. 环境变量：`export UNIPARSER_API_KEY="YOUR_KEY"`
3. 配置文件：`~/.uniparser/config.yaml`（由 `uniparser auth` 写入）

### 查看与检查

```bash
uniparser auth --show      # 查看 Key 来源与脱敏后的 Key
uniparser auth --verify    # 检查是否已配置（不联网校验）
```

### 使用实例

#### 首次配置

```bash
uniparser auth
```

终端交互示例（输入 Key 后）：

```text
UniParser API Key Setup
Get your API key from: https://uniparser.dp.tech/

Enter your API key: ********
API key saved successfully
```

写入 `~/.uniparser/config.yaml`：

```yaml
api_key: your-api-key-here
```

#### 保留已有 Key

若已配置过，直接回车即可保留：

```text
Current API key source: config
Enter new API key (or press Enter to keep current):
Keeping existing API key.
```

#### 查看当前配置（`--show`）

```bash
uniparser auth --show
```

```text
API key source: config
API key: abcd...mnop
```

未配置时：

```text
No API key configured.
Run 'uniparser auth' to set up your API key.
```

#### 检查是否已配置（`--verify`）

```bash
uniparser auth --verify
```

```text
API key is configured.
  Source: config
```

---

## 解析文档：`parse`

### 基本用法

```bash
# 本地 PDF
uniparser parse paper.pdf

# 本地图片（png、jpg、webp 等）
uniparser parse figure.png

# 公网 PDF 链接
uniparser parse https://example.com/paper.pdf
```

### 常用选项

| 选项 | 说明 |
|------|------|
| `-o` / `--output-dir DIR` | 首选输出目录（默认 `~/Uni-Parser-Skill/<文件名>/`）；已存在时自动使用同级后缀目录 |
| `--async` | 异步提交任务（适合较大文档） |

### 解析配置（7 类语义）

未指定的字段使用 **scientific-paper** 默认值（与网站推荐科学文献配置一致）。只写你关心的 `--xxx` 即可覆盖对应项。

| 选项 | 含义 | 可选值 | 默认值 |
|------|------|--------|--------|
| `--textual` | 普通文本 | `disable` / `ocr-fast` / `ocr-hq` / `digital` / `base64` | `ocr-hq` |
| `--equation` | 数学公式 | `disable` / `ocr-fast` / `ocr-hq` / `base64` | `ocr-hq` |
| `--table` | 表格 | 同上 | `ocr-hq` |
| `--chart` | 图表 | 同上 | `base64` |
| `--figure` | 插图 | 同上 | `base64` |
| `--expression` | 化学反应式 | 同上 | `base64` |
| `--molecule` | 化学分子 | 同上 | `ocr-fast` |

示例：

```bash
# 全部默认（scientific-paper）
uniparser parse paper.pdf

# 数字 PDF 直接抽文本，其余保持默认
uniparser parse paper.pdf --textual digital

# 关闭分子解析
uniparser parse paper.pdf --molecule disable

# 多项覆盖
uniparser parse paper.pdf --textual digital --table ocr-fast --molecule disable

# 与输出目录、异步组合
uniparser parse paper.pdf -o ./results/ --async
```

`trigger_meta.json` 会记录本次实际提交的 `trigger_kwargs`，便于复现配置。

### 使用实例

#### 解析本地 PDF（默认输出目录）

```bash
uniparser parse paper.pdf
```

解析过程中 **stderr** 显示进度：

```text
Parsing... paper.pdf
```

完成后 **stdout** 打印结果路径：

```text
Token: a1b2c3d4e5f6789012345678901234ab
Markdown: /Users/you/Uni-Parser-Skill/paper/paper.md
Pages tree: /Users/you/Uni-Parser-Skill/paper/pages_tree.json
Output directory: /Users/you/Uni-Parser-Skill/paper
```

**stderr** 还会提示 meta 文件路径：

```text
Trigger meta: /Users/you/Uni-Parser-Skill/paper/trigger_meta.json
```

默认输出目录结构：

```text
~/Uni-Parser-Skill/paper/
├── paper.md              # 解析后的 Markdown（主要结果）
├── pages_tree.json       # 版面结构树
├── formatted_meta.json   # 格式化元数据（不含正文）
└── trigger_meta.json     # 任务 token 与 trigger_kwargs
```


#### 指定输出目录与解析参数

```bash
uniparser parse paper.pdf -o ./out/paper --textual digital --molecule disable
```

`trigger_meta.json` 中 `trigger_kwargs` 会反映覆盖项，例如：

```json
"trigger_kwargs": {
  "textual": "digital",
  "equation": "ocr-hq",
  "table": "ocr-hq",
  "chart": "base64",
  "figure": "base64",
  "expression": "base64",
  "molecule": "disable",
  "sync": true
}
```

#### 三种输入类型

| 输入 | 命令 | 进度提示（stderr） |
|------|------|-------------------|
| 本地 PDF | `uniparser parse report.pdf` | `Parsing... report.pdf` |
| 本地图片 | `uniparser parse figure.png` | `Parsing... figure.png` |
| 公网 PDF 链接 | `uniparser parse https://example.com/paper.pdf` | `Parsing... paper.pdf` |

#### 常见错误

文件不存在时，**stderr** 输出 JSON 错误（exit code 1）：

```bash
uniparser parse /no/such/file.pdf
```

```json
{"ok": false, "error": {"code": "INPUT_ERROR", "message": "File not found: /no/such/file.pdf"}}
```

输出目录已存在时，旧目录保持不变，并自动创建第一个可用的同级后缀目录：

```bash
uniparser parse paper.pdf -o ./out/paper
```

```text
./out/paper    # 已有结果，不修改
./out/paper_1  # 本次实际输出；若也存在则继续使用 paper_2
```

请以成功结果中的 `output_dir` 为准。程序不会复用或删除任何已有目录。

> 出于安全原因，根目录、HOME、当前工作目录及 Git 元数据目录不能作为首选输出目录。

> 说明：成功信息在 stdout；进度 `Parsing...` 与部分路径提示在 stderr；错误统一为 stderr 单行 JSON。

### 输出说明

解析成功后，输出目录中通常包含：

| 文件 | 用途 |
|------|------|
| `<文件名>.md` | 完整 Markdown 正文（主要结果） |
| `pages_tree.json` | 版面结构树，便于按章节/块定位 |
| `formatted_meta.json` | 格式化元数据（不含 `content` 正文） |
| `trigger_meta.json` | 任务 token、输入信息及 `trigger_kwargs`，供 `fetch` 恢复使用 |

大文档可能需要等待数分钟；详见上文「使用实例」中的终端输出示例。

---

## 恢复下载：`fetch`

若解析中断，或你已有任务 token，可用 `fetch` 继续拉取结果（不会重新上传文件）：

```bash
uniparser fetch --token YOUR_TOKEN
```

CLI 触发时固定发送 `token=None` 和 `server_generated_token=True`。可用于恢复的 token 只能从成功
`parse` 的 JSON 输出或 `trigger_meta.json` 中获取。失败 JSON 可能保留标准 `token` 字段用于诊断，但不得
将它传给 `fetch`。

| 选项 | 说明 |
|------|------|
| `-o` / `--output-dir DIR` | 首选输出目录（默认 `~/Uni-Parser-Skill/token_<前8位>/`）；已存在时自动使用同级后缀目录 |

```bash
uniparser fetch --token abcdef1234567890 -o ./out/
```

---

## 其他命令

### 检查服务

```bash
uniparser health
```

需要已配置 API Key。

### 查看版本

```bash
uniparser version
```

未配置 API Key 时仍会显示本地工具版本；配置后可同时查看远端服务版本。

---

## 脚本与自动化：`--json`

在命令**最前面**加上 `--json`，成功时会在**标准输出（stdout）**打印 JSON，便于脚本解析：

```bash
uniparser --json parse paper.pdf
uniparser --json fetch --token YOUR_TOKEN
```

注意：`--json` 必须写在子命令**之前**：

```bash
uniparser --json parse paper.pdf    # 正确
uniparser parse paper.pdf --json    # 错误
```

### `parse` 成功时的输出示例

```bash
uniparser --json parse /path/to/report.pdf
```

**stderr**（进度，与是否 `--json` 无关）：

```text
Parsing... report.pdf
```

**stdout**（成功时唯一的 JSON，exit code 0）：

```json
{
  "ok": true,
  "output_dir": "/Users/you/Uni-Parser-Skill/report",
  "pages_tree_path": "/Users/you/Uni-Parser-Skill/report/pages_tree.json",
  "markdown_path": "/Users/you/Uni-Parser-Skill/report/report.md",
  "content_chars": 12345,
  "token": "a1b2c3d4e5f6789012345678901234ab",
  "input_type": "file",
  "trigger_meta_path": "/Users/you/Uni-Parser-Skill/report/trigger_meta.json"
}
```

| 字段 | 含义 |
|------|------|
| `ok` | 是否成功（`true`） |
| `output_dir` | 结果目录绝对路径 |
| `markdown_path` | 主 Markdown 文件路径 |
| `pages_tree_path` | 版面结构树 JSON 路径 |
| `content_chars` | Markdown 正文字符数（随文档而异） |
| `token` | 任务 token，可用于 `uniparser fetch --token` |
| `input_type` | 输入类型：`file` / `image` / `url` |
| `trigger_meta_path` | `trigger_meta.json` 路径（含 `trigger_kwargs`） |

脚本中可只解析 stdout，例如：

```bash
uniparser --json parse paper.pdf 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['markdown_path'])"
```

失败时仍为 **stderr 单行 JSON**（`ok: false`），见下方「常见问题」；exit code 为 1。失败 JSON 中的
`token` 仅用于诊断；只有成功 `parse` 输出或 `trigger_meta.json` 中的 token 可用于 `fetch`。

---

## 常见问题

**提示 `No API key found`**

运行 `uniparser auth`，或设置环境变量 `UNIPARSER_API_KEY`。

**提示 `File not found`**

检查 `parse` 后的文件路径是否正确（建议使用绝对路径）。

**指定的输出目录已存在**

无需额外参数。程序会自动创建同级后缀目录（例如 `paper_1`），并在结果中返回实际路径。

**命令 `uniparser` 找不到**

确认已执行 `pip install "git+https://github.com/dptech-corp/UniParser-Tools.git"`（或在本仓库内 `pip install -e .`），且与当前终端使用的是同一 Python 环境。

**解析时间较长**

属正常现象；请等待 `Parsing...` 出现后保持终端不要关闭。若中断，用 `trigger_meta.json` 中的 token 执行 `uniparser fetch`。

**提示 `UPLOAD_ERROR` / write timeout**

本次直传失败。请重新运行 `parse`；错误结果中的 token 即使存在也仅供诊断，不能用于 `fetch`。

**提示 `TOKEN_NOT_FOUND` 或持续 `status: undefined`**

CLI 会在三次检查后停止，不会继续等待 1800 秒。请确认 token 来自成功输出或 `trigger_meta.json`，并检查任务结果是否仍在 24 小时保留期内。

---

## 推荐使用流程

```bash
# 1. 安装并配置
pip install "git+https://github.com/dptech-corp/UniParser-Tools.git"
uniparser auth

# 2. 解析
uniparser parse ~/Documents/paper.pdf

# 3. 打开结果
open ~/Uni-Parser-Skill/paper/paper.md
```

若需中断后恢复：

```bash
# 从 trigger_meta.json 复制 token
uniparser fetch --token <token>
```
