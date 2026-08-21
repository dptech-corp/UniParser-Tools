# UniParser MCP Server

基于 [Model Context Protocol](https://modelcontextprotocol.io/) 的 MCP 服务，通过单一工具 `uniparser_parse` 调用 [UniParser](https://uniparser.dp.tech/) API（经 `uniparser-tools` 的 `UniParserClient`）。

## Tool

| Tool | 说明 |
|------|------|
| `uniparser_parse` | 解析本地 PDF、本地图片或公网 PDF URL；落盘 Markdown + `pages_tree.json`；返回路径与 `content_preview` |

### `uniparser_parse` 参数

提供 **三选一** 输入：`file_path`、`image_path`、`pdf_url`。

| 参数 | 说明 |
|------|------|
| `output_dir` | 可选的首选目录；默认 `~/Uni-Parser-Skill/<stem>/`；已存在时自动使用同级 `<name>_1`、`<name>_2` 等新目录 |
| `async_mode` | `sync=false` 提交后轮询直至完成 |
| `textual` … `molecule` | 7 个语义字段，默认 scientific-paper preset |

成功返回 JSON（Pydantic）：`markdown_path`、`pages_tree_path`、`content_preview`（默认 2000 字）、`message` 等。
调用方应以返回的 `output_dir` 为准；服务不会复用或删除已有目录。

本地 PDF 固定通过 `trigger_file` 直接 multipart 上传，不使用 TOS 或自动回退。MCP 触发时固定发送
`token=None` 和 `server_generated_token=True`；同步请求使用 `(60, 1860)`，异步请求使用 `(60, 60)`，
只有服务端确认的 token 才能进入结果轮询。

> 出于安全原因，根目录、HOME、当前工作目录及 Git 元数据目录不能作为首选输出目录。

健康检查、版本查询、按 token 手动恢复请使用 CLI：`uniparser health`、`uniparser version`、`uniparser fetch`。

失败返回的 `error.code` 可能为 `CONFIG_ERROR`、`INPUT_ERROR`、`UPLOAD_ERROR`、`PARSE_ERROR` 或
`TOKEN_NOT_FOUND`。触发失败时可能保留标准 `token` 供诊断，但 MCP 不会使用它或进入恢复流程；只有
成功触发返回的 token 才会被轮询。`undefined` 最多检查三次，不会持续等待 1800 秒。

## 环境变量

| 变量 | 说明 | 默认 |
|------|------|------|
| `UNIPARSER_API_KEY` | 必填，`X-API-Key` | — |
| `UNIPARSER_BASE_URL` | API 根 URL | `https://uniparser.dp.tech` |
| `OUTPUT_DIR` | 输出根目录 | `~/Uni-Parser-Skill` |
| `UNIPARSER_PREVIEW_CHARS` | `content_preview` 长度 | `2000` |
| `UNIPARSER_MCP_TRANSPORT` | `stdio` / `sse` / `streamable-http` | `stdio` |

## 安装与运行

```bash
cd mcp_server
uv sync
uv run python -m uniparser_mcp
```

## 测试

```bash
cd mcp_server
uv sync --extra dev
uv run pytest tests/ -v
```

## Cursor / Claude Code 接入示例

先克隆本仓库并在本目录执行 `uv sync`，再在 MCP 配置中增加如下内容。**必须**将两处占位符改成你的本机值，否则 MCP 无法启动：

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

`UNIPARSER_BASE_URL` 可省略（默认云服务）；本地自托管时设置为 `http://127.0.0.1:40001` 等。

本地 PDF 同步直传显式使用 `(60, 1860)`，异步直传显式使用 `(60, 60)`；轮询和结果获取使用
`UniParserClient.request_timeout`。这些都是客户端 HTTP 超时，与服务端解析预算相互独立。
