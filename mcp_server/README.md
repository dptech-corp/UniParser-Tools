# UniParser MCP Server

基于 [Model Context Protocol](https://modelcontextprotocol.io/) 的 **stdio** 服务，通过 MCP `tools` 调用 UniParser 用户面 HTTP API。不在 MCP 进程内加载解析栈，仅做 HTTP 转发，并与 [`config.yaml`](./config.yaml) 中的默认参数合并。

## 前置条件

1. **已运行 UniParser 用户服务**（提供 `GET /health`、`POST /trigger-file-async`、`POST /trigger-url-async`、`POST /get-formatted` 等）。若在 UniParser 主仓库中开发，典型启动方式为（路径以主仓库为准）：

   ```bash
   python services/server_user.py
   ```

   默认监听 **40001** 端口；本地联调时请将 `UNIPARSER_BASE_URL` 设为 `http://127.0.0.1:40001`。

2. **Python 3.10+**，推荐使用 [uv](https://github.com/astral-sh/uv) 在 **本目录**（`src/mcp_server`）安装依赖，避免与主项目其它依赖版本冲突：

   ```bash
   cd mcp_server
   uv sync
   ```

## 环境变量

MCP 进程通过 [`UniParserClient`](../uniparser_tools/api/clients.py) 访问服务，**以下两项在运行 MCP 时均为必填**（未设置时工具会返回明确错误，而非静默使用默认 URL）：

| 变量 | 说明 |
|------|------|
| `UNIPARSER_BASE_URL` | 用户服务根 URL，**无代码内默认值**。本地示例：`http://127.0.0.1:40001` |
| `UNIPARSER_API_KEY` | 对应请求头 `X-API-Key`；与服务端鉴权配置一致 |

密钥仅通过环境变量注入，勿写入 MCP 工具参数或提交到版本库。

**集成测试**（`pytest` 标记 `integration`）还会读取：

| 变量 | 说明 |
|------|------|
| `UNIPARSER_BASE_URL` | 未设置时，[`tests/conftest.py`](./tests/conftest.py) 默认使用 `https://uniparser.dp.tech`（与 MCP 手动配置本地 URL 的行为不同，请注意） |
| `UNIPARSER_API_KEY` | 未设置时跳过全部集成测试 |
| `UNIPARSER_INTEGRATION` | 设为 `0` / `false` / `no` / `skip` 时，不探测服务并跳过集成测试（适用于 CI 无后端场景） |

## 配置文件 `config.yaml`

与 `uniparser_mcp` 包同级目录下的 [`config.yaml`](./config.yaml) 控制 **trigger** 与 **get-formatted** 的默认字段（与 HTTP API 一致）：

| 段名 | 用途 |
|------|------|
| `default_trigger_file` | `uniparser_parse_file` → `POST /trigger-file-async` 的默认参数（如 `lang`、`sync`、`textual`、`table` 及各模态解析档位） |
| `default_trigger_url` | `uniparser_parse_url` → `POST /trigger-url-async` 的默认参数 |
| `default_get_result` | 解析完成后 `POST /get-formatted` 的默认参数（如各类型输出为 `markdown`、`content: true` 等） |

修改解析行为时，优先编辑该文件；若需在工具层暴露更多 MCP 参数，需扩展 [`uniparser_mcp/server.py`](./uniparser_mcp/server.py)。

## Tools

| Tool | 说明 |
|------|------|
| `uniparser_health` | `GET /health`，返回服务健康状态字符串 |
| `uniparser_version` | `GET /version`，返回版本信息 |
| `uniparser_parse_file` | 参数：本机 PDF **绝对路径**。流程：`POST /trigger-file-async` → 成功后再 `POST /get-formatted`，返回合并后的 `content` 文本 |
| `uniparser_parse_url` | 参数：服务端可访问的文件 **URL**（支持 `http(s)`、`s3`、`oss`、`tos`、`file`）。流程：`POST /trigger-url-async` → `POST /get-formatted`，返回 `content` |

`uniparser_parse_file` 仅适合**同机**或**体积较小**的 PDF；大文件或带宽受限时，建议将文件放到公网可访问地址后使用 `uniparser_parse_url`。

## 本地运行

```bash
cd mcp_server
uv run python -m uniparser_mcp
```

或：

```bash
uv run uniparser-mcp
```

## 测试

```bash
cd mcp_server
uv sync --extra dev
uv run pytest tests/ -v
```

调试时要看 `print` 或中间步骤输出，可关闭输出捕获：

```bash
uv run pytest tests/ -v -s
# 或
uv run pytest tests/ -v --capture=no
```

需要把 **logging** 打到终端时：

```bash
uv run pytest tests/ -v -s -o log_cli=true -o log_cli_level=DEBUG
```

- **集成测试**：[`tests/test_client_integration.py`](./tests/test_client_integration.py)（标记 `integration`）。会话开始时用 `httpx` 请求 `{UNIPARSER_BASE_URL}/health`；不可达或非正常状态则**整会话 skip**。仅跑集成：`uv run pytest tests/test_client_integration.py -v`。CI 无服务：`UNIPARSER_INTEGRATION=0 uv run pytest tests/ -v`。

## Cursor / Claude 等 MCP 接入示例

在 MCP 配置中增加（将 `/path/to/UniParser-Tools/src/mcp_server` 换为你的本机路径）：

```json
{
  "mcpServers": {
    "uniparser": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/UniParser-Tools/src/mcp_server",
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

`UNIPARSER_BASE_URL`、`UNIPARSER_API_KEY` 均需在 `env` 中配置（示例仅作占位）。

## 故障排查

| 现象 | 可能原因 |
|------|----------|
| 工具返回「未设置 UNIPARSER_BASE_URL」或「未设置 UNIPARSER_API_KEY」 | MCP 的 `env` 未注入或拼写错误 |
| `uniparser_health` 失败 | 用户服务未启动、端口/防火墙不一致、或 `UNIPARSER_BASE_URL` 与真实监听地址不符 |
| `uniparser_parse_file` 失败 | 路径不存在、无读权限，或服务端无法访问该路径（远程部署时常见） |
| `uniparser_parse_url` 长时间无结果 | PDF 较大或 URL 较慢；`UniParserClient` 对单次请求使用固定超时（见客户端实现），可适当调大服务或网关超时 |

## 与主仓库的关系

本目录为**独立**子项目（自有 [`pyproject.toml`](./pyproject.toml)），通过 `[tool.uv.sources]` 以可编辑方式依赖上一级 [`uniparser_tools`](../)。主仓库根 `pyproject.toml` 中若有可选依赖组 `mcp-server`，仅作文档引用；**推荐**始终在 `src/mcp_server/` 下执行 `uv sync`，以保证 MCP 与主服务依赖隔离。
