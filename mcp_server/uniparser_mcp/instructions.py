"""Agent guidance for the UniParser MCP server."""

SERVER_INSTRUCTIONS = """
You are connected to the UniParser MCP server for document parsing.

## When to call uniparser_parse

Call uniparser_parse immediately, without asking for confirmation, when:
- The user provides an absolute local path to a PDF or image (for example `/Users/.../paper.pdf`)
- The user provides a public PDF URL (`http://` or `https://`)
- The user asks to parse, convert, read, or extract text from a document

## How to call it

- Provide exactly one input: `file_path`, `image_path`, or `pdf_url`
- Use absolute paths for local files when possible
- Optional parse fields default to the scientific-paper preset (high-quality OCR for text/tables/equations)
- Use `output_dir` only when the user requests a specific save location. It is a preferred path: if it is
  occupied, a new sibling such as `<name>_1` is created automatically. Use the actual `output_dir` returned.
- Large documents: read `markdown_path` and `pages_tree_path` from the response; `content_preview` is only a short preview

## After calling uniparser_parse

On success, always include the `message` field from the tool response in your reply to the user.
On failure, report `error.message`. Never poll a token from a failed trigger; it is diagnostic only. Only a
token returned with `status=success` may be polled. Do not attempt recovery from a failed trigger.
Stop retrying when the error code is `TOKEN_NOT_FOUND`.
"""
