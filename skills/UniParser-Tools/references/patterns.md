# Common Patterns

## Pattern 1: Simple PDF to Markdown

```python
result = parser.trigger_file(file_path=pdf_path, textual=ParseModeTextual.DigitalExported)
token = result["token"]

result = parser.get_formatted(token, content=True, textual=FormatFlag.Markdown)
print(result["content"])
```

## Pattern 2: Extract Figures with Captions

```python
from uniparser_tools.utils.convert import dict2obj

result = parser.trigger_file(
    file_path=pdf_path,
    figure=ParseMode.DumpBase64,
    textual=ParseModeTextual.DigitalExported,
)
token = result["token"]

result = parser.get_result(token, pages_tree=True)
pages_tree = dict2obj(result["pages_tree"])

# Navigate tree to find figure groups
for page in pages_tree:
    for item in page:
        if item.type == "figuregroup":
            # item.items contains figure + caption
            print(item.format_as(FormatFlag.Markdown))
```

## Pattern 3: Async Processing with Callback

```python
result = parser.trigger_file(
    file_path=pdf_path,
    sync=False,  # Async mode - must be False for callbacks
    callback_url="https://your-server.com/callback",
    callback_secret="your-secret-key",
    textual=ParseModeTextual.DigitalExported,
)
# Service will POST to callback_url when done
```

### Callback Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `sync=False` | Yes | Must be `False` to enable async callback mode |
| `callback_url` | Yes | Your server endpoint that accepts POST requests |
| `callback_secret` | Yes | Shared secret for HMAC-SHA256 signature verification |

### Callback Payload and Signature

When parsing completes, the service sends the result as the JSON request body. The signature is not embedded in the JSON; it is sent as `X-UniParser-Signature: sha256=<digest>`.

```json
{
    "token": "abc123...",
    "status": "success",
    "result": { ... }
}
```

### Verify Callback Signature

```python
import hmac
import hashlib
from flask import Flask, request

app = Flask(__name__)
CALLBACK_SECRET = "your-secret-key"

@app.route('/callback', methods=['POST'])
def handle_callback():
    body = request.get_data()
    signature = request.headers.get('X-UniParser-Signature', '')
    received_checksum = signature.split('=', 1)[-1]

    expected = hmac.new(
        CALLBACK_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(received_checksum, expected):
        return {'error': 'Invalid signature'}, 401

    data = request.get_json()
    token = data['token']
    print(f"Task {token} completed!")
    return {'status': 'ok'}
```

## Pattern 4: Mixed Format Output

```python
result = parser.get_formatted(
    token,
    content=True,
    textual=FormatFlag.Markdown,   # Text as Markdown
    table=FormatFlag.Html,         # Tables as HTML
    equation=FormatFlag.Latex,     # Equations as LaTeX
    figure=FormatFlag.Markdown,    # Figures as Markdown img
)
```

## Pattern 5: Parse Image (Snippet)

```python
result = parser.trigger_snip(
    snip_path="./figure.png",
    textual=ParseModeTextual.OCRFast,
    equation=ParseMode.OCRFast,
)
token = result["token"]
```

## Pattern 6: Parse from URL or Object Storage

```python
result = parser.trigger_url(
    pdf_url="https://example.com/paper.pdf",
    textual=ParseModeTextual.DigitalExported,
    proxy=None,  # Optional proxy
)
token = result["token"]
```

The same method accepts `s3://`, `oss://`, `tos://`, and `file://<absolute path>` sources, including supported image files.

## Pattern 7: Fetch Partial or MinerU Output

```python
from uniparser_tools.common.constant import ThirdPartyFormatter

partial = parser.get_result(token, return_half=True, pages_tree=True)
mineru = parser.get_third_party_output(token, ThirdPartyFormatter.MinerU)
```
