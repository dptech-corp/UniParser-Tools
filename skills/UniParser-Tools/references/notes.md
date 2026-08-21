# Important Notes

## Result Quality and Retention

- High-quality modes use generative models and may omit, misread, misassociate, or add plausible-looking content across text, tables, equations, charts, figures, reactions, and molecules. Verify critical fields, numbers, equations, names, and structures against the source. Do not use parsing output as the sole basis for high-risk decisions.
- High-quality table parsing recovers semantics and structure but does not provide precise source-page coordinates for each cell. Choose a method that explicitly provides position data when downstream work requires overlays, highlighting, or coordinate-level auditing.
- Chart parsing may recover labels, legends, axes, values, or trends incorrectly. Verify precise values and interpretations against the original chart.
- High-quality modes are slower. Prefer asynchronous submission with suitable polling, callbacks, and timeouts for long documents or batches.
- Online parsing results are retained for only **24 hours**. Fetch and store required results promptly; a task token is not a long-term storage reference.

## Key Points

1. **Concurrency Limit**: Maximum 5 concurrent requests on public service

2. **Token Reuse**: A token can be used multiple times to fetch different formats

3. **Host Selection**: Different hosts may have different features/quality
   - `https://uniparser.dp.tech/` - Official site

4. **Callback Verification**: Use HMAC-SHA256 with `callback_secret` to verify callbacks
   ```python
   import hmac
   import hashlib


   def verify_callback(raw_body: bytes, signature: str, secret: str) -> bool:
       if not signature.startswith("sha256="):
           return False
       expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
       return hmac.compare_digest(expected, signature[len("sha256=") :])
   ```
   Read `raw_body` before JSON parsing and take `signature` from the
   `X-UniParser-Signature` header. The body is not wrapped in
   `checksum` / `content` fields.

5. **Ordering Methods**: Default is `GapTree`; alternatives: `Naive`, `XYCut`, `XYCutExp`

6. **Page Selection**: Use `pages=[1, 2, 3]` to parse specific pages only
   ```python
   result = parser.trigger_file(
       file_path="./document.pdf",
       pages=[1, 2, 3],  # Only parse pages 1, 2, 3
   )
   ```

## Error Response Format

All API methods return a dict with consistent structure:

```python
# Success
{
    "status": "success",
    "token": "abc123...",
    ...
}

# SDK error
{
    "status": "error",
    "token": "abc123...",
    "message": "Error description",
    "description": "Detailed traceback (optional)"
}
```

The low-level SDK retains its deterministic-token behavior for backward compatibility and may include that local value
in an error response. This is not part of the Agent Skill workflow. CLI and MCP always send `token=None` with
`server_generated_token=True` and only persist the `token` from a successful trigger response; never use an SDK error
token with CLI `fetch`.

## Common Error Messages

CLI workflow errors (config, duplicate token, 502, etc.) are documented in SKILL.md **Common issues**. This table covers additional SDK/API messages when calling the client directly:

| Error | Cause | Solution |
|-------|-------|----------|
| `token: ... contains illegal characters` | Invalid token format | Token must match `^[-\._?=&a-zA-Z0-9]{1,128}$` |
| `host must start with http or https` | Invalid host URL | Use full URL including protocol |
