# Important Notes

## Key Points

1. **Concurrency Limit**: Maximum 5 concurrent requests on public service

2. **Token Reuse**: A token can be used multiple times to fetch different formats

3. **Host Selection**: Different hosts may have different features/quality
   - `https://uniparser.dp.tech/` - Official site

4. **Callback Verification**: Use HMAC-SHA256 with `callback_secret` to verify callbacks
   ```python
   import hmac
   import hashlib

   def verify_callback(content, checksum, secret):
       expected = hmac.new(secret.encode(), content.encode(), hashlib.sha256).hexdigest()
       return hmac.compare_digest(expected, checksum)
   ```

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

# Error
{
    "status": "error",
    "token": "abc123...",
    "message": "Error description",
    "description": "Detailed traceback (optional)"
}
```

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `token: ... contains illegal characters` | Invalid token format | Token must match `^[-\._?=&a-zA-Z0-9]{1,128}$` |
| `api_key can not be empty` | Missing API key | Set `UNIPARSER_API_KEY` environment variable |
| `host must start with http or https` | Invalid host URL | Use full URL including protocol |
