UNIPARSER_BASE_URL = "https://uniparser.dp.tech/"

POLL_INTERVAL_SEC = 3
POLL_TIMEOUT_SEC = 1800
UNDEFINED_MAX_POLLS = 3
DIRECT_UPLOAD_REQUEST_TIMEOUT = (60.0, 60.0)
DIRECT_SYNC_UPLOAD_REQUEST_TIMEOUT = (60.0, 1860.0)

PENDING_STATUSES = frozenset({"waiting", "processing"})

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"})
