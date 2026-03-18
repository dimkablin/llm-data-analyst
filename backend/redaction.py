from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_SENSITIVE_PARTS = ("password", "secret", "token", "key", "passwd", "pwd")
_KEY_VALUE_RE = re.compile(
    r"(?i)\b(password|secret|token|api[_-]?key|passwd|pwd)\b\s*[:=]\s*([^\s,;]+)"
)


def is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SENSITIVE_PARTS)


def redact_sensitive_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("***REDACTED***" if is_sensitive_key(key) else redact_sensitive_data(nested))
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item) for item in value)
    return value


def sanitize_error_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "Connection test failed."

    sanitized = _KEY_VALUE_RE.sub(r"\1=***REDACTED***", raw)
    lowered = sanitized.lower()
    if any(part in lowered for part in _SENSITIVE_PARTS):
        return "Sensitive details were removed."

    if "://" in sanitized and "@" in sanitized:
        try:
            parts = urlsplit(sanitized)
            if parts.username or parts.password:
                netloc = parts.hostname or ""
                if parts.port:
                    netloc = f"{netloc}:{parts.port}"
                sanitized = urlunsplit(
                    (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
                )
        except Exception:  # noqa: BLE001
            return "Sensitive details were removed."

    return sanitized
