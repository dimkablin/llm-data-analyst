"""In-process cache of SQL table candidates per session (avoids repeated DB introspection)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from backend.core import redis_cache

_CACHE_TTL_SEC = 3600.0
_CACHE_PREFIX = "catalog:candidates:"
_lock = threading.Lock()
_candidates_cache: dict[str, tuple[float, list[Any]]] = {}


def get_or_build_candidates(
    cache_key: str,
    builder: Callable[[], list[Any]],
) -> list[Any]:
    key = str(cache_key or "").strip()
    if not key:
        return builder()

    now = time.monotonic()
    with _lock:
        entry = _candidates_cache.get(key)
        if entry is not None:
            ts, candidates = entry
            if now - ts <= _CACHE_TTL_SEC:
                return candidates
    cached = redis_cache.get_pickle(f"{_CACHE_PREFIX}{key}")
    if isinstance(cached, list):
        with _lock:
            _candidates_cache[key] = (now, cached)
        return cached

    built = builder()
    with _lock:
        _candidates_cache[key] = (now, built)
    redis_cache.set_pickle(f"{_CACHE_PREFIX}{key}", built, ttl_sec=int(_CACHE_TTL_SEC))
    return built


def invalidate_candidates_cache(cache_key: str | None = None) -> None:
    with _lock:
        if cache_key is None:
            _candidates_cache.clear()
            redis_cache.delete_prefix(_CACHE_PREFIX)
            return
        key = str(cache_key or "").strip()
        _candidates_cache.pop(key, None)
        redis_cache.delete(f"{_CACHE_PREFIX}{key}")
