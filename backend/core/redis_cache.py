from __future__ import annotations

import logging
import os
import pickle
from typing import Any

logger = logging.getLogger(__name__)

_client: Any | None = None
_warned = False


def _redis_url() -> str:
    return str(os.getenv("REDIS_URL") or "").strip()


def _get_client() -> Any | None:
    global _client, _warned
    if _client is not None:
        return _client
    url = _redis_url()
    if not url:
        return None
    try:
        import redis

        _client = redis.Redis.from_url(url, socket_connect_timeout=0.2, socket_timeout=0.5)
        _client.ping()
        return _client
    except Exception as exc:  # noqa: BLE001 - cache must never break requests
        if not _warned:
            logger.warning("Redis cache unavailable, using in-process cache: %s", exc)
            _warned = True
        return None


def get_pickle(key: str) -> Any | None:
    client = _get_client()
    if client is None:
        return None
    try:
        raw = client.get(key)
        if raw is None:
            logger.debug("redis cache miss key=%s", key)
            return None
        logger.debug("redis cache hit key=%s", key)
        return pickle.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.debug("redis cache get failed key=%s error=%s", key, exc)
        return None


def set_pickle(key: str, value: Any, *, ttl_sec: int) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.setex(key, max(1, int(ttl_sec)), pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        logger.debug("redis cache set key=%s ttl=%s", key, ttl_sec)
    except Exception as exc:  # noqa: BLE001
        logger.debug("redis cache set failed key=%s error=%s", key, exc)


def delete(key: str) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.delete(key)
    except Exception as exc:  # noqa: BLE001
        logger.debug("redis cache delete failed key=%s error=%s", key, exc)


def delete_prefix(prefix: str) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        # ponytail: SCAN is enough here; move to per-namespace index if key counts get large.
        for key in client.scan_iter(f"{prefix}*"):
            client.delete(key)
    except Exception as exc:  # noqa: BLE001
        logger.debug("redis cache delete_prefix failed prefix=%s error=%s", prefix, exc)
