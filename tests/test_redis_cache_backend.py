from __future__ import annotations

from backend.core import redis_cache
from backend.data_access import session_catalog_cache


class _FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> bytes | None:
        return self.data.get(key)

    def setex(self, key: str, ttl: int, value: bytes) -> None:
        _ = ttl
        self.data[key] = value

    def delete(self, key: str) -> None:
        self.data.pop(key, None)

    def scan_iter(self, pattern: str):
        prefix = pattern.removesuffix("*")
        yield from [key for key in self.data if key.startswith(prefix)]


def test_redis_cache_pickle_roundtrip_without_real_redis(monkeypatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(redis_cache, "_client", fake)

    redis_cache.set_pickle("test:key", {"ok": [1]}, ttl_sec=60)

    assert redis_cache.get_pickle("test:key") == {"ok": [1]}

    redis_cache.delete_prefix("test:")

    assert redis_cache.get_pickle("test:key") is None


def test_session_catalog_cache_uses_redis_after_local_invalidation(monkeypatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(redis_cache, "_client", fake)
    session_catalog_cache.invalidate_candidates_cache()
    calls = 0

    def builder() -> list[dict[str, str]]:
        nonlocal calls
        calls += 1
        return [{"table": "sales"}]

    assert session_catalog_cache.get_or_build_candidates("s1", builder) == [{"table": "sales"}]
    session_catalog_cache._candidates_cache.clear()  # noqa: SLF001 - force Redis path

    assert session_catalog_cache.get_or_build_candidates("s1", builder) == [{"table": "sales"}]
    assert calls == 1
