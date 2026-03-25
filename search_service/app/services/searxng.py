from __future__ import annotations

import asyncio
import re
from html import unescape
from typing import Any

import httpx

from app.config import settings
from app.models import SearchResultItem


class SearXNGClient:
    """Async SearXNG HTTP client."""

    _USER_AGENT = "search-service-bot/1.0"

    def __init__(self) -> None:
        self._base = str(settings.searxng_api_url).rstrip("/")
        self._search_timeout = settings.search_timeout_sec
        self._fetch_timeout = settings.fetch_timeout_sec
        self._min_score = settings.searxng_min_score
        self._max_chars = settings.fetch_max_chars

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        language: str | None = None,
        engines: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query SearXNG and return raw result dicts."""
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "count": max_results,
        }
        if language:
            params["language"] = language
        if engines:
            params["engines"] = engines

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._search_timeout)) as client:
            resp = await client.get(f"{self._base}/search", params=params)
            resp.raise_for_status()
            payload = resp.json()

        results: list[dict[str, Any]] = payload.get("results") or []
        if not isinstance(results, list):
            return []

        # Optional score filter
        if self._min_score > 0:
            results = [
                r for r in results
                if not isinstance(r.get("score"), (int, float)) or float(r["score"]) >= self._min_score
            ]

        return results[:max_results]

    async def fetch_urls(
        self,
        urls: list[str],
        *,
        max_chars_override: int | None = None,
    ) -> dict[str, str | Exception]:
        """
        Fetch multiple URLs in parallel.

        Returns {url: stripped_text} on success, {url: Exception} on failure.
        max_chars_override overrides the service-level SEARCH_FETCH_MAX_CHARS setting.
        """
        if not urls:
            return {}

        effective_max = max_chars_override if max_chars_override is not None else self._max_chars

        async def _one(client: httpx.AsyncClient, url: str) -> tuple[str, str | Exception]:
            try:
                resp = await client.get(url)
                text = self._strip_html(resp.text)
                if effective_max and effective_max > 0:
                    text = text[:effective_max]
                return url, text
            except Exception as exc:  # noqa: BLE001
                return url, exc

        headers = {"User-Agent": self._USER_AGENT}
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self._fetch_timeout),
            follow_redirects=True,
            headers=headers,
        ) as client:
            pairs = await asyncio.gather(*[_one(client, u) for u in urls])

        return dict(pairs)

    def normalize(self, raw: list[dict[str, Any]]) -> list[SearchResultItem]:
        """Convert raw SearXNG dicts to typed SearchResultItem list."""
        items: list[SearchResultItem] = []
        for r in raw:
            title = (r.get("title") or r.get("name") or r.get("url") or "").strip()
            url = (r.get("url") or r.get("link") or "").strip()
            if not title:
                continue
            items.append(
                SearchResultItem(
                    title=title,
                    url=url,
                    snippet=(r.get("content") or r.get("snippet") or "").strip() or None,
                    source_name=(r.get("engine") or r.get("source") or "").strip() or None,
                    published_at=(r.get("publishedDate") or r.get("published") or "").strip() or None,
                )
            )
        return items

    @staticmethod
    def _strip_html(html: str) -> str:
        if not html:
            return ""
        text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
        text = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        return re.sub(r"\s+", " ", text).strip()
