from __future__ import annotations

from pydantic import BaseModel, Field

# ── Search ────────────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    max_search_results: int = Field(default=5, ge=1, le=30)
    fetch_top_n: int = Field(default=3, ge=0, le=10, description="Number of top URLs to fetch for full text")
    language: str | None = Field(default=None, description="Language code (e.g. ru, en)")
    engines: str | None = Field(default=None, description="Comma-separated SearXNG engines")


class SearchResultItem(BaseModel):
    title: str
    url: str
    snippet: str | None = None
    source_name: str | None = None
    published_at: str | None = None


class SearchResponse(BaseModel):
    query: str
    answer: str | None = None  # Always null — summarization is the caller's job
    results: list[SearchResultItem]
    sources: list[str]


# ── Fetch ─────────────────────────────────────────────────────────────────────


class FetchRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, description="List of URLs to fetch")
    max_chars: int | None = Field(
        default=None,
        ge=0,
        description="Max characters per page (0 or null = use service default)",
    )


class FetchedPage(BaseModel):
    url: str
    content: str
    status: str = Field(description="'ok' or 'error'")
    error: str | None = None


class FetchResponse(BaseModel):
    pages: list[FetchedPage]
