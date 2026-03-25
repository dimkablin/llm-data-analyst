from __future__ import annotations

from fastapi import APIRouter

from app.models import FetchRequest, FetchResponse, FetchedPage
from app.services.searxng import SearXNGClient

router = APIRouter()
_client = SearXNGClient()


@router.post("/fetch/", response_model=FetchResponse, summary="Fetch web pages")
async def fetch_pages(request: FetchRequest) -> FetchResponse:
    """
    Fetch the text content of one or more URLs.

    - Strips HTML, scripts, styles.
    - Returns up to `max_chars` characters per page.
    - Errors per URL are reported individually — the endpoint itself never returns 5xx for fetch failures.
    """
    pages: list[FetchedPage] = []

    if not request.urls:
        return FetchResponse(pages=pages)

    max_chars = request.max_chars if request.max_chars and request.max_chars > 0 else None
    fetched = await _client.fetch_urls(request.urls, max_chars_override=max_chars)

    for url in request.urls:
        result = fetched.get(url)
        if result is None:
            pages.append(FetchedPage(url=url, content="", status="error", error="not fetched"))
        elif isinstance(result, Exception):
            pages.append(FetchedPage(url=url, content="", status="error", error=str(result)))
        else:
            pages.append(FetchedPage(url=url, content=result, status="ok"))

    return FetchResponse(pages=pages)
