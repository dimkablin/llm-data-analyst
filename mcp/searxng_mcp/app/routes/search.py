from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models import SearchRequest, SearchResponse
from app.services.searxng import SearXNGClient

router = APIRouter()
_client = SearXNGClient()


@router.post("/search/", response_model=SearchResponse, summary="Quick web search")
async def quick_search(request: SearchRequest) -> SearchResponse:
    """
    Bridge to SearXNG.

    - Queries SearXNG with the given parameters.
    - Optionally fetches full text of top-N URLs.
    - Returns raw results; LLM summarization is the **caller's** responsibility.
    """
    try:
        raw = await _client.search(
            request.query,
            max_results=request.max_search_results,
            language=request.language,
            engines=request.engines,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"SearXNG unavailable: {exc}") from exc

    results = _client.normalize(raw)

    if request.fetch_top_n > 0 and results:
        top_urls = [r.url for r in results[: request.fetch_top_n] if r.url]
        fetched = await _client.fetch_urls(top_urls)

        # Enrich thin snippets with fetched page text (skip failed fetches)
        for item in results:
            result = fetched.get(item.url)
            if isinstance(result, str) and result:
                if not item.snippet or len(item.snippet) < 120:
                    item.snippet = result[:600] if len(result) > 600 else result

    sources = list(dict.fromkeys(r.url for r in results if r.url))

    return SearchResponse(
        query=request.query,
        answer=None,
        results=results,
        sources=sources,
    )
