from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes.fetch import router as fetch_router
from app.routes.search import router as search_router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


app = FastAPI(
    title="Search Service",
    description="Lightweight SearXNG bridge for llm-data-analyst. "
                "Handles raw search — LLM summarization stays in the caller.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router, prefix="/api/v1")
app.include_router(fetch_router, prefix="/api/v1")


@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Health check. Also pings SearXNG to confirm it is reachable."""
    searxng_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{str(settings.searxng_api_url).rstrip('/')}/healthz")
            searxng_ok = resp.status_code == 200
    except Exception:
        pass

    return {
        "status": "healthy" if searxng_ok else "degraded",
        "searxng": "up" if searxng_ok else "down",
        "searxng_url": str(settings.searxng_api_url),
    }


@app.get("/", tags=["ops"])
async def root() -> dict:
    return {"service": "search-service", "version": "1.0.0", "docs": "/docs"}
