from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    searxng_api_url: str = Field(default="http://localhost:8080", alias="SEARXNG_API_URL")
    searxng_min_score: float = Field(default=0.0, alias="SEARXNG_MIN_SCORE")
    search_timeout_sec: float = Field(default=20.0, alias="SEARCH_TIMEOUT_SEC")
    fetch_timeout_sec: float = Field(default=15.0, alias="FETCH_TIMEOUT_SEC")
    fetch_max_chars: int = Field(default=10000, alias="SEARCH_FETCH_MAX_CHARS")

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


settings = Settings()
