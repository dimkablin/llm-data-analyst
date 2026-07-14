from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from backend.agent.runtime_llm import build_runtime_llm
from backend.agent.services.runtime_context import build_runtime_metadata
from backend.core.config import Settings
from backend.observability.phoenix import record_llm_usage_on_active_span


class ChatTitleRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_name: str | None = None
    user_queries: list[str] = Field(default_factory=list)
    trace_context: dict[str, Any] | None = None


class ChatTitleService(BaseModel):
    """Generate short UI titles for chat sessions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: Settings

    def generate(self, request: ChatTitleRequest) -> str | None:
        cleaned_queries = [
            str(query).strip() for query in request.user_queries if str(query).strip()
        ]
        if not cleaned_queries:
            return None

        prompt_messages = self._build_messages(
            dataset_name=request.dataset_name,
            user_queries=cleaned_queries[-8:],
        )
        runtime_config: dict[str, Any] = {}
        metadata = build_runtime_metadata(request.trace_context)
        if metadata:
            runtime_config["metadata"] = metadata

        try:
            llm = build_runtime_llm(
                self.settings,
                role="chat",
                include_reasoning=False,
                timeout_sec=max(8, min(20, self.settings.backend_query_timeout_sec)),
            )
            response = llm.invoke(prompt_messages, config=runtime_config or None)
            record_llm_usage_on_active_span(
                response,
                fallback_model=self.settings.llm_model,
                fallback_provider=self.settings.llm_provider,
            )
            generated = self._content_to_text(getattr(response, "content", ""))
        except Exception:
            generated = ""

        return self._normalize_title_candidate(generated)

    @staticmethod
    def _build_messages(
        *,
        dataset_name: str | None,
        user_queries: list[str],
    ) -> list[Any]:
        dataset_part = str(dataset_name or "").strip() or "not specified"
        query_lines = "\n".join(
            f"{idx}. {item}" for idx, item in enumerate(user_queries, start=1)
        )
        return [
            SystemMessage(
                content=(
                    "Create a concise chat title for a data analysis session. "
                    "Return only the title, without quotes or explanation. "
                    "Use the user's language when it is clear from the queries. "
                    "The title must be exactly 3 or 4 words."
                )
            ),
            HumanMessage(
                content=(
                    f"Dataset: {dataset_part}\n"
                    "User queries:\n"
                    f"{query_lines}\n\n"
                    "Generate the chat title."
                )
            ),
        ]

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content or "")

    @staticmethod
    def _normalize_title_candidate(raw: str) -> str | None:
        text = str(raw or "").strip()
        if not text:
            return None
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if not first_line:
            return None
        first_line = first_line.strip("`\"'.,:;!?- ")
        words = re.findall(r"[\w]+(?:-[\w]+)?", first_line, flags=re.UNICODE)
        if len(words) < 3:
            return None
        title = " ".join(words[:4]).strip()
        if not title:
            return None
        return f"{title[0].upper()}{title[1:]}" if len(title) > 1 else title.upper()
