"""RAG tool — knowledge-base retrieval via an external RAG service.

The agent calls this tool when the user's question requires information
from the internal knowledge base (documentation, policies, domain-specific
facts).  The tool forwards the query to the configured RAG backend and
returns the answer together with source references.

Unlike sandbox-execution tools, this is a plain ``BaseTool`` — no code
is generated or run, just an HTTP call to the RAG service.
"""

from __future__ import annotations

from typing import Any, ClassVar

import anyio
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from backend.integrations.rag import RAGService
from backend.tools.instructions import tool_description


class _RagInput(BaseModel):
    query: str = Field(description="Поисковый запрос на естественном языке для поиска в базе знаний.")


class RagTool(BaseTool):
    """Query an external RAG service and return the knowledge-base answer."""

    name: str = "rag_tool"
    description: str = tool_description("rag_tool")
    args_schema: type[BaseModel] = _RagInput
    parallel_safe: ClassVar[bool] = True

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, rag_service: RAGService, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_rag_service", rag_service)

    def _run(self, query: str, *args: Any, **_kwargs: Any) -> str:
        query = query.strip()
        if not query:
            return "Query must not be empty."

        svc: RAGService = object.__getattribute__(self, "_rag_service")
        result = svc.search(query=query, include_references=True)

        answer = (result.answer or "").strip()
        no_context = not answer or any(
            "no context" in (warning or "").lower()
            or "no chunks" in (warning or "").lower()
            or "no normalized answer" in (warning or "").lower()
            for warning in result.warnings
        )
        if no_context:
            return (
                "No relevant knowledge-base context was found for this query. "
                "Do not invent a definition or explanation. Ask for clarification."
            )

        references = [str(item).strip() for item in result.references if str(item).strip()]
        grounding_note = (
            "Grounding constraint: the knowledge-base answer below is synthesized evidence, "
            "not verbatim source passages. Use only entities and attributes explicitly named "
            "in it. Unless it explicitly establishes completeness, run a complementary query "
            "or return a grounded partial list; never complete it from memory.\n\n"
        )
        if references:
            sources = "\n".join(f"- {reference}" for reference in references)
            return f"{grounding_note}{answer}\n\nSources:\n{sources}"
        return grounding_note + answer

    async def _arun(self, query: str, *args: Any, **_kwargs: Any) -> str:
        return await anyio.to_thread.run_sync(self._run, query)
