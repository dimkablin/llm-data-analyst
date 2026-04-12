"""RAG tool — knowledge-base retrieval via an external RAG service.

The agent calls this tool when the user's question requires information
from the internal knowledge base (documentation, policies, domain-specific
facts).  The tool forwards the query to the configured RAG backend and
returns the answer together with source references.

Unlike sandbox-execution tools, this is a plain ``BaseTool`` — no code
is generated or run, just an HTTP call to the RAG service.
"""
from __future__ import annotations

from typing import Any

import anyio
from langchain_core.tools import BaseTool

from backend.integrations.rag import RAGService

_DESCRIPTION = """\
Search the internal knowledge base for documented information: processes,
policies, domain knowledge, product descriptions, or any topic covered
by the organisation's documentation.

Use this tool when the user asks a question that requires factual
information from documentation rather than computation over data.

Input: a clear search query in natural language.
Output: an answer from the knowledge base, optionally with source links.

Examples:
  rag_tool("How does the monthly billing reconciliation work?")
  rag_tool("What are the SLA requirements for priority incidents?")
"""


class RagTool(BaseTool):
    """Query an external RAG service and return the knowledge-base answer."""

    name: str = "rag_tool"
    description: str = _DESCRIPTION

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, rag_service: RAGService, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_rag_service", rag_service)

    def _run(self, query: str, *args: Any, **_kwargs: Any) -> str:
        query = query.strip()
        if not query:
            return "Query must not be empty."

        svc: RAGService = object.__getattribute__(self, "_rag_service")
        try:
            result = svc.retrieve(query=query)
        except Exception as exc:
            return f"Knowledge base unavailable: {exc}"

        answer = (result.answer or "").strip()
        if not answer:
            return "The knowledge base returned no context for this query."

        return answer

    async def _arun(self, query: str, *args: Any, **_kwargs: Any) -> str:
        return await anyio.to_thread.run_sync(self._run, query)
