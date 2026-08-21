"""Tests for the rag_tool integration.

RAG is no longer a keyword-based dispatch shortcut — it is a regular
LangChain tool (RagTool) that the agent decides to call based on the
semantics of the user's request.

These tests verify:
  - AgentRunner has no keyword quick route shortcut
  - RagTool is registered and available when rag_service is enabled
  - RagTool.is_available respects per-user tool permissions
"""

from __future__ import annotations

import unittest
from dataclasses import replace

_IMPORT_ERROR: Exception | None = None

try:
    from backend.agent import AgentRunner
    from backend.core import Settings
    from backend.tools.catalog import ALL_TOOL_SPECS
    from backend.tools.impl.rag_tool import RagTool
    from backend.tools.registry import ToolRegistry
except Exception as exc:  # pragma: no cover
    AgentRunner = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment]
    ALL_TOOL_SPECS = ()  # type: ignore[assignment]
    RagTool = None  # type: ignore[assignment]
    ToolRegistry = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


class _FakeRAGService:
    is_enabled = True

    def search(self, *, query: str, include_references: bool = False):
        from backend.integrations.rag import RAGQueryResult

        return RAGQueryResult(
            query=query,
            answer="Use access token from login endpoint.",
            references=["https://docs.example/auth"],
            warnings=[],
            request_params={},
            raw_payload={},
        )

    def retrieve(self, *, query: str, include_references: bool = False):
        return self.search(query=query, include_references=include_references)


@unittest.skipIf(
    AgentRunner is None or Settings is None,
    f"Agent runtime dependencies unavailable: {_IMPORT_ERROR}",
)
class RagToolTests(unittest.TestCase):
    def _build_runner(self, *, allowed_tool_keys: set[str] | None) -> AgentRunner:
        settings = replace(
            Settings(),
            agent_cache_enabled=False,
            llm_warmup_enabled=False,
        )
        return AgentRunner(
            settings,
            rag_service=_FakeRAGService(),
            allowed_tool_keys=allowed_tool_keys,
        )

    def test_rag_route_is_not_keyword_routed(self) -> None:
        """RAG must stay a regular tool, not a keyword dispatch shortcut."""
        self.assertFalse(hasattr(AgentRunner, "_quick_route"))

    def test_rag_tool_registered_when_service_enabled(self) -> None:
        """ToolRegistry must include RagToolFactory when rag_service is provided."""
        registry = ToolRegistry.from_services(rag_service=_FakeRAGService())
        self.assertTrue(registry.is_available("rag_tool", _make_ctx(allowed=None)))

    def test_rag_tool_unavailable_when_user_disables_it(self) -> None:
        """RagToolFactory must respect per-user allowed_tool_keys."""
        registry = ToolRegistry.from_services(rag_service=_FakeRAGService())
        self.assertFalse(registry.is_available("rag_tool", _make_ctx(allowed=set())))

    def test_rag_tool_unavailable_when_service_disabled(self) -> None:
        """RagToolFactory must not expose the tool when the service is off."""

        class _DisabledService:
            is_enabled = False

        registry = ToolRegistry.from_services(rag_service=_DisabledService())
        self.assertFalse(registry.is_available("rag_tool", _make_ctx(allowed=None)))

    def test_rag_tool_run_returns_answer(self) -> None:
        """RagTool._run must return the answer from the service."""
        tool = RagTool(rag_service=_FakeRAGService())
        result = tool._run("how does auth work?")
        self.assertIn("access token", result)
        self.assertIn("login endpoint", result)

    def test_rag_tool_run_includes_sources(self) -> None:
        """RagTool._run must append references when the service returns them."""
        tool = RagTool(rag_service=_FakeRAGService())
        result = tool._run("auth flow?")
        self.assertIn("Sources:", result)
        self.assertIn("docs.example/auth", result)
        self.assertIn("knowledge-base answer below is synthesized evidence", result)
        self.assertIn("not verbatim source passages", result)
        self.assertTrue(result.startswith("Grounding constraint:"))
        self.assertIn("run a complementary query", result)
        self.assertIn("grounded partial list", result)

    def test_rag_tool_propagates_backend_errors_to_the_tool_loop(self) -> None:
        """Infrastructure failures must be observable as tool errors."""

        class _UnavailableService(_FakeRAGService):
            def search(self, *, query: str, include_references: bool = False):
                raise RuntimeError("network unavailable")

        tool = RagTool(rag_service=_UnavailableService())
        with self.assertRaisesRegex(RuntimeError, "network unavailable"):
            tool._run("auth flow?")

    def test_rag_tool_run_without_context_refuses_to_invent_definition(self) -> None:
        """RagTool should return an explicit non-answer when no KB context exists."""

        class _NoContextService(_FakeRAGService):
            def search(self, *, query: str, include_references: bool = False):
                from backend.integrations.rag import RAGQueryResult

                return RAGQueryResult(
                    query=query,
                    answer="",
                    references=[],
                    warnings=[],
                    request_params={},
                    raw_payload={},
                )

        tool = RagTool(rag_service=_NoContextService())
        result = tool._run("unknown term abc")
        self.assertIn("Do not invent a definition", result)
        self.assertIn("Ask for clarification", result)

    def test_rag_tool_run_with_no_context_warning_refuses_to_invent_definition(self) -> None:
        """Warnings from retrieval should also trigger no-context clarification."""

        class _NoContextWithWarningService(_FakeRAGService):
            def search(self, *, query: str, include_references: bool = False):
                from backend.integrations.rag import RAGQueryResult

                return RAGQueryResult(
                    query=query,
                    answer="",
                    references=[],
                    warnings=["RAG backend returned no context chunks."],
                    request_params={},
                    raw_payload={},
                )

        tool = RagTool(rag_service=_NoContextWithWarningService())
        result = tool._run("undefined metric")
        self.assertIn("No relevant knowledge-base context", result)
        self.assertIn("Ask for clarification", result)

    def test_rag_tool_description_scopes_retrieval_to_indexed_knowledge_base(self) -> None:
        """Tool descriptions must make the RAG source boundary explicit."""
        tool = RagTool(rag_service=_FakeRAGService())
        description = tool.description.lower()

        self.assertIn("configured indexed knowledge base", description)
        self.assertIn("synthesized grounded answer", description)
        self.assertIn("explicitly named in that answer", description)

        rag_spec = next(spec for spec in ALL_TOOL_SPECS if spec.tool_key == "rag_tool")
        self.assertIn("indexed knowledge base", rag_spec.description.lower())
        self.assertIn("индексированной базе знаний", rag_spec.description_ru.lower())


def _make_ctx(*, allowed: set[str] | None):
    from backend.core import Settings
    from backend.tools.context import ToolBuildContext

    return ToolBuildContext(settings=Settings(), allowed_tool_keys=allowed)
