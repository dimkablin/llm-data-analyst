"""Tests for the rag_tool integration.

RAG is no longer a keyword-based dispatch shortcut — it is a regular
LangChain tool (RagTool) that the agent decides to call based on the
semantics of the user's request.

These tests verify:
  - _quick_route never returns "rag" (the shortcut is removed)
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
    from backend.tools.impl.rag_tool import RagTool
    from backend.tools.registry import ToolRegistry
except Exception as exc:  # pragma: no cover
    AgentRunner = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment]
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

    def test_quick_route_never_returns_rag(self) -> None:
        """RAG shortcut is removed — _quick_route must not return 'rag'."""
        runner = self._build_runner(allowed_tool_keys={"rag_tool"})

        for prompt in (
            "что сказано в документации про auth flow?",
            "найди в базе знаний информацию",
            "rag search for billing",
            "в документации написано что",
        ):
            route = runner._quick_route(prompt)
            self.assertNotEqual(route, "rag", f"Unexpected 'rag' route for: {prompt!r}")

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


def _make_ctx(*, allowed: set[str] | None):
    from dataclasses import replace as _replace
    from backend.tools.context import ToolBuildContext
    from backend.core import Settings
    return ToolBuildContext(settings=Settings(), allowed_tool_keys=allowed)


if __name__ == "__main__":
    unittest.main()
