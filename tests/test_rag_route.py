from __future__ import annotations

from dataclasses import replace
import unittest

_IMPORT_ERROR: Exception | None = None

try:
    from backend.agent_runner import AgentRunner
    from backend.config import Settings
except Exception as exc:  # pragma: no cover - environment-dependent import gate
    AgentRunner = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


class _FakeRAGService:
    is_enabled = True

    def stream_search(self, *, query: str, include_references: bool = False):
        _ = include_references
        if "auth" not in query:
            return []
        return ["Используй access token ", "из login endpoint."]

    def search(self, *, query: str, include_references: bool = False):
        _ = (query, include_references)
        return None

    def format_for_user(self, result) -> str:
        _ = result
        return "fallback"


@unittest.skipIf(
    AgentRunner is None or Settings is None,
    f"Agent runtime dependencies unavailable: {_IMPORT_ERROR}",
)
class RAGRouteTests(unittest.TestCase):
    def _build_runner(self, *, allowed_tool_keys: set[str] | None) -> AgentRunner:
        settings = replace(
            Settings(),
            agent_cache_enabled=False,
            llm_warmup_enabled=False,
            agent_evaluate_enabled=False,
        )
        return AgentRunner(
            settings,
            rag_service=_FakeRAGService(),
            allowed_tool_keys=allowed_tool_keys,
        )

    def test_route_intent_detects_knowledge_base_request(self) -> None:
        runner = self._build_runner(allowed_tool_keys={"rag_tool"})

        route = runner._route_intent(
            None,
            "Что сказано в документации про auth flow?",
            session_source=None,
        )

        self.assertEqual(route, "rag")

    def test_run_query_uses_rag_route(self) -> None:
        runner = self._build_runner(allowed_tool_keys={"rag_tool"})

        response = runner.run_query(
            None,
            "Что сказано в документации про auth flow?",
            history=[],
            use_history=False,
            include_reasoning=False,
            callbacks=[],
            trace_context={},
            session_source={},
        )

        self.assertEqual(response.route, "rag")
        self.assertEqual(
            response.final_text,
            "Используй access token из login endpoint.",
        )
        self.assertEqual(response.artifacts, [])
        self.assertEqual(response.tool_calls, 0)

    def test_rag_route_respects_user_tool_toggle(self) -> None:
        runner = self._build_runner(allowed_tool_keys=set())

        response = runner.run_query(
            None,
            "Что сказано в документации про auth flow?",
            history=[],
            use_history=False,
            include_reasoning=False,
            callbacks=[],
            trace_context={},
            session_source={},
        )

        self.assertEqual(response.route, "rag")
        self.assertIn("RAG интеграция отключена", response.final_text)


if __name__ == "__main__":
    unittest.main()
