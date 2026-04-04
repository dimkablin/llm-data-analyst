"""Routing tests for AgentRunner._quick_route."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from backend.core import Settings


def _make_settings() -> Settings:
    settings = MagicMock(spec=Settings)
    settings.agent_analysis_depth = "light"
    settings.agent_history_max_messages = 6
    settings.agent_prompt_head_rows = 5
    settings.tool_exec_timeout_sec = 30
    settings.tool_cache_size = 32
    return settings


def _make_runner():
    from backend.agent import AgentRunner

    with patch.object(AgentRunner, "_build_query_graph", return_value=MagicMock()):
        return AgentRunner(_make_settings())


class QuickRouteSummaryTests(unittest.TestCase):
    def test_rezyumiruiy_routes_to_summary(self) -> None:
        self.assertEqual(
            _make_runner()._quick_route("резюмируй итоги", has_rag=False), "summary"
        )

    def test_executive_summary_routes_to_summary(self) -> None:
        self.assertEqual(
            _make_runner()._quick_route("executive summary please", has_rag=False), "summary"
        )

    def test_upravlencheskaya_zapiska_routes_to_summary(self) -> None:
        self.assertEqual(
            _make_runner()._quick_route("сделай управленческую записку", has_rag=False),
            "summary",
        )

    def test_podvedi_itog_routes_to_summary(self) -> None:
        self.assertEqual(
            _make_runner()._quick_route("подведи итог встречи", has_rag=False), "summary"
        )


class QuickRouteRAGTests(unittest.TestCase):
    def test_rag_keyword_routes_to_rag_when_rag_enabled(self) -> None:
        self.assertEqual(
            _make_runner()._quick_route("rag auth flow", has_rag=True), "rag"
        )

    def test_documentation_phrase_routes_to_rag(self) -> None:
        self.assertEqual(
            _make_runner()._quick_route("что сказано в документации", has_rag=True), "rag"
        )

    def test_knowledge_base_phrase_routes_to_rag(self) -> None:
        self.assertEqual(
            _make_runner()._quick_route("что есть в базе знаний", has_rag=True), "rag"
        )

    def test_rag_keyword_returns_none_when_rag_disabled(self) -> None:
        self.assertIsNone(
            _make_runner()._quick_route("rag auth flow", has_rag=False)
        )


class QuickRoutePassThroughTests(unittest.TestCase):
    def test_greeting_returns_none(self) -> None:
        self.assertIsNone(_make_runner()._quick_route("привет", has_rag=False))

    def test_analytical_prompt_returns_none(self) -> None:
        self.assertIsNone(_make_runner()._quick_route("посчитай продажи", has_rag=False))

    def test_empty_prompt_returns_none(self) -> None:
        self.assertIsNone(_make_runner()._quick_route("", has_rag=False))

    def test_generic_question_returns_none(self) -> None:
        self.assertIsNone(
            _make_runner()._quick_route("what is linear regression?", has_rag=False)
        )


class UserMemoryInjectionInRouterTests(unittest.TestCase):
    def test_memory_block_is_included_in_think_prompt(self) -> None:
        from backend.auth import UserMemory

        runner = _make_runner()
        runner.user_memory = UserMemory(profile="Analyst", notes="- Uses bar charts")
        prompt = runner._think_system_prompt()
        self.assertIn("User memory", prompt)
        self.assertIn("Analyst", prompt)
        self.assertIn("Uses bar charts", prompt)

    def test_empty_memory_block_is_omitted_from_think_prompt(self) -> None:
        from backend.auth import UserMemory

        runner = _make_runner()
        runner.user_memory = UserMemory(profile="", notes="")
        prompt = runner._think_system_prompt()
        self.assertNotIn("User memory", prompt)

    def test_memory_block_is_included_in_built_messages(self) -> None:
        from langchain_core.messages import SystemMessage

        from backend.auth import UserMemory

        runner = _make_runner()
        runner.user_memory = UserMemory(profile="Data lead", notes="")
        messages = runner._build_messages("hello", [], use_history=False)
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        self.assertTrue(
            any("Data lead" in str(m.content) for m in system_messages),
            msg="Memory block should be injected as a system message",
        )


if __name__ == "__main__":
    unittest.main()
