"""Tests for the two-tier intent router in AgentRunner.

We test _route_intent + _classify_intent without spinning up a real LLM:
Tier-1 fast-path rules are fully deterministic and tested here.
Tier-2 (_classify_intent LLM fallback) is patched to return a controlled value.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from backend.config import Settings


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_settings() -> Settings:
    s = MagicMock(spec=Settings)
    s.agent_analysis_depth = "light"
    s.agent_history_max_messages = 6
    s.agent_prompt_head_rows = 5
    s.tool_exec_timeout_sec = 30
    s.tool_cache_size = 32
    return s


def _make_runner(
    *,
    search_enabled: bool = False,
    rag_enabled: bool = False,
) -> "AgentRunner":
    from backend.agent_runner import AgentRunner
    from backend.search_integration import SearchIntegrationService, SearchIntegrationConfig
    from backend.rag_service import RAGService

    settings = _make_settings()
    kwargs: dict = {}

    if search_enabled:
            cfg = SearchIntegrationConfig(
                enabled=True,
                base_url="http://search",
                search_endpoint="/api/v1/search/",
                fetch_endpoint="/api/v1/fetch/",
                timeout_sec=5.0,
                max_results_default=5,
                fetch_top_n_default=2,
            )
            kwargs["search_service"] = SearchIntegrationService(cfg)

    if rag_enabled:
        rag_mock = MagicMock(spec=RAGService)
        rag_mock.is_enabled = True
        kwargs["rag_service"] = rag_mock

    # Bypass LangGraph graph build (slow + needs env)
    with patch.object(AgentRunner, "_build_query_graph", return_value=MagicMock()):
        runner = AgentRunner(settings, **kwargs)
    return runner


class RouteIntentTier1Tests(unittest.TestCase):
    """Tier-1 deterministic rules — no LLM invoked."""

    def setUp(self) -> None:
        self.runner = _make_runner()

    # ── Empty / greeting ────────────────────────────────────────────────────

    def test_empty_prompt_returns_chat(self) -> None:
        route = self.runner._route_intent(None, "")
        self.assertEqual(route, "chat")

    def test_whitespace_only_returns_chat(self) -> None:
        route = self.runner._route_intent(None, "   \n  ")
        self.assertEqual(route, "chat")

    def test_greeting_no_data_returns_chat(self) -> None:
        for greeting in ("привет", "здравствуйте", "hello", "hi there"):
            with self.subTest(greeting=greeting):
                route = self.runner._route_intent(None, greeting)
                self.assertEqual(route, "chat")

    # ── Data loaded ─────────────────────────────────────────────────────────

    def test_any_query_with_df_returns_analysis(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        # Greetings with data are allowed to return "chat" — skip them here.
        for prompt in ("покажи данные", "?", "расчёт", "summarize"):
            with self.subTest(prompt=prompt):
                route = self.runner._route_intent(df, prompt)
                self.assertEqual(route, "analysis")

    def test_db_source_without_df_returns_analysis(self) -> None:
        session_source = {
            "source_type": "db_connection",
            "source_ref_id": "conn-1",
            "source_label": "Prod DB",
            "source_mode": "live",
        }
        # A non-greeting query with a DB source should route to analysis.
        route = self.runner._route_intent(None, "сколько заказов?", session_source)
        self.assertEqual(route, "analysis")

    # ── Analytical hints ────────────────────────────────────────────────────

    def test_analytical_hint_returns_analysis(self) -> None:
        for hint in ("построй график", "покажи таблицу", "посчитай среднее", "данных"):
            with self.subTest(hint=hint):
                route = self.runner._route_intent(None, hint)
                self.assertEqual(route, "analysis")

    # ── No tools, general question ────────────────────────────────────────

    def test_no_tools_general_question_returns_chat(self) -> None:
        # No services provided → no tools → classifier never called
        with patch.object(self.runner, "_classify_intent") as mock_cls:
            route = self.runner._route_intent(None, "что такое линейная регрессия?")
        # Without tools, classify_intent should NOT be called
        mock_cls.assert_not_called()
        self.assertEqual(route, "chat")


class RouteIntentRAGTests(unittest.TestCase):
    """RAG routing."""

    def setUp(self) -> None:
        self.runner = _make_runner(rag_enabled=True)

    def test_rag_hint_returns_rag(self) -> None:
        # Only test prompts that actually contain RAG_HINTS keywords
        for prompt in (
            "найди в базе знаний",
            "что есть в базе знаний",
            "в базе знаний",
            "по документации",
        ):
            with self.subTest(prompt=prompt):
                route = self.runner._route_intent(None, prompt)
                self.assertEqual(route, "rag")

    def test_rag_hint_with_df_still_returns_rag(self) -> None:
        df = pd.DataFrame({"x": [1]})
        route = self.runner._route_intent(df, "найди в базе знаний")
        self.assertEqual(route, "rag")


class RouteIntentSearchHintTests(unittest.TestCase):
    """Search keyword hints → analysis (tier-1 fast path)."""

    def setUp(self) -> None:
        self.runner = _make_runner(search_enabled=True)

    def test_search_hint_returns_analysis(self) -> None:
        for prompt in ("найди информацию о Python", "поищи новости", "погугли это"):
            with self.subTest(prompt=prompt):
                route = self.runner._route_intent(None, prompt)
                self.assertEqual(route, "analysis")

    def test_search_hint_not_triggered_without_search_service(self) -> None:
        # Runner with no search service → search hint words have no effect
        runner_no_search = _make_runner()
        with patch.object(runner_no_search, "_classify_intent") as mock_cls:
            route = runner_no_search._route_intent(None, "найди")
        mock_cls.assert_not_called()
        self.assertEqual(route, "chat")


class RouteIntentTier2LLMTests(unittest.TestCase):
    """Tier-2: _classify_intent is called when tier-1 is inconclusive."""

    def setUp(self) -> None:
        self.runner = _make_runner(search_enabled=True)

    def test_ambiguous_no_hint_calls_classify_intent(self) -> None:
        with patch.object(
            self.runner, "_classify_intent", return_value="analysis"
        ) as mock_cls:
            route = self.runner._route_intent(None, "расскажи про ключевую ставку в России")

        mock_cls.assert_called_once()
        self.assertEqual(route, "analysis")

    def test_classify_intent_chat_result_respected(self) -> None:
        with patch.object(self.runner, "_classify_intent", return_value="chat"):
            route = self.runner._route_intent(None, "как дела?")
        self.assertEqual(route, "chat")

    def test_classify_intent_rag_result_respected(self) -> None:
        runner = _make_runner(search_enabled=True, rag_enabled=True)
        with patch.object(runner, "_classify_intent", return_value="rag"):
            route = runner._route_intent(None, "some ambiguous query")
        self.assertEqual(route, "rag")


class ClassifyIntentLogicTests(unittest.TestCase):
    """Test _classify_intent text parsing (LLM response mocked)."""

    def setUp(self) -> None:
        self.runner = _make_runner(search_enabled=True)

    def _call(self, llm_reply: str) -> str:
        fake_result = MagicMock()
        fake_result.content = llm_reply
        with patch.object(self.runner, "_build_llm") as mock_build:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = fake_result
            mock_build.return_value = mock_llm
            return self.runner._classify_intent(
                "test prompt",
                has_search=True,
                has_rag=False,
            )

    def test_analysis_reply(self) -> None:
        self.assertEqual(self._call("analysis"), "analysis")

    def test_chat_reply(self) -> None:
        self.assertEqual(self._call("chat"), "chat")

    def test_rag_reply(self) -> None:
        self.assertEqual(self._call("rag"), "rag")

    def test_uppercased_reply_still_works(self) -> None:
        self.assertEqual(self._call("ANALYSIS"), "analysis")
        self.assertEqual(self._call("CHAT"), "chat")

    def test_extra_whitespace_in_reply(self) -> None:
        self.assertEqual(self._call("  analysis  "), "analysis")

    def test_fallback_on_exception(self) -> None:
        with patch.object(self.runner, "_build_llm", side_effect=RuntimeError("oops")):
            result = self.runner._classify_intent(
                "test", has_search=True, has_rag=False
            )
        self.assertEqual(result, "analysis")

    def test_unknown_reply_defaults_to_chat(self) -> None:
        # LLM returns something weird → "analysis" not in text → "chat"
        self.assertEqual(self._call("banana"), "chat")


class UserMemoryInjectionInRouterTests(unittest.TestCase):
    """Verify that user memory block shows up in system prompts."""

    def test_memory_block_in_think_system_prompt(self) -> None:
        from backend.user_memory import UserMemory

        runner = _make_runner()
        runner.user_memory = UserMemory(profile="Analyst", notes="- Uses bar charts")
        prompt = runner._think_system_prompt()
        self.assertIn("User memory", prompt)
        self.assertIn("Analyst", prompt)
        self.assertIn("Uses bar charts", prompt)

    def test_empty_memory_not_in_prompt(self) -> None:
        from backend.user_memory import UserMemory

        runner = _make_runner()
        runner.user_memory = UserMemory(profile="", notes="")
        prompt = runner._think_system_prompt()
        self.assertNotIn("User memory", prompt)

    def test_memory_block_in_build_messages(self) -> None:
        from backend.user_memory import UserMemory
        from langchain_core.messages import SystemMessage

        runner = _make_runner()
        runner.user_memory = UserMemory(profile="Data lead", notes="")
        messages = runner._build_messages("hello", [], use_history=False)
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        self.assertTrue(
            any("Data lead" in str(m.content) for m in system_msgs),
            msg="Memory block should appear as a SystemMessage",
        )


if __name__ == "__main__":
    unittest.main()
