"""Focused routing tests for AgentRunner."""
from __future__ import annotations

import unittest
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd

from backend.core import Settings

if TYPE_CHECKING:
    from backend.agent import AgentRunner


def _make_settings() -> Settings:
    settings = MagicMock(spec=Settings)
    settings.agent_analysis_depth = "light"
    settings.agent_history_max_messages = 6
    settings.agent_prompt_head_rows = 5
    settings.tool_exec_timeout_sec = 30
    settings.tool_cache_size = 32
    return settings


def _make_runner(
    *,
    search_enabled: bool = False,
    rag_enabled: bool = False,
) -> AgentRunner:
    from backend.agent import AgentRunner
    from backend.integrations import RAGService, SearchIntegrationConfig, SearchIntegrationService

    kwargs: dict[str, object] = {}
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

    with patch.object(AgentRunner, "_build_query_graph", return_value=MagicMock()):
        return AgentRunner(_make_settings(), **kwargs)


class RouteIntentTier1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = _make_runner()

    def test_empty_prompt_routes_to_chat(self) -> None:
        self.assertEqual(self.runner._route_intent(None, ""), "chat")

    def test_whitespace_only_prompt_routes_to_chat(self) -> None:
        self.assertEqual(self.runner._route_intent(None, "   \n  "), "chat")

    def test_greeting_without_data_routes_to_chat(self) -> None:
        self.assertEqual(self.runner._route_intent(None, "привет"), "chat")

    def test_any_non_greeting_prompt_with_dataframe_routes_to_analysis(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        self.assertEqual(self.runner._route_intent(df, "покажи данные"), "analysis")

    def test_greeting_with_dataframe_stays_chat(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        self.assertEqual(self.runner._route_intent(df, "привет"), "chat")

    def test_db_source_routes_to_analysis_even_without_dataframe(self) -> None:
        session_source = {
            "source_type": "db_connection",
            "source_ref_id": "conn-1",
            "source_label": "Prod DB",
            "source_mode": "live",
        }
        self.assertEqual(
            self.runner._route_intent(None, "сколько заказов?", session_source),
            "analysis",
        )

    def test_russian_analytical_hint_routes_to_analysis(self) -> None:
        self.assertEqual(self.runner._route_intent(None, "посчитай среднее"), "analysis")

    def test_english_plot_hint_routes_to_analysis(self) -> None:
        self.assertEqual(self.runner._route_intent(None, "plot sales"), "analysis")

    def test_no_tools_and_no_fast_path_hint_routes_to_chat(self) -> None:
        with patch.object(self.runner, "_classify_intent") as classify_mock:
            route = self.runner._route_intent(None, "what is linear regression?")
        classify_mock.assert_not_called()
        self.assertEqual(route, "chat")


class RouteIntentRAGTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = _make_runner(rag_enabled=True)

    def test_explicit_rag_keyword_routes_to_rag(self) -> None:
        self.assertEqual(self.runner._route_intent(None, "rag auth flow"), "rag")

    def test_documentation_phrase_routes_to_rag(self) -> None:
        self.assertEqual(
            self.runner._route_intent(None, "что сказано в документации"),
            "rag",
        )

    def test_knowledge_base_phrase_routes_to_rag(self) -> None:
        self.assertEqual(
            self.runner._route_intent(None, "что есть в базе знаний"),
            "rag",
        )

    def test_rag_phrase_keeps_priority_even_with_dataframe(self) -> None:
        df = pd.DataFrame({"x": [1]})
        self.assertEqual(self.runner._route_intent(df, "rag auth flow"), "rag")


class RouteIntentSearchHintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = _make_runner(search_enabled=True)

    def test_search_hint_routes_to_analysis_when_search_tool_is_available(self) -> None:
        self.assertEqual(self.runner._route_intent(None, "latest news"), "analysis")

    def test_search_hint_does_not_trigger_without_search_service(self) -> None:
        runner_no_search = _make_runner()
        with patch.object(runner_no_search, "_classify_intent") as classify_mock:
            route = runner_no_search._route_intent(None, "search")
        classify_mock.assert_not_called()
        self.assertEqual(route, "chat")


class RouteIntentTier2LLMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = _make_runner(search_enabled=True)

    def test_ambiguous_prompt_uses_llm_classifier_when_tools_exist(self) -> None:
        with patch.object(self.runner, "_classify_intent", return_value="analysis") as classify_mock:
            route = self.runner._route_intent(None, "tell me about the key rate in Russia")
        classify_mock.assert_called_once()
        self.assertEqual(route, "analysis")

    def test_classifier_chat_result_is_respected(self) -> None:
        with patch.object(self.runner, "_classify_intent", return_value="chat"):
            route = self.runner._route_intent(None, "how are you?")
        self.assertEqual(route, "chat")

    def test_classifier_rag_result_is_respected(self) -> None:
        runner = _make_runner(search_enabled=True, rag_enabled=True)
        with patch.object(runner, "_classify_intent", return_value="rag"):
            route = runner._route_intent(None, "some ambiguous query")
        self.assertEqual(route, "rag")


class ClassifyIntentLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = _make_runner(search_enabled=True)

    def _call_classifier(self, llm_reply: str) -> str:
        fake_result = MagicMock()
        fake_result.content = llm_reply
        with patch.object(self.runner, "_build_llm") as build_mock:
            llm = MagicMock()
            llm.invoke.return_value = fake_result
            build_mock.return_value = llm
            return self.runner._classify_intent(
                "test prompt",
                has_search=True,
                has_rag=False,
            )

    def test_classifier_accepts_analysis_reply(self) -> None:
        self.assertEqual(self._call_classifier("analysis"), "analysis")

    def test_classifier_accepts_chat_reply(self) -> None:
        self.assertEqual(self._call_classifier("chat"), "chat")

    def test_classifier_accepts_rag_reply(self) -> None:
        self.assertEqual(self._call_classifier("rag"), "rag")

    def test_classifier_ignores_reply_casing(self) -> None:
        self.assertEqual(self._call_classifier("ANALYSIS"), "analysis")

    def test_classifier_trims_whitespace(self) -> None:
        self.assertEqual(self._call_classifier("  analysis  "), "analysis")

    def test_classifier_falls_back_to_analysis_on_llm_exception(self) -> None:
        with patch.object(self.runner, "_build_llm", side_effect=RuntimeError("oops")):
            result = self.runner._classify_intent("test", has_search=True, has_rag=False)
        self.assertEqual(result, "analysis")

    def test_classifier_defaults_to_chat_for_unknown_reply(self) -> None:
        self.assertEqual(self._call_classifier("banana"), "chat")


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
        from backend.auth import UserMemory
        from langchain_core.messages import SystemMessage

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
