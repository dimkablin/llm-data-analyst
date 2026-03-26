from __future__ import annotations

from dataclasses import replace
import unittest

from backend.agent_runner import AgentResponse, AgentRunner
from backend.config import Settings
from backend.tool_policy import supports_artifact_optional_output


class MessageBackedToolTests(unittest.TestCase):
    def _build_runner(self) -> AgentRunner:
        settings = replace(
            Settings(),
            agent_cache_enabled=False,
            llm_warmup_enabled=False,
            agent_evaluate_enabled=False,
        )
        return AgentRunner(settings)

    def test_artifact_optional_policy_matches_external_message_tools(self) -> None:
        self.assertTrue(supports_artifact_optional_output(["search_tool"]))
        self.assertTrue(supports_artifact_optional_output(["search_tool", "memory"]))
        self.assertFalse(supports_artifact_optional_output(["pandas_tool"]))
        self.assertFalse(
            supports_artifact_optional_output(["search_tool", "pandas_tool"])
        )

    def test_evaluate_accepts_search_without_artifacts(self) -> None:
        runner = self._build_runner()
        response = AgentResponse(
            final_text="Краткий вывод по внешнему исследованию.",
            reasoning="Research completed.",
            artifacts=[],
            route="analysis",
            tool_calls=1,
            tool_names=["search_tool"],
        )

        result = runner._evaluate_node(
            {
                "response": response,
                "callbacks": [],
                "prompt": "Найди в интернете факты по теме",
                "step_index": 1,
                "max_steps": 2,
                "capability_context": {},
            }
        )

        self.assertTrue(result["eval_passed"])

    def test_decide_treats_message_backed_output_as_ready(self) -> None:
        runner = self._build_runner()
        response = AgentResponse(
            final_text="Нашел и собрал внешний research-ответ.",
            reasoning="Research completed.",
            artifacts=[],
            route="analysis",
            tool_calls=1,
            tool_names=["search_tool"],
        )

        result = runner._decide_node(
            {
                "response": response,
                "callbacks": [],
                "eval_passed": True,
                "eval_reason": "pre-check passed",
                "step_index": 1,
                "max_steps": 2,
            }
        )

        self.assertTrue(result["done"])
        self.assertEqual(result["stop_reason"], "ready")

    def test_finalize_keeps_message_backed_output_without_artifacts(self) -> None:
        runner = self._build_runner()
        response = AgentResponse(
            final_text="Вот текстовый итог поиска без табличного артефакта.",
            reasoning="Search completed.",
            artifacts=[],
            route="analysis",
            tool_calls=1,
            tool_names=["search_tool"],
        )

        result = runner._finalize_node(
            {
                "response": response,
                "callbacks": [],
                "route": "analysis",
                "prompt": "Найди в сети краткую справку по теме",
                "df": None,
                "stop_reason": "ready",
                "eval_passed": True,
                "step_index": 1,
                "max_steps": 2,
            }
        )

        final_response = result["response"]
        self.assertEqual(
            final_response.final_text,
            "Вот текстовый итог поиска без табличного артефакта.",
        )
        self.assertEqual(final_response.tool_names, ["search_tool"])
        self.assertEqual(final_response.artifacts, [])

    def test_finalize_still_rejects_artifact_first_tool_without_artifacts(self) -> None:
        runner = self._build_runner()
        response = AgentResponse(
            final_text="Я якобы построил график.",
            reasoning="Chart attempted.",
            artifacts=[],
            route="analysis",
            tool_calls=1,
            tool_names=["plotly_tool"],
        )

        result = runner._finalize_node(
            {
                "response": response,
                "callbacks": [],
                "route": "analysis",
                "prompt": "Построй график продаж",
                "df": None,
                "stop_reason": "eval_failed",
                "eval_passed": False,
                "step_index": 1,
                "max_steps": 2,
            }
        )

        final_response = result["response"]
        self.assertNotEqual(final_response.final_text, "Я якобы построил график.")
        self.assertEqual(final_response.artifacts, [])


if __name__ == "__main__":
    unittest.main()
