from __future__ import annotations

from dataclasses import replace
import unittest

import pandas as pd

from backend.agent_capabilities import build_runtime_capability_context
from backend.agent_runner import AgentRunner
from backend.config import Settings


class _ExplodingGraph:
    def invoke(self, _state):
        raise AssertionError("graph.invoke should not be called when data access is denied")


class DataToolGuardrailTests(unittest.TestCase):
    def _build_runner(self, *, allowed_tool_keys: set[str] | None) -> AgentRunner:
        settings = replace(
            Settings(),
            agent_cache_enabled=False,
            llm_warmup_enabled=False,
        )
        runner = AgentRunner(
            settings,
            allowed_tool_keys=allowed_tool_keys,
        )
        runner._graph = _ExplodingGraph()
        return runner

    def test_dataset_query_is_blocked_when_all_data_tools_disabled(self) -> None:
        runner = self._build_runner(allowed_tool_keys={"search_tool"})
        df = pd.DataFrame({"sales": [10, 20, 30]})

        response = runner.run_query(
            df,
            "Посчитай среднее значение sales",
            history=[],
            use_history=False,
            include_reasoning=False,
            callbacks=[],
            trace_context={},
            session_source={"source_type": "csv"},
        )

        self.assertEqual(response.route, "analysis")
        self.assertIn("Не могу выполнить анализ по датасету", response.final_text)
        self.assertIn("pandas_tool", response.final_text)
        self.assertEqual(response.tool_calls, 0)

    def test_db_query_is_blocked_when_db_tool_disabled(self) -> None:
        runner = self._build_runner(allowed_tool_keys={"search_tool", "plotly_tool"})

        response = runner.run_query(
            None,
            "Сколько заказов было в прошлом месяце?",
            history=[],
            use_history=False,
            include_reasoning=False,
            callbacks=[],
            trace_context={},
            session_source={
                "source_type": "db_connection",
                "source_ref_id": "conn_123",
            },
        )

        self.assertEqual(response.route, "analysis")
        self.assertIn("подключенной базе данных", response.final_text)
        self.assertIn("db_tool", response.final_text)
        self.assertEqual(response.tool_calls, 0)

    def test_chat_request_on_dataset_is_not_blocked_by_guardrail(self) -> None:
        runner = self._build_runner(allowed_tool_keys=set())
        df = pd.DataFrame({"sales": [10, 20, 30]})

        response = runner._build_data_tools_disabled_response(
            df,
            "Привет",
            session_source={"source_type": "csv"},
        )

        self.assertIsNone(response)

    def test_capability_map_is_derived_from_actual_toolset(self) -> None:
        capability_context = build_runtime_capability_context(
            available_tool_keys={"pandas_tool", "value_tool", "search_tool"},
            has_dataframe=True,
            has_db_source=False,
        )

        self.assertEqual(capability_context["source_mode"], "dataset")
        self.assertIn("table_analysis", capability_context["available_capability_keys"])
        self.assertIn("external_search", capability_context["available_capability_keys"])
        self.assertIn("charting", capability_context["unavailable_capability_keys"])
        self.assertIn("forecasting", capability_context["unavailable_capability_keys"])
        self.assertNotIn("db_query", capability_context["available_capability_keys"])
        self.assertEqual(capability_context["available_tool_keys"], ["pandas_tool", "search_tool", "value_tool"])

    def test_capability_prompt_is_built_from_toolset_not_keywords(self) -> None:
        runner = self._build_runner(allowed_tool_keys={"pandas_tool"})
        capability_context = build_runtime_capability_context(
            available_tool_keys={"pandas_tool"},
            has_dataframe=True,
            has_db_source=False,
        )

        prompt = runner._think_system_prompt(capability_context)

        self.assertIn("[ROLE: CAPABILITIES]", prompt)
        self.assertIn("`pandas_tool`", prompt)
        self.assertIn("Недоступные capabilities", prompt)
        self.assertNotIn("CHART_HINTS", prompt)
        # Prompt must warn not to promise unavailable tools
        self.assertTrue(
            "не обещай" in prompt.lower() or "нельзя обещать" in prompt.lower(),
            msg="Prompt should warn against promising unavailable tools",
        )


if __name__ == "__main__":
    unittest.main()
