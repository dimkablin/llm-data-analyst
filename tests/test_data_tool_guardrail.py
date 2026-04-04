from __future__ import annotations

import unittest
from dataclasses import replace

import pandas as pd

from backend.agent import AgentRunner
from backend.core import Settings
from backend.tools import build_runtime_capability_context


class _ExplodingGraph:
    def invoke(self, _state):
        raise AssertionError("graph.invoke should not be called when access to data tools is denied")


class DataToolGuardrailTests(unittest.TestCase):
    def _build_runner(self, *, allowed_tool_keys: set[str] | None) -> AgentRunner:
        settings = replace(
            Settings(),
            agent_cache_enabled=False,
            llm_warmup_enabled=False,
        )
        runner = AgentRunner(settings, allowed_tool_keys=allowed_tool_keys)
        runner._graph = _ExplodingGraph()
        return runner

    def test_dataset_analysis_is_blocked_when_only_non_data_tools_are_enabled(self) -> None:
        runner = self._build_runner(allowed_tool_keys={"search_tool"})
        df = pd.DataFrame({"sales": [10, 20, 30]})

        response = runner.run_query(
            df,
            "посчитай среднее значение sales",
            history=[],
            use_history=False,
            include_reasoning=False,
            callbacks=[],
            trace_context={},
            session_source={"source_type": "csv"},
        )

        self.assertEqual(response.route, "analysis")
        self.assertEqual(response.tool_calls, 0)
        self.assertEqual(response.artifacts, [])
        self.assertIn("pandas_tool", response.final_text)
        self.assertIn("value_tool", response.final_text)
        self.assertIn("plotly_tool", response.final_text)

    def test_dataset_analysis_is_not_blocked_when_dataframe_tool_is_enabled(self) -> None:
        runner = self._build_runner(allowed_tool_keys={"pandas_tool"})
        df = pd.DataFrame({"sales": [10, 20, 30]})

        response = runner._build_data_tools_disabled_response(
            df,
            "посчитай среднее значение sales",
            session_source={"source_type": "csv"},
        )

        self.assertIsNone(response)

    def test_db_analysis_is_blocked_when_sql_tool_is_disabled(self) -> None:
        runner = self._build_runner(allowed_tool_keys={"search_tool", "plotly_tool"})

        response = runner.run_query(
            None,
            "сколько заказов было в прошлом месяце?",
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
        self.assertEqual(response.tool_calls, 0)
        self.assertEqual(response.artifacts, [])
        self.assertIn("sql_tool", response.final_text)
        self.assertIn("plotly_tool", response.final_text)

    def test_greeting_with_dataset_is_not_blocked_by_guardrail(self) -> None:
        runner = self._build_runner(allowed_tool_keys=set())
        df = pd.DataFrame({"sales": [10, 20, 30]})

        response = runner._build_data_tools_disabled_response(
            df,
            "привет",
            session_source={"source_type": "csv"},
        )

        self.assertIsNone(response)

    def test_capability_context_reflects_only_runtime_toolset(self) -> None:
        capability_context = build_runtime_capability_context(
            available_tool_keys={"pandas_tool", "value_tool", "search_tool"},
            has_dataframe=True,
            has_db_source=False,
        )

        self.assertEqual(capability_context["source_mode"], "dataset")
        self.assertEqual(
            capability_context["available_tool_keys"],
            ["pandas_tool", "search_tool", "value_tool"],
        )
        self.assertIn("table_analysis", capability_context["available_capability_keys"])
        self.assertIn("external_search", capability_context["available_capability_keys"])
        self.assertIn("charting", capability_context["unavailable_capability_keys"])
        self.assertIn("forecasting", capability_context["unavailable_capability_keys"])
        self.assertNotIn("db_query", capability_context["available_capability_keys"])

    def test_think_prompt_describes_available_and_unavailable_capabilities(self) -> None:
        runner = self._build_runner(allowed_tool_keys={"pandas_tool"})
        capability_context = build_runtime_capability_context(
            available_tool_keys={"pandas_tool"},
            has_dataframe=True,
            has_db_source=False,
        )

        prompt = runner._think_system_prompt(capability_context)

        self.assertIn("[ROLE: CAPABILITIES]", prompt)
        self.assertIn("`pandas_tool`", prompt)
        self.assertIn("Доступные capabilities", prompt)
        self.assertIn("Недоступные capabilities", prompt)
        self.assertIn("plotly_tool", prompt)
        self.assertNotIn("CHART_HINTS", prompt)


if __name__ == "__main__":
    unittest.main()
