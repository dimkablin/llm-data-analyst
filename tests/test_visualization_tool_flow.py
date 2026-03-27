from __future__ import annotations

from dataclasses import replace
from types import MethodType, SimpleNamespace
import unittest

import pandas as pd

from backend.agent import AgentResponse, AgentRunner
from backend.core import ArtifactRecord, Settings


class VisualizationToolFlowTests(unittest.TestCase):
    def _build_runner(self) -> AgentRunner:
        settings = replace(
            Settings(),
            agent_cache_enabled=False,
            llm_warmup_enabled=False,
            agent_evaluate_enabled=False,
        )
        return AgentRunner(settings, allowed_tool_keys={"plotly_tool", "pandas_tool"})

    def test_act_node_retries_when_plot_request_returns_no_plot_artifact(self) -> None:
        runner = self._build_runner()
        calls: list[str] = []

        first = AgentResponse(
            final_text="План готов, сейчас построю график.",
            reasoning=None,
            artifacts=[],
            route="analysis",
            tool_calls=0,
            tool_names=[],
        )
        second = AgentResponse(
            final_text="Построил график.",
            reasoning=None,
            artifacts=[
                ArtifactRecord(
                    artifact_type="plot",
                    data={"figure": "ok"},
                    text="survival_chart",
                )
            ],
            route="analysis",
            tool_calls=1,
            tool_names=["plotly_tool"],
        )

        def fake_analysis_step(self, **kwargs):
            calls.append(str(kwargs["prompt"]))
            return first if len(calls) == 1 else second

        runner._analysis_step = MethodType(fake_analysis_step, runner)  # type: ignore[method-assign]

        result = runner._act_node(
            {
                "df": pd.DataFrame({"Survived": [0, 1], "Age": [22, 38]}),
                "prompt": "Построй график выживаемости",
                "history": [],
                "use_history": False,
                "include_reasoning": False,
                "callbacks": [],
                "trace_context": {},
                "session_source": {"source_type": "csv"},
                "tools": [SimpleNamespace(name="plotly_tool")],
                "plan": "Нужен график.",
                "step_index": 0,
                "max_steps": 4,
                "capability_context": {},
            }
        )

        self.assertEqual(len(calls), 2)
        self.assertIn("[ROLE: VISUALIZATION_RETRY]", calls[1])
        self.assertEqual(result["response"].tool_names, ["plotly_tool"])
        self.assertEqual(len(result["response"].artifacts), 1)
        self.assertEqual(result["response"].artifacts[0].artifact_type, "plot")

    def test_evaluate_node_requires_plot_artifact_for_visualization_prompt(self) -> None:
        runner = self._build_runner()
        response = AgentResponse(
            final_text="Подготовил сводную таблицу по выживаемости.",
            reasoning=None,
            artifacts=[
                ArtifactRecord(
                    artifact_type="table",
                    data={"rows": []},
                    text="survival_table",
                )
            ],
            route="analysis",
            tool_calls=1,
            tool_names=["pandas_tool"],
        )

        result = runner._evaluate_node(
            {
                "response": response,
                "callbacks": [],
                "prompt": "Построй график выживаемости по полу",
                "step_index": 1,
                "max_steps": 4,
                "capability_context": {},
                "df": pd.DataFrame({"Survived": [0, 1], "Sex": ["male", "female"]}),
            }
        )

        self.assertFalse(result["eval_passed"])
        self.assertIn("plot-артефакт", result["eval_reason"])


if __name__ == "__main__":
    unittest.main()
