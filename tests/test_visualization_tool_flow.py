from __future__ import annotations

import unittest
from dataclasses import replace

import pandas as pd

from backend.agent import AgentResponse, AgentRunner
from backend.artifacts.execution import ExecArtifactType, ExecutionArtifact
from backend.core import Settings


def _build_runner() -> AgentRunner:
    settings = replace(
        Settings(),
        agent_cache_enabled=False,
        llm_warmup_enabled=False,
    )
    return AgentRunner(settings, allowed_tool_keys={"plotly_tool", "pandas_tool", "value_tool"})


@unittest.skip("_evaluate_node was removed in stream-based runner refactor")
class VisualizationToolFlowTests(unittest.TestCase):
    def _build_runner(self) -> AgentRunner:
        return _build_runner()

    def test_evaluate_node_requires_plot_artifact_for_visualization_prompt(self) -> None:
        runner = self._build_runner()
        response = AgentResponse(
            final_text="Подготовил сводную таблицу по выживаемости.",
            reasoning=None,
            artifacts=[
                ExecutionArtifact(
                    artifact_type=ExecArtifactType.DATAFRAME,
                    producer_tool="pandas_tool",
                    data={"rows": []},
                    name="survival_table",
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

    def test_evaluate_rewrites_plan_like_text_from_value_artifact(self) -> None:
        runner = self._build_runner()
        response = AgentResponse(
            final_text=(
                "План выполнения задачи:\n"
                "Что хочет пользователь?\n"
                "Тип задачи: метрика\n"
                "Выполняю шаг 1"
            ),
            reasoning=None,
            artifacts=[
                ExecutionArtifact(
                    artifact_type=ExecArtifactType.SCALAR,
                    producer_tool="value_tool",
                    data={
                        "number_of_rows": 891,
                        "columns_size": 12,
                    },
                    name="values",
                )
            ],
            route="analysis",
            tool_calls=1,
            tool_names=["value_tool"],
        )

        result = runner._evaluate_node(
            {
                "response": response,
                "callbacks": [],
                "prompt": "Сколько строк и столбцов в датасете?",
                "step_index": 1,
                "max_steps": 4,
                "capability_context": {},
                "df": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            }
        )

        self.assertTrue(result["eval_passed"])
        self.assertIn("891", response.final_text)
        self.assertIn("12", response.final_text)
        self.assertNotIn("План выполнения задачи", response.final_text)


if __name__ == "__main__":
    unittest.main()
