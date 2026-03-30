from __future__ import annotations

from dataclasses import replace
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
