from __future__ import annotations

import unittest

import pandas as pd
import plotly.graph_objects as go

from agent.tools.plotly_tool import PlotlyTool


class PlotlyToolContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
                "value": [10, 20, 15],
            }
        )
        self.tool = PlotlyTool(self.df, execution_timeout_sec=5.0, tool_cache_size=0)

    def test_invalid_string_result_is_rejected(self) -> None:
        code = """
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "plot",
    "items": {
        "comparison_plot": "not a figure"
    }
}
tool_result
"""

        text, payload = self.tool._run(code)

        self.assertIn("Ошибка валидации результатов", text)
        self.assertIn("Тип данных: str", text)
        self.assertIsNone(payload["plot"])

    def test_invalid_dict_result_is_rejected(self) -> None:
        code = """
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "plot",
    "items": {
        "comparison_plot": {"x": [1, 2], "y": [3, 4]}
    }
}
tool_result
"""

        text, payload = self.tool._run(code)

        self.assertIn("Ошибка валидации результатов", text)
        self.assertIn("Тип данных: dict", text)
        self.assertIsNone(payload["plot"])

    def test_missing_variable_error_is_reported(self) -> None:
        code = """
fig = data_for_plot
tool_result = chart.result(fig, artifact_name="comparison_plot")
tool_result
"""

        text, payload = self.tool._run(code)

        self.assertIn("Ошибка при создании", text)
        self.assertIn("data_for_plot", text)
        self.assertIsNone(payload["plot"])

    def test_valid_figure_result_passes(self) -> None:
        code = """
fig = px.bar(df, x="category", y="value", title="Сравнение значений")
tool_result = chart.result(fig, artifact_name="comparison_plot")
tool_result
"""

        text, payload = self.tool._run(code)

        self.assertIn("Создано через plotly_tool", text)
        self.assertIsInstance(payload["plot"]["comparison_plot"], go.Figure)


if __name__ == "__main__":
    unittest.main()
