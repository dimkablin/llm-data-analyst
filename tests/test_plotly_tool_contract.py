from __future__ import annotations

import unittest

import pandas as pd
import plotly.graph_objects as go

from backend.tools.impl.plotly_tool import PlotlyTool


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

        self.assertIn("str", text)
        self.assertIn("plot", text.lower())
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

        self.assertIn("dict", text)
        self.assertIn("plot", text.lower())
        self.assertIsNone(payload["plot"])

    def test_missing_variable_error_is_reported(self) -> None:
        code = """
fig = data_for_plot
tool_result = chart.result(fig, artifact_name="comparison_plot")
tool_result
"""

        text, payload = self.tool._run(code)

        self.assertIn("data_for_plot", text)
        self.assertIsNone(payload["plot"])

    def test_valid_figure_result_passes(self) -> None:
        code = """
fig = px.bar(df, x="category", y="value", title="Value comparison")
tool_result = chart.result(fig, artifact_name="comparison_plot")
tool_result
"""

        text, payload = self.tool._run(code)

        self.assertIn("plotly_tool", text)
        self.assertIsInstance(payload["plot"]["comparison_plot"], go.Figure)

    def test_positional_artifact_name_passes(self) -> None:
        code = """
fig = px.bar(df, x="category", y="value")
tool_result = chart.result(fig, "comparison_plot")
tool_result
"""

        text, payload = self.tool._run(code)

        self.assertIn("plotly_tool", text)
        self.assertIsInstance(payload["plot"]["comparison_plot"], go.Figure)

    def test_multiple_chart_result_calls_are_accumulated(self) -> None:
        code = """
fig1 = px.bar(df, x="category", y="value", title="Bar")
fig2 = px.line(df, x="category", y="value", title="Line")

chart.result(fig1, "bar_plot")
chart.result(fig2, "line_plot")
"""

        text, payload = self.tool._run(code)

        self.assertIn("plotly_tool", text)
        self.assertEqual(set(payload["plot"].keys()), {"bar_plot", "line_plot"})
        self.assertIsInstance(payload["plot"]["bar_plot"], go.Figure)
        self.assertIsInstance(payload["plot"]["line_plot"], go.Figure)


if __name__ == "__main__":
    unittest.main()
