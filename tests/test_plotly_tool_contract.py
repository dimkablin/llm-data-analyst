from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go

from backend.tools.impl.plotly_tool import PlotlyTool, _figure_has_data_points
from backend.tools.sandbox import SessionSandbox


class PlotlyToolContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
                "value": [10, 20, 15],
            }
        )
        self.sandbox = SessionSandbox()
        self.sandbox.bind_dataframe(self.df)
        # 20s: Windows uses spawn (no fork), subprocess start takes ~3-4s per call.
        self.tool = PlotlyTool(
            self.df,
            execution_timeout_sec=20.0,
            tool_cache_size=0,
            sandbox=self.sandbox,
        )

    def test_single_axis_and_matrix_traces_are_valid_figures(self) -> None:
        figures = [
            go.Figure(go.Histogram(x=[1, 2, 3])),
            go.Figure(go.Box(y=[1, 2, 3])),
            go.Figure(go.Heatmap(z=[[1, 2], [3, 4]])),
        ]

        self.assertTrue(all(_figure_has_data_points(fig) for fig in figures))

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

    def test_fig_show_without_tool_result_uses_created_figure(self) -> None:
        code = """
fig = px.bar(df, x="category", y="value", title="Value comparison")
fig.show()
"""

        with patch.object(go.Figure, "show", return_value=None):
            text, payload = self.tool._run(code)

        self.assertIn("plotly_tool", text)
        self.assertIsInstance(payload["plot"]["plot"], go.Figure)

    def test_positional_artifact_name_passes(self) -> None:
        code = """
fig = px.bar(df, x="category", y="value")
tool_result = chart.result(fig, "comparison_plot")
tool_result
"""

        text, payload = self.tool._run(code)

        self.assertIn("plotly_tool", text)
        self.assertIsInstance(payload["plot"]["comparison_plot"], go.Figure)

    def test_px_bar_showlegend_kwarg_is_supported(self) -> None:
        code = """
fig = px.bar(df, x="category", y="value", showlegend=False)
tool_result = chart.result(fig, artifact_name="comparison_plot")
tool_result
"""

        text, payload = self.tool._run(code)

        self.assertIn("plotly_tool", text)
        self.assertIsInstance(payload["plot"]["comparison_plot"], go.Figure)
        self.assertIs(payload["plot"]["comparison_plot"].layout.showlegend, False)

    def test_axis_spanning_shape_annotations_support_nonnumeric_axes(self) -> None:
        cases = {
            "vline_string": """
fig = px.line(df, x="category", y="value")
fig.add_vline(x="B", line_dash="dash", annotation_text="Marker")
""",
            "hline_string": """
fig = px.line(df, x="value", y="category")
fig.add_hline(y="B", line_dash="dash", annotation_text="Marker")
""",
            "vrect_string_center": """
fig = px.line(df, x="category", y="value")
fig.add_vrect(x0="A", x1="B", annotation_text="Marker", annotation_position="inside")
""",
            "hrect_string_center": """
fig = px.line(df, x="value", y="category")
fig.add_hrect(y0="A", y1="B", annotation_text="Marker", annotation_position="inside")
""",
            "vline_datetime": """
ts_df = df.copy()
ts_df["month"] = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
fig = px.line(ts_df, x="month", y="value")
fig.add_vline(x=pd.Timestamp("2024-02-01"), line_dash="dash", annotation_text="Marker")
""",
            "subplot_vline": """
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=df["category"], y=df["value"]), row=1, col=1)
fig.add_vline(x="B", line_dash="dash", annotation_text="Marker", row=1, col=1)
""",
            "hline_annotation_dict": """
fig = px.line(df, x="value", y="category")
fig.add_hline(y="B", annotation={"text": "Marker"})
""",
        }

        for name, body in cases.items():
            with self.subTest(name=name):
                code = f"""
{body}
tool_result = chart.result(fig, artifact_name="{name}")
tool_result
"""

                text, payload = self.tool._run(code)

                self.assertIn("plotly_tool", text)
                self.assertIn("shape.label", text)
                fig = payload["plot"][name]
                self.assertIsInstance(fig, go.Figure)
                annotation_texts = [
                    item.text for item in fig.layout.annotations if getattr(item, "text", None)
                ]
                label_texts = [
                    item.label.text
                    for item in fig.layout.shapes
                    if getattr(item, "label", None) and getattr(item.label, "text", None)
                ]
                self.assertIn("Marker", annotation_texts + label_texts)

    def test_axis_spanning_shape_numeric_annotations_still_use_plotly_annotations(self) -> None:
        code = """
fig = px.line(df, x="value", y="value")
fig.add_vline(x=20, line_dash="dash", annotation_text="Marker")
tool_result = chart.result(fig, artifact_name="comparison_plot")
tool_result
"""

        _text, payload = self.tool._run(code)
        fig = payload["plot"]["comparison_plot"]

        self.assertEqual([item.text for item in fig.layout.annotations], ["Marker"])

    def test_axis_spanning_shape_compat_does_not_hide_unrelated_errors(self) -> None:
        code = """
fig = px.line(df, x="category", y="value")
fig.add_vline(x="B", annotation_text="Marker", line_width="wide")
tool_result = chart.result(fig, artifact_name="comparison_plot")
tool_result
"""

        text, payload = self.tool._run(code)

        self.assertIn("plotly_tool failed", text)
        self.assertIsNone(payload["plot"])

    def test_pandas_plot_bar_is_rejected(self) -> None:
        code = """
fig = df.plot.bar(x="category", y="value")
tool_result = chart.result(fig, artifact_name="comparison_plot")
tool_result
"""

        text, payload = self.tool._run(code)

        self.assertIn("plot", text.lower())
        self.assertIn("px.bar", text.lower())
        self.assertIsNone(payload["plot"])

    def test_forbidden_matplotlib_code_does_not_call_hidden_repair(self) -> None:
        code = """
fig = plt.figure()
tool_result = chart.result(fig, artifact_name="comparison_plot")
tool_result
"""

        with patch.object(PlotlyTool, "_fix_with_llm", side_effect=AssertionError("hidden repair"), create=True):
            text, payload = self.tool._run(code)

        self.assertIn("plotly_tool failed", text)
        self.assertIn("matplotlib", text)
        self.assertEqual(payload["status"], "error")
        self.assertIsNone(payload["plot"])

    def test_forbidden_matplotlib_code_reports_plotly_specific_guidance(self) -> None:
        tool = PlotlyTool(
            self.df,
            execution_timeout_sec=20.0,
            tool_cache_size=0,
            sandbox=self.sandbox,
        )

        text, payload = tool._run("fig = plt.figure()")

        self.assertIn("plotly_tool нельзя использовать matplotlib", text)
        self.assertIn("px", text)
        self.assertIn("go", text)
        self.assertNotIn("используй plotly_tool", text)
        self.assertIsNone(payload["plot"])

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
