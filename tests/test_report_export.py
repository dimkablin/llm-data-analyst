import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backend.services.report_export import (
    _append_plot_data_sheet,
    _board_export_sections,
    _normalize_plotly_trace,
    _strip_duplicate_note_heading,
    build_board_export,
)
from backend.tools.impl.plotly_tool import apply_default_chart_style


def test_strip_duplicate_note_heading_removes_suty_prefix() -> None:
    content = "Суть Лидером по вкладу является акция NREH."
    cleaned = _strip_duplicate_note_heading(content, "Суть")
    assert cleaned.startswith("Лидером")
    assert "Суть Суть" not in cleaned


def test_strip_duplicate_note_heading_removes_bold_line() -> None:
    content = "**Суть**\n\nПортфель демонстрирует концентрацию."
    cleaned = _strip_duplicate_note_heading(content, "**Суть**")
    assert cleaned.startswith("Портфель")
    assert "**Суть**" not in cleaned.splitlines()[0]


def test_board_export_sections_group_by_question() -> None:
    artifacts = [
        {"id": "plot1", "type": "plot", "text": "chart_a"},
        {"id": "note1", "type": "note", "text": "note"},
        {"id": "plot2", "type": "plot", "text": "chart_b"},
    ]
    sections = [
        {"label": "Вопрос 1: структура", "artifact_ids": ["plot1", "note1"]},
        {"label": "Вопрос 2: концентрация", "artifact_ids": ["plot2"]},
    ]
    grouped = _board_export_sections(artifacts, sections)
    assert [label for label, _ in grouped] == [
        "Вопрос 1: структура",
        "Вопрос 2: концентрация",
    ]
    assert grouped[0][1][0]["id"] == "plot1"
    assert grouped[1][1][0]["id"] == "plot2"


def test_board_bar_export_uses_single_color() -> None:
    from backend.services import report_export

    df = pd.DataFrame({"ticker": ["A", "B", "C"], "value": [1.0, 2.0, 3.0]})
    fig = apply_default_chart_style(
        go.Figure(go.Bar(x=df["ticker"], y=df["value"], orientation="v"))
    )
    colors_before = fig.data[0].marker.color
    assert isinstance(colors_before, (list, tuple)) and len(colors_before) > 1

    report_export._apply_board_bar_colors(fig)
    assert fig.data[0].marker.color == report_export._BOARD_BAR_COLOR


def test_report_plotly_trace_decodes_typed_array_bar_points() -> None:
    fig = go.Figure(
        go.Bar(x=np.array([10.0, 20.0, 30.0]), y=["a", "b", "c"], orientation="h")
    )
    trace = fig.to_plotly_json()["data"][0]

    normalized = _normalize_plotly_trace(trace, 0)

    assert normalized["marker"]["color"] == ["#2563eb", "#7c3aed", "#0f766e"]


def test_report_xlsx_plot_data_decodes_typed_arrays() -> None:
    from openpyxl import Workbook

    fig = go.Figure(
        go.Bar(x=np.array([10.0, 20.0, 30.0]), y=["a", "b", "c"], name="series")
    )
    artifact = {
        "id": "plot1",
        "type": "plot",
        "text": "Chart",
        "data": {"format": "plotly-json", "data": fig.to_plotly_json()},
        "meta": {},
    }
    workbook = Workbook()
    summary_sheet = workbook.active

    _append_plot_data_sheet(
        workbook=workbook,
        artifact=artifact,
        artifact_index={"Chart": artifact},
        used_titles=set(),
        summary_sheet=summary_sheet,
    )

    rows = list(workbook["Chart"].iter_rows(values_only=True))
    assert rows[1:4] == [("series", 10.0, "a"), ("series", 20.0, "b"), ("series", 30.0, "c")]


def test_build_board_export_requires_artifacts(tmp_path) -> None:
    try:
        build_board_export(
            title="Test",
            artifacts=[],
            output_dir=tmp_path,
            export_format="docx",
        )
    except ValueError as exc:
        assert "артефакт" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for empty artifacts")
