import plotly.graph_objects as go

from backend.tools.impl.plotly_tool import (
    _bar_label_texts,
    _bar_point_count,
    apply_default_chart_style,
)


def test_bar_label_texts_percentages_for_structure() -> None:
    trace = go.Bar(x=["акция", "облигация"], y=[70, 30])
    labels = _bar_label_texts(trace)
    assert labels == ["70.0%", "30.0%"]


def test_bar_label_texts_skipped_for_many_bars() -> None:
    trace = go.Bar(x=[f"t{i}" for i in range(10)], y=list(range(10, 20)))
    assert _bar_label_texts(trace) is None
    assert _bar_point_count(trace) == 10


def test_apply_default_chart_style_enables_legend_for_multi_series() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 1], name="Оффлайн"))
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 1, 2], name="Онлайн"))
    styled = apply_default_chart_style(fig)
    assert styled.layout.showlegend is True


def test_apply_default_chart_style_hides_legend_for_single_bar_series() -> None:
    fig = go.Figure(go.Bar(x=["акция", "облигация"], y=[70, 30]))
    styled = apply_default_chart_style(fig)
    assert styled.layout.showlegend is False
    assert list(styled.data[0].text or []) == ["70.0%", "30.0%"]
