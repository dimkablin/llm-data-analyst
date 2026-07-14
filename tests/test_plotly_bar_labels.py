import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backend.services import plot_recipes
from backend.tools.impl.plotly_tool import (
    _bar_label_texts,
    _numeric_bar_values,
    apply_default_chart_style,
)


def test_numeric_bar_values_with_numpy_axis() -> None:
    trace = go.Bar(
        x=np.array([2.26, 5.05, 1.94]),
        y=np.array(["BRIZ", "STEL", "TERA"]),
        orientation="h",
    )
    values = _numeric_bar_values(trace)
    assert values == [2.26, 5.05, 1.94]


def test_apply_default_chart_style_on_horizontal_bar() -> None:
    fig = plot_recipes.build_single_segment_horizontal_bar_figure(
        pd.DataFrame(
            {
                "ticker": ["BRIZ", "STEL", "TERA", "VEGA", "BRIZ27", "VEGA29", "KEDR", "NREH"],
                "market_value_mln_rub": [2.26, 2.2, 1.94, 2.1, 5.05, 2.0, 2.3, 2.5],
            }
        ),
        value_col="market_value_mln_rub",
        segment_col="ticker",
        title="Топ: ticker",
    )
    assert fig is not None
    styled = apply_default_chart_style(fig)
    assert styled is not None
    assert len(styled.data) >= 1


def test_horizontal_bar_large_numeric_x_stays_numeric() -> None:
    fig = go.Figure(
        data=[
            go.Bar(
                x=[2139052307.0, 2113121616.0],
                y=["A", "B"],
                orientation="h",
            )
        ]
    )

    styled = apply_default_chart_style(fig)

    assert list(styled.data[0].x) == [2139052307.0, 2113121616.0]
    assert styled.layout.xaxis.type != "date"


def test_horizontal_bar_labels_decode_plotly_typed_array_x() -> None:
    fig = go.Figure(
        go.Bar(
            x=np.array([5_350_000_000.0, 4_540_000_000.0, 2_100_000_000.0]),
            y=["Обувь", "Туризм и отдых", "Одежда"],
            orientation="h",
            texttemplate="%{x:.2s}",
        )
    )
    rehydrated = go.Figure(fig.to_plotly_json())

    styled = apply_default_chart_style(rehydrated)

    assert list(styled.data[0].text) == ["44.6%", "37.9%", "17.5%"]
    assert styled.data[0].texttemplate == "%{text}"


def test_bar_label_texts_with_numpy_values() -> None:
    trace = go.Bar(x=np.array([10.0, 20.0, 30.0]), y=["a", "b", "c"], orientation="h")
    labels = _bar_label_texts(trace)
    assert labels is not None
    assert len(labels) == 3
