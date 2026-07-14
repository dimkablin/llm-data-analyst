import pandas as pd

from backend.services.chart_type_selector import (
    infer_tabular_plot_profile,
    pick_chart_kind,
    pick_metric_pair,
)


def test_infer_profile_from_generic_sales_table() -> None:
    df = pd.DataFrame(
        {
            "report_month": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "region": ["North", "South", "North"],
            "revenue": [100.0, 120.0, 90.0],
            "units": [10, 12, 9],
        }
    )
    profile = infer_tabular_plot_profile(df)

    assert profile.time_columns
    assert "region" in profile.dimension_columns
    assert "revenue" in profile.metric_columns


def test_pick_chart_kind_uses_cardinality_not_column_names() -> None:
    df = pd.DataFrame(
        {
            "entity_label": [f"item_{index}" for index in range(10)],
            "amount": [float(index) for index in range(10)],
        }
    )

    assert (
        pick_chart_kind(
            intent="structure",
            segment_index=0,
            df=df,
            segment_col="entity_label",
            prompt="distribution",
        )
        == "bar_h"
    )


def test_pick_metric_pair_from_numeric_columns() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "label": ["x", "y"]})
    pair = pick_metric_pair(df)

    assert pair == ("a", "b")
