"""Unit tests for the generic autogen plot recipe registry."""

import pandas as pd

from backend.services import plot_recipes
from backend.services.chart_type_selector import classify_plot_intent


def test_classify_plot_intent_matches_chart_selector() -> None:
    assert classify_plot_intent("show concentration by channel") == "concentration"
    assert classify_plot_intent("show revenue trend by month") == "dynamics"
    assert classify_plot_intent("describe structure by category") == "structure"


def test_build_autogen_structure_specs() -> None:
    df = pd.DataFrame(
        {
            "category": ["A", "B"],
            "account_type": ["retail", "enterprise"],
            "value": [12.0, 7.0],
        }
    )
    specs = plot_recipes.build_autogen_plot_specs(
        "describe structure by category and account",
        df,
        value_col="value",
        segment_cols=["category", "account_type"],
        source_table="customer_segments",
    )

    assert specs
    names = {name for name, _, _ in specs}
    assert any(name.startswith("structure_by_") for name in names)


def test_quantity_column_is_not_time_axis() -> None:
    from backend.services.chart_type_selector import is_plausible_time_column

    assert not is_plausible_time_column("quantity", pd.Series([10, 20, 30, 40]))


def test_single_as_of_date_is_not_timeseries() -> None:
    df = pd.DataFrame(
        {
            "as_of_date": ["2025-12-31"] * 6,
            "quantity": [10, 20, 30, 40, 50, 60],
            "item_id": [f"L{i}" for i in range(6)],
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    assert not plot_recipes._frame_suitable_for_timeseries_line("customer_segments", df)
