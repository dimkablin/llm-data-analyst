"""Unit tests for the generic autogen plot recipe registry."""

import inspect

import pandas as pd

from backend.services import plot_recipes


def test_autogen_specs_use_table_shape_without_prompt_input() -> None:
    df = pd.DataFrame(
        {
            "category": ["A", "B"],
            "account_type": ["retail", "enterprise"],
            "value": [12.0, 7.0],
        }
    )
    specs = plot_recipes.build_autogen_plot_specs(
        df,
        value_col="value",
        segment_cols=["category", "account_type"],
        source_table="customer_segments",
    )
    assert "prompt" not in inspect.signature(plot_recipes.build_autogen_plot_specs).parameters
    assert specs


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
