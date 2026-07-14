import pandas as pd

from backend.services import plot_recipes


def test_segment_value_breakdown_percentages() -> None:
    df = pd.DataFrame(
        {
            "security_type": ["акция", "акция", "облигация"],
            "market_value_mln_rub": [70.0, 0.0, 30.0],
        }
    )
    rows = plot_recipes.segment_value_breakdown(
        df,
        segment_col="security_type",
        value_col="market_value_mln_rub",
    )
    assert len(rows) == 2
    by_label = {str(row["label"]): float(row["share_pct"]) for row in rows}
    assert by_label["акция"] == 70.0
    assert by_label["облигация"] == 30.0
