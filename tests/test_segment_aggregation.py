import pandas as pd

from backend.data_access.segment_aggregation import (
    compute_segment_shares,
    detect_overload_segments,
    resolve_segment_metric,
)


def test_resolve_weight_column_for_share_table() -> None:
    df = pd.DataFrame(
        {
            "item_id": ["L1", "L2", "L3"],
            "segment": ["low", "medium", "high"],
            "share_pct": [18.0, 45.0, 37.0],
            "value": [1.0, 2.0, 3.0],
        }
    )
    metric = resolve_segment_metric(df)

    assert metric is not None
    assert metric.column == "share_pct"
    assert metric.kind == "weight"


def test_segment_shares_sum_weights_not_average() -> None:
    df = pd.DataFrame(
        {
            "segment": ["medium", "medium", "high", "low"],
            "share_pct": [20.0, 43.0, 19.0, 18.0],
        }
    )
    rows = compute_segment_shares(df, "segment")
    by_label = {row.label: row.share_pct for row in rows}

    assert by_label["medium"] == 63.0
    assert by_label["high"] == 19.0
    assert by_label["low"] == 18.0


def test_consistent_shares_for_structure_and_concentration() -> None:
    df = pd.DataFrame(
        {
            "account_type": ["A", "A", "B", "C"],
            "share_pct": [20.0, 21.71, 27.68, 30.61],
        }
    )
    rows = compute_segment_shares(df, "account_type")
    shares = {row.label: row.share_pct for row in rows}

    assert abs(shares["A"] - 41.71) < 0.1
    assert abs(shares["C"] - 30.61) < 0.1


def test_overload_detection() -> None:
    df = pd.DataFrame(
        {
            "account_type": ["A", "B"],
            "share_pct": [41.71, 27.68],
        }
    )
    rows = compute_segment_shares(df, "account_type")
    overloaded = detect_overload_segments(rows)

    assert any(row.label == "A" for row in overloaded)


def test_absolute_metric_when_no_weight_column() -> None:
    df = pd.DataFrame(
        {
            "channel": ["online", "offline", "online"],
            "revenue": [100.0, 50.0, 200.0],
        }
    )
    metric = resolve_segment_metric(df)
    assert metric is not None
    assert metric.kind == "absolute"
    rows = compute_segment_shares(df, "channel", metric=metric)
    shares = {row.label: row.share_pct for row in rows}

    assert abs(shares["online"] - 300 / 350 * 100) < 0.1
