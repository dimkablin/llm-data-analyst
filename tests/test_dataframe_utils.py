import pandas as pd

from backend.data_access.dataframe_utils import column_nunique, deduplicate_dataframe_columns


def test_column_nunique_with_duplicate_labels() -> None:
    df = pd.DataFrame(
        [[1, 10, "a"], [2, 20, "b"]],
        columns=["x", "x", "group"],
    )
    assert column_nunique(df, "group") == 2
    deduped = deduplicate_dataframe_columns(df)
    assert list(deduped.columns) == ["x", "x_2", "group"]
