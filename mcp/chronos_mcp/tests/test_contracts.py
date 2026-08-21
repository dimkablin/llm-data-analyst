from __future__ import annotations

import pytest
from pydantic import ValidationError

from chronos_mcp.contracts import (
    Aggregation,
    ForecastRequest,
    Frequency,
    InlineSource,
    MissingPolicy,
    Target,
)


def valid_request(**updates) -> ForecastRequest:
    values = {
        "source": InlineSource(
            rows=[
                {"ts": "2026-01-01", "y": 10.0},
                {"ts": "2026-02-01", "y": 12.0},
            ]
        ),
        "time_column": "ts",
        "targets": [Target(name="sales", column="y", aggregation=Aggregation.none)],
        "horizon": 1,
        "frequency": Frequency.month_start,
        "missing_policy": MissingPolicy.error,
    }
    values.update(updates)
    return ForecastRequest(**values)


def test_forecast_request_rejects_pandas_frequency_alias() -> None:
    with pytest.raises(ValidationError):
        valid_request(frequency="MS")


def test_forecast_request_rejects_raw_sql() -> None:
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(
            {
                **valid_request().model_dump(mode="json"),
                "sql": "SELECT * FROM private_table",
            }
        )


def test_quantiles_are_sorted_unique_and_finite() -> None:
    request = valid_request(quantiles=[0.9, 0.1, 0.5, 0.5])

    assert request.quantiles == [0.1, 0.5, 0.9]

    with pytest.raises(ValidationError):
        valid_request(quantiles=[0.1, float("nan"), 0.9])
