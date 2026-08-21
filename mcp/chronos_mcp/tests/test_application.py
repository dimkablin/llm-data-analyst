from __future__ import annotations

import math

import pytest

from chronos_mcp.application import (
    ChronosApplication,
    ModelForecast,
    ModelMetadata,
    ModelPoint,
)
from chronos_mcp.contracts import (
    Aggregation,
    BacktestEvaluation,
    BacktestRequest,
    ForecastRequest,
    Frequency,
    InlineSource,
    MissingPolicy,
    Target,
)
from chronos_mcp.errors import ChronosMCPError, ErrorCode


class FakeRuntime:
    def __init__(self) -> None:
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        points = []
        for context in request.contexts:
            points.extend(
                [
                    ModelPoint(
                        series_id=context.series_id,
                        target=context.target,
                        timestamp=timestamp,
                        prediction=42.0,
                        quantiles={0.1: 40.0, 0.5: 42.0, 0.9: 44.0},
                    )
                    for timestamp in context.future_timestamps
                ]
            )
        return ModelForecast(
            points=points,
            metadata=ModelMetadata(
                library_version="test",
                model_alias="fake",
                model_id="fake/model",
                model_revision="test-revision",
                family="fake",
                capabilities=["quantiles"],
                inference_ms=1,
                cached=True,
            ),
        )

    def capabilities(self):
        return []


def request_with_rows(rows, *, missing_policy=MissingPolicy.error) -> ForecastRequest:
    return ForecastRequest(
        source=InlineSource(rows=rows),
        time_column="ts",
        targets=[Target(name="sales", column="y", aggregation=Aggregation.none)],
        horizon=1,
        frequency=Frequency.month_start,
        missing_policy=missing_policy,
        quantiles=[0.1, 0.5, 0.9],
    )


def test_horizon_one_stays_an_array_with_one_forecast_row() -> None:
    runtime = FakeRuntime()
    application = ChronosApplication(runtime=runtime)

    response = application.forecast(
        request_with_rows(
            [
                {"ts": "2026-01-01", "y": 10.0},
                {"ts": "2026-02-01", "y": 12.0},
            ]
        )
    )

    assert len(response.rows) == 1
    assert response.rows[0].ts.isoformat() == "2026-03-01T00:00:00+00:00"
    assert response.rows[0].prediction == 42.0
    assert response.rows[0].lower == 40.0
    assert response.rows[0].upper == 44.0


def test_zero_policy_fills_missing_period_before_model_call() -> None:
    runtime = FakeRuntime()
    application = ChronosApplication(runtime=runtime)

    response = application.forecast(
        request_with_rows(
            [
                {"ts": "2026-01-01", "y": 10.0},
                {"ts": "2026-03-01", "y": 30.0},
            ],
            missing_policy=MissingPolicy.zero,
        )
    )

    assert runtime.requests[0].contexts[0].values == (10.0, 0.0, 30.0)
    assert response.data_meta.missing_periods == 1


def test_non_finite_target_is_rejected_before_model_call() -> None:
    runtime = FakeRuntime()
    application = ChronosApplication(runtime=runtime)

    with pytest.raises(ChronosMCPError) as captured:
        application.forecast(
            request_with_rows(
                [
                    {"ts": "2026-01-01", "y": 10.0},
                    {"ts": "2026-02-01", "y": math.nan},
                ]
            )
        )

    assert captured.value.code == ErrorCode.non_finite_value
    assert runtime.requests == []


def test_missing_period_error_is_typed() -> None:
    application = ChronosApplication(runtime=FakeRuntime())

    with pytest.raises(ChronosMCPError) as captured:
        application.forecast(
            request_with_rows(
                [
                    {"ts": "2026-01-01", "y": 10.0},
                    {"ts": "2026-03-01", "y": 30.0},
                ]
            )
        )

    assert captured.value.code == ErrorCode.missing_periods


def test_backtest_uses_rolling_historical_cutoffs() -> None:
    runtime = FakeRuntime()
    application = ChronosApplication(runtime=runtime)
    request = BacktestRequest(
        **request_with_rows(
            [
                {"ts": "2026-01-01", "y": 1.0},
                {"ts": "2026-02-01", "y": 2.0},
                {"ts": "2026-03-01", "y": 3.0},
                {"ts": "2026-04-01", "y": 4.0},
                {"ts": "2026-05-01", "y": 5.0},
                {"ts": "2026-06-01", "y": 6.0},
            ]
        ).model_dump(),
        evaluation=BacktestEvaluation(
            windows=2,
            step=1,
            min_context_points=2,
            metrics=["mae", "mase"],
        ),
    )

    response = application.backtest(request)

    assert [row.actual for row in response.rows] == [5.0, 6.0]
    assert [row.cutoff.month for row in response.rows] == [4, 5]
    assert response.metrics.overall == {"mae": 36.5, "mase": 36.5}
    assert [len(call.contexts[0].values) for call in runtime.requests] == [4, 5]
