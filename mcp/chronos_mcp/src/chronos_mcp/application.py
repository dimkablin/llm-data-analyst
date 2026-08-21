from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from chronos_mcp.contracts import (
    BacktestMetrics,
    BacktestRequest,
    BacktestRow,
    BacktestSuccess,
    CapabilitiesResponse,
    DataMeta,
    ErrorDetail,
    ErrorResponse,
    ForecastRequest,
    ForecastRow,
    ForecastSuccess,
    Interval,
    ModelCapability,
    ModelMeta,
    PlotPayload,
    TableSource,
    WarningMessage,
)
from chronos_mcp.errors import ChronosMCPError, ErrorCode
from chronos_mcp.preparation import (
    PreparationMetadata,
    PreparedContext,
    PreparedForecast,
    TableReader,
    prepare_forecast,
)


@dataclass(frozen=True)
class ModelPoint:
    series_id: dict[str, str | int | float | bool | None]
    target: str
    timestamp: datetime
    prediction: float
    quantiles: dict[float, float]


@dataclass(frozen=True)
class ModelMetadata:
    library_version: str
    model_alias: str
    model_id: str
    model_revision: str
    family: str
    capabilities: list[str]
    inference_ms: int
    cached: bool


@dataclass(frozen=True)
class ModelForecast:
    points: list[ModelPoint]
    metadata: ModelMetadata


class ModelRuntime(Protocol):
    def predict(self, request: PreparedForecast) -> ModelForecast: ...

    def capabilities(self) -> list[ModelCapability]: ...


@dataclass(frozen=True)
class _BacktestObservation:
    actual: float
    prediction: float
    quantiles: dict[float, float]


PointKey = tuple[tuple[tuple[str, object], ...], str, datetime]


class ChronosApplication:
    def __init__(
        self,
        *,
        runtime: ModelRuntime,
        table_reader: TableReader | None = None,
    ) -> None:
        self._runtime = runtime
        self._table_reader = table_reader

    def forecast(self, request: ForecastRequest) -> ForecastSuccess:
        prepared = prepare_forecast(request, table_reader=self._table_reader)
        model_forecast = self._runtime.predict(prepared.forecast)
        rows = _forecast_rows(prepared.forecast, model_forecast)
        warnings = _context_warnings(prepared.forecast.contexts)
        return ForecastSuccess(
            request_id=_request_id("fc", request.request_id),
            rows=rows,
            intervals=_intervals(request.quantiles),
            plot=_forecast_plot(rows) if request.options.include_plot else None,
            warnings=warnings,
            model_meta=_model_meta(prepared.forecast, model_forecast.metadata),
            data_meta=_data_meta(request, prepared.metadata, prepared.forecast),
        )

    def backtest(self, request: BacktestRequest) -> BacktestSuccess:
        prepared = prepare_forecast(request, table_reader=self._table_reader)
        rows: list[BacktestRow] = []
        observations: list[_BacktestObservation] = []
        last_forecast: ModelForecast | None = None
        for window in range(1, request.evaluation.windows + 1):
            window_request, actuals = _backtest_window(prepared.forecast, request, window)
            last_forecast = self._runtime.predict(window_request)
            window_rows, window_observations = _backtest_rows(
                window,
                window_request,
                actuals,
                last_forecast,
            )
            rows.extend(window_rows)
            observations.extend(window_observations)
        if last_forecast is None:
            raise ChronosMCPError(ErrorCode.internal_error, "Backtest produced no windows.")
        return BacktestSuccess(
            request_id=_request_id("bt", request.request_id),
            rows=rows,
            metrics=BacktestMetrics(
                overall=_metrics(observations, prepared.forecast.contexts, request),
                by_series=[],
            ),
            plot=_backtest_plot(rows) if request.options.include_plot else None,
            warnings=_context_warnings(prepared.forecast.contexts),
            model_meta=_model_meta(prepared.forecast, last_forecast.metadata),
            data_meta=_data_meta(request, prepared.metadata, prepared.forecast),
        )

    def capabilities(self) -> CapabilitiesResponse:
        capabilities = self._runtime.capabilities()
        library_version = getattr(self._runtime, "library_version", "unknown")
        return CapabilitiesResponse(
            library_version=str(library_version),
            models=capabilities,
            table_source_available=self._table_reader is not None,
            table_dialects=["postgresql"] if self._table_reader is not None else [],
        )


def error_response(
    error: ChronosMCPError,
    *,
    request_id: str | None,
    prefix: str,
) -> ErrorResponse:
    return ErrorResponse(
        request_id=_request_id(prefix, request_id),
        error=ErrorDetail(
            code=error.code,
            message=str(error),
            field=error.field,
            retryable=error.retryable,
            details=error.details,
        ),
    )


def _request_id(prefix: str, requested: str | None) -> str:
    return requested or f"{prefix}_{uuid.uuid4().hex}"


def _forecast_rows(
    prepared: PreparedForecast,
    forecast: ModelForecast,
) -> list[ForecastRow]:
    points = _point_index(forecast.points)
    rows: list[ForecastRow] = []
    for context in prepared.contexts:
        for timestamp in context.future_timestamps:
            point = points.get(_point_key(context.series_id, context.target, timestamp))
            if point is None:
                raise ChronosMCPError(
                    ErrorCode.model_input_rejected,
                    "Model result does not match requested series, target and horizon.",
                    details={"target": context.target, "timestamp": timestamp.isoformat()},
                )
            quantiles = _finite_quantiles(point.quantiles)
            prediction = _finite_prediction(point.prediction)
            rows.append(
                ForecastRow(
                    series_id=context.series_id,
                    target=context.target,
                    ts=timestamp,
                    prediction=prediction,
                    lower=quantiles[min(quantiles)] if len(quantiles) > 1 else None,
                    upper=quantiles[max(quantiles)] if len(quantiles) > 1 else None,
                    quantiles={str(level): value for level, value in sorted(quantiles.items())},
                )
            )
    return rows


def _point_index(
    points: list[ModelPoint],
) -> dict[PointKey, ModelPoint]:
    indexed: dict[PointKey, ModelPoint] = {}
    for point in points:
        key = _point_key(point.series_id, point.target, point.timestamp)
        if key in indexed:
            raise ChronosMCPError(
                ErrorCode.model_input_rejected,
                "Model returned duplicate forecast points.",
                details={"target": point.target, "timestamp": point.timestamp.isoformat()},
            )
        indexed[key] = point
    return indexed


def _point_key(
    series_id: dict[str, object],
    target: str,
    timestamp: datetime,
) -> PointKey:
    return tuple(sorted(series_id.items())), target, timestamp


def _finite_quantiles(quantiles: dict[float, float]) -> dict[float, float]:
    if not quantiles:
        raise ChronosMCPError(ErrorCode.model_input_rejected, "Model returned no quantiles.")
    return {level: _finite_prediction(value) for level, value in quantiles.items()}


def _finite_prediction(value: float) -> float:
    number = float(value)
    if math.isfinite(number):
        return number
    raise ChronosMCPError(
        ErrorCode.model_input_rejected,
        "Model returned NaN or Infinity.",
    )


def _intervals(quantiles: list[float]) -> list[Interval]:
    if len(quantiles) < 2:
        return []
    lower = min(quantiles)
    upper = max(quantiles)
    return [
        Interval(
            name=f"q{lower:g}_q{upper:g}",
            lower_quantile=lower,
            upper_quantile=upper,
            nominal_coverage=upper - lower,
        )
    ]


def _model_meta(prepared: PreparedForecast, metadata: ModelMetadata) -> ModelMeta:
    return ModelMeta(
        library_version=metadata.library_version,
        model_alias=metadata.model_alias,
        model_id=metadata.model_id,
        model_revision=metadata.model_revision,
        family=metadata.family,
        capabilities=metadata.capabilities,
        context_points=min(len(context.values) for context in prepared.contexts),
        prediction_length=prepared.horizon,
        inference_ms=metadata.inference_ms,
        cached=metadata.cached,
    )


def _data_meta(
    request: ForecastRequest,
    metadata: PreparationMetadata,
    prepared: PreparedForecast,
) -> DataMeta:
    source = request.source
    return DataMeta(
        source_kind=source.kind,
        connection_id=source.connection_id if isinstance(source, TableSource) else None,
        table=f"{source.schema_name}.{source.table}" if isinstance(source, TableSource) else None,
        time_column=request.time_column,
        targets=[target.name for target in request.targets],
        series_count=len(prepared.contexts),
        input_row_count=metadata.input_row_count,
        prepared_point_count=metadata.prepared_point_count,
        history_start=metadata.history_start,
        history_end=metadata.history_end,
        frequency=request.frequency,
        timezone=request.timezone,
        missing_periods=metadata.missing_periods,
        missing_policy=request.missing_policy,
        query=metadata.query,
    )


def _context_warnings(contexts: tuple[PreparedContext, ...]) -> list[WarningMessage]:
    shortest = min(len(context.values) for context in contexts)
    if shortest >= 18:
        return []
    return [
        WarningMessage(
            code="SHORT_CONTEXT",
            message=f"The shortest series contains only {shortest} observations.",
            details={"observations": shortest},
        )
    ]


def _forecast_plot(rows: list[ForecastRow]) -> PlotPayload:
    traces = []
    for label, grouped in _group_forecast_rows(rows).items():
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": label,
                "x": [row.ts.isoformat() for row in grouped],
                "y": [row.prediction for row in grouped],
            }
        )
    return PlotPayload(
        figure={
            "data": traces,
            "layout": {
                "title": {"text": "Chronos forecast"},
                "xaxis": {"title": {"text": "Period"}},
                "yaxis": {"title": {"text": "Forecast"}},
            },
        }
    )


def _group_forecast_rows(rows: list[ForecastRow]) -> dict[str, list[ForecastRow]]:
    groups: dict[str, list[ForecastRow]] = {}
    for row in rows:
        series_label = ", ".join(f"{key}={value}" for key, value in sorted(row.series_id.items()))
        label = f"{row.target} ({series_label})" if series_label else row.target
        groups.setdefault(label, []).append(row)
    return groups


def _backtest_window(
    prepared: PreparedForecast,
    request: BacktestRequest,
    window: int,
) -> tuple[PreparedForecast, dict[tuple[tuple[tuple[str, object], ...], str], tuple[float, ...]]]:
    distance = (request.evaluation.windows - window) * request.evaluation.step
    window_contexts: list[PreparedContext] = []
    actuals = {}
    for context in prepared.contexts:
        context_end = len(context.values) - request.horizon - distance
        if context_end < request.evaluation.min_context_points:
            raise ChronosMCPError(
                ErrorCode.series_too_short,
                "Series is too short for the requested backtest windows.",
                details={
                    "target": context.target,
                    "required_context": request.evaluation.min_context_points,
                    "available_context": max(0, context_end),
                },
            )
        window_context = context.with_window(context_end=context_end, horizon=request.horizon)
        window_contexts.append(window_context)
        key = tuple(sorted(context.series_id.items())), context.target
        actuals[key] = context.values[context_end : context_end + request.horizon]
    return replace(prepared, contexts=tuple(window_contexts)), actuals


def _backtest_rows(
    window: int,
    prepared: PreparedForecast,
    actuals: dict[tuple[tuple[tuple[str, object], ...], str], tuple[float, ...]],
    forecast: ModelForecast,
) -> tuple[list[BacktestRow], list[_BacktestObservation]]:
    points = _point_index(forecast.points)
    rows: list[BacktestRow] = []
    observations: list[_BacktestObservation] = []
    for context in prepared.contexts:
        actual_values = actuals[(tuple(sorted(context.series_id.items())), context.target)]
        cutoff = context.timestamps[-1]
        for timestamp, actual in zip(context.future_timestamps, actual_values, strict=True):
            point = points.get(_point_key(context.series_id, context.target, timestamp))
            if point is None:
                raise ChronosMCPError(
                    ErrorCode.model_input_rejected,
                    "Model result does not match a backtest observation.",
                )
            quantiles = _finite_quantiles(point.quantiles)
            prediction = _finite_prediction(point.prediction)
            rows.append(
                BacktestRow(
                    window=window,
                    cutoff=cutoff,
                    series_id=context.series_id,
                    target=context.target,
                    ts=timestamp,
                    actual=actual,
                    prediction=prediction,
                    lower=quantiles[min(quantiles)] if len(quantiles) > 1 else None,
                    upper=quantiles[max(quantiles)] if len(quantiles) > 1 else None,
                )
            )
            observations.append(
                _BacktestObservation(
                    actual=actual,
                    prediction=prediction,
                    quantiles=quantiles,
                )
            )
    return rows, observations


def _metrics(
    observations: list[_BacktestObservation],
    contexts: tuple[PreparedContext, ...],
    request: BacktestRequest,
) -> dict[str, float | None]:
    actual = [item.actual for item in observations]
    predicted = [item.prediction for item in observations]
    errors = [abs(left - right) for left, right in zip(actual, predicted, strict=True)]
    requested = set(request.evaluation.metrics)
    values: dict[str, float | None] = {}
    if "mae" in requested:
        values["mae"] = sum(errors) / len(errors)
    if "rmse" in requested:
        values["rmse"] = math.sqrt(
            sum((left - right) ** 2 for left, right in zip(actual, predicted, strict=True)) / len(actual)
        )
    if "smape" in requested:
        terms = [
            0.0 if abs(left) + abs(right) == 0 else 2 * abs(left - right) / (abs(left) + abs(right))
            for left, right in zip(actual, predicted, strict=True)
        ]
        values["smape"] = sum(terms) / len(terms)
    if "mase" in requested:
        denominator = _mase_denominator(contexts)
        values["mase"] = None if denominator == 0 else (sum(errors) / len(errors)) / denominator
    if "wql" in requested:
        values["wql"] = _weighted_quantile_loss(observations)
    return values


def _mase_denominator(contexts: tuple[PreparedContext, ...]) -> float:
    naive_errors = [
        abs(current - previous)
        for context in contexts
        for previous, current in zip(context.values, context.values[1:], strict=False)
    ]
    return sum(naive_errors) / len(naive_errors) if naive_errors else 0.0


def _weighted_quantile_loss(observations: list[_BacktestObservation]) -> float | None:
    denominator = sum(abs(item.actual) for item in observations)
    if denominator == 0:
        return None
    losses = []
    for item in observations:
        for quantile, prediction in item.quantiles.items():
            error = item.actual - prediction
            losses.append(2 * max(quantile * error, (quantile - 1) * error))
    return sum(losses) / len(losses) / denominator * len(observations)


def _backtest_plot(rows: list[BacktestRow]) -> PlotPayload:
    return PlotPayload(
        figure={
            "data": [
                {
                    "type": "scatter",
                    "mode": "markers",
                    "name": "Actual",
                    "x": [row.ts.isoformat() for row in rows],
                    "y": [row.actual for row in rows],
                },
                {
                    "type": "scatter",
                    "mode": "markers",
                    "name": "Prediction",
                    "x": [row.ts.isoformat() for row in rows],
                    "y": [row.prediction for row in rows],
                },
            ],
            "layout": {"title": {"text": "Chronos backtest"}},
        }
    )
