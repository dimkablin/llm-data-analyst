from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any

import pandas as pd

from chronos_mcp.contracts import (
    Aggregation,
    ForecastRequest,
    Frequency,
    InlineSource,
    JsonScalar,
    MissingPolicy,
    PointForecast,
    QueryMeta,
    TableSource,
    Target,
)
from chronos_mcp.errors import ChronosMCPError, ErrorCode

MIN_CONTEXT_POINTS = 2

_PERIOD_SPECS: dict[Frequency, tuple[str, str, str]] = {
    Frequency.minute: ("min", "start", "min"),
    Frequency.hour: ("h", "start", "h"),
    Frequency.day: ("D", "start", "D"),
    Frequency.week_monday: ("W-SUN", "start", "W-MON"),
    Frequency.week_sunday: ("W-SAT", "start", "W-SUN"),
    Frequency.month_start: ("M", "start", "MS"),
    Frequency.month_end: ("M", "end", "ME"),
    Frequency.quarter_start: ("Q-DEC", "start", "QS-JAN"),
    Frequency.quarter_end: ("Q-DEC", "end", "QE-DEC"),
    Frequency.year_start: ("Y-DEC", "start", "YS-JAN"),
    Frequency.year_end: ("Y-DEC", "end", "YE-DEC"),
}


@dataclass(frozen=True)
class TableReadPlan:
    connection_id: str
    schema_name: str
    table: str
    columns: tuple[str, ...]
    filter_payload: dict[str, Any] | None
    history_start: str | None
    history_end: str | None
    horizon: int
    frequency: Frequency
    future_columns: tuple[str, ...]
    max_rows: int


@dataclass(frozen=True)
class TableReadResult:
    rows: list[dict[str, JsonScalar]]
    query: QueryMeta | None = None


TableReader = Callable[[TableReadPlan], TableReadResult]


@dataclass(frozen=True)
class PreparedContext:
    series_id: dict[str, JsonScalar]
    target: str
    timestamps: tuple[datetime, ...]
    values: tuple[float, ...]
    future_timestamps: tuple[datetime, ...]
    past_covariates: dict[str, tuple[JsonScalar, ...]]
    future_covariates: dict[str, tuple[JsonScalar, ...]]

    def with_window(self, *, context_end: int, horizon: int) -> PreparedContext:
        return replace(
            self,
            timestamps=self.timestamps[:context_end],
            values=self.values[:context_end],
            future_timestamps=self.timestamps[context_end : context_end + horizon],
            past_covariates={name: values[:context_end] for name, values in self.past_covariates.items()},
            future_covariates={
                name: values[context_end : context_end + horizon]
                for name, values in self.past_covariates.items()
            },
        )


@dataclass(frozen=True)
class PreparedForecast:
    contexts: tuple[PreparedContext, ...]
    horizon: int
    quantiles: tuple[float, ...]
    point_forecast: PointForecast
    cross_learning: bool
    model_alias: str | None


@dataclass(frozen=True)
class PreparationMetadata:
    input_row_count: int
    prepared_point_count: int
    missing_periods: int
    history_start: datetime
    history_end: datetime
    query: QueryMeta | None


@dataclass(frozen=True)
class PreparationResult:
    forecast: PreparedForecast
    metadata: PreparationMetadata


def prepare_forecast(
    request: ForecastRequest,
    *,
    table_reader: TableReader | None,
) -> PreparationResult:
    rows, query = _read_rows(request, table_reader)
    if not rows:
        raise ChronosMCPError(ErrorCode.series_empty, "Source returned no rows.")
    _reject_non_finite_rows(rows)

    frame = pd.DataFrame(rows)
    _require_columns(frame, request)
    frame["_timestamp"] = [
        _parse_timestamp(value, request.timezone, request.time_column) for value in frame[request.time_column]
    ]
    frame = _apply_history_start(frame, request)

    contexts: list[PreparedContext] = []
    missing_periods = 0
    for series_id, group in _series_groups(frame, request.series_id_columns):
        for target in request.targets:
            context, target_missing = _prepare_context(group, series_id, target, request)
            contexts.append(context)
            missing_periods += target_missing

    if not contexts:
        raise ChronosMCPError(ErrorCode.series_empty, "No time series remained after preparation.")

    prepared_points = sum(len(context.values) for context in contexts)
    history_start = min(context.timestamps[0] for context in contexts)
    history_end = max(context.timestamps[-1] for context in contexts)
    return PreparationResult(
        forecast=PreparedForecast(
            contexts=tuple(contexts),
            horizon=request.horizon,
            quantiles=tuple(request.quantiles),
            point_forecast=request.options.point_forecast,
            cross_learning=request.options.cross_learning,
            model_alias=request.model_alias,
        ),
        metadata=PreparationMetadata(
            input_row_count=len(rows),
            prepared_point_count=prepared_points,
            missing_periods=missing_periods,
            history_start=history_start,
            history_end=history_end,
            query=query,
        ),
    )


def _reject_non_finite_rows(rows: list[dict[str, JsonScalar]]) -> None:
    for row_index, row in enumerate(rows):
        for column, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ChronosMCPError(
                    ErrorCode.non_finite_value,
                    f"Column {column} contains NaN or Infinity.",
                    field=f"source.rows[{row_index}].{column}",
                )


def _read_rows(
    request: ForecastRequest,
    table_reader: TableReader | None,
) -> tuple[list[dict[str, JsonScalar]], QueryMeta | None]:
    if isinstance(request.source, InlineSource):
        return request.source.rows, None
    if table_reader is None:
        raise ChronosMCPError(
            ErrorCode.source_unavailable,
            "Table source is not configured; use inline rows or configure the data gateway.",
            field="source",
            retryable=True,
        )
    return _read_table(request, request.source, table_reader)


def _read_table(
    request: ForecastRequest,
    source: TableSource,
    table_reader: TableReader,
) -> tuple[list[dict[str, JsonScalar]], QueryMeta | None]:
    columns = {
        request.time_column,
        *request.series_id_columns,
        *(target.column for target in request.targets if target.column),
    }
    if request.covariates:
        columns.update(request.covariates.past_columns)
        columns.update(request.covariates.future_columns)
    filter_payload = request.filter.model_dump(mode="json") if request.filter else None
    read_result = table_reader(
        TableReadPlan(
            connection_id=source.connection_id,
            schema_name=source.schema_name,
            table=source.table,
            columns=tuple(sorted(columns)),
            filter_payload=filter_payload,
            history_start=_iso_value(request.history_start),
            history_end=_iso_value(request.history_end),
            horizon=request.horizon,
            frequency=request.frequency,
            future_columns=tuple(request.covariates.future_columns if request.covariates else []),
            max_rows=50_000,
        )
    )
    return read_result.rows, read_result.query


def _iso_value(value: date | datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _require_columns(frame: pd.DataFrame, request: ForecastRequest) -> None:
    required = {
        request.time_column,
        *request.series_id_columns,
        *(target.column for target in request.targets if target.column),
    }
    if request.covariates:
        required.update(request.covariates.past_columns)
        required.update(request.covariates.future_columns)
    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        raise ChronosMCPError(
            ErrorCode.column_not_found,
            f"Source is missing required columns: {', '.join(missing)}.",
            details={"columns": missing},
        )


def _parse_timestamp(value: object, timezone: str, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ChronosMCPError(
            ErrorCode.column_type_mismatch,
            f"{field} contains an invalid timestamp.",
            field=field,
        ) from exc
    if pd.isna(timestamp):
        raise ChronosMCPError(
            ErrorCode.column_type_mismatch,
            f"{field} contains a null timestamp.",
            field=field,
        )
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def _apply_history_start(frame: pd.DataFrame, request: ForecastRequest) -> pd.DataFrame:
    if request.history_start is None:
        return frame
    start = _parse_timestamp(request.history_start, request.timezone, "history_start")
    filtered = frame.loc[frame["_timestamp"] >= start].copy()
    if filtered.empty:
        raise ChronosMCPError(ErrorCode.series_empty, "No rows are on or after history_start.")
    return filtered


def _series_groups(
    frame: pd.DataFrame,
    columns: list[str],
) -> list[tuple[dict[str, JsonScalar], pd.DataFrame]]:
    if not columns:
        return [({}, frame)]
    group_key: str | list[str] = columns[0] if len(columns) == 1 else columns
    groups: list[tuple[dict[str, JsonScalar], pd.DataFrame]] = []
    for raw_key, group in frame.groupby(group_key, dropna=False, sort=True):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        series_id = {column: _json_scalar(value) for column, value in zip(columns, values, strict=True)}
        groups.append((series_id, group))
    return groups


def _json_scalar(value: object) -> JsonScalar:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _prepare_context(
    group: pd.DataFrame,
    series_id: dict[str, JsonScalar],
    target: Target,
    request: ForecastRequest,
) -> tuple[PreparedContext, int]:
    history = _history_rows(group, target, request)
    observations = _aggregate_target(history, target, request)
    if not observations:
        raise ChronosMCPError(
            ErrorCode.series_empty,
            f"Target {target.name} has no historical observations.",
            details={"series_id": series_id, "target": target.name},
        )

    full_index = _full_index(min(observations), max(observations), request)
    values = [observations.get(timestamp) for timestamp in full_index]
    missing_count = sum(value is None for value in values)
    filled = _fill_missing(values, request.missing_policy, target.name)
    max_points = request.options.max_history_points
    if max_points and len(filled) > max_points:
        full_index = full_index[-max_points:]
        filled = filled[-max_points:]
    if len(filled) < MIN_CONTEXT_POINTS:
        raise ChronosMCPError(
            ErrorCode.series_too_short,
            f"Target {target.name} needs at least {MIN_CONTEXT_POINTS} observations.",
            details={"observations": len(filled), "target": target.name},
        )

    future_timestamps = _future_index(full_index[-1], request)
    past_covariates, future_covariates = _prepare_covariates(
        group,
        full_index,
        future_timestamps,
        request,
    )
    return (
        PreparedContext(
            series_id=series_id,
            target=target.name,
            timestamps=tuple(_as_datetime(value) for value in full_index),
            values=tuple(filled),
            future_timestamps=tuple(_as_datetime(value) for value in future_timestamps),
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        ),
        missing_count,
    )


def _history_rows(
    group: pd.DataFrame,
    target: Target,
    request: ForecastRequest,
) -> pd.DataFrame:
    history = group
    if request.history_end is not None:
        end = _parse_timestamp(request.history_end, request.timezone, "history_end")
        history = history.loc[history["_timestamp"] <= end]
    if target.column is not None:
        history = history.loc[history[target.column].notna()]
    return history.sort_values("_timestamp")


def _aggregate_target(
    history: pd.DataFrame,
    target: Target,
    request: ForecastRequest,
) -> dict[pd.Timestamp, float]:
    buckets: dict[pd.Timestamp, list[tuple[pd.Timestamp, object]]] = {}
    for _, row in history.iterrows():
        bucket = _bucket_timestamp(row["_timestamp"], request)
        raw_value = row[target.column] if target.column else 1
        buckets.setdefault(bucket, []).append((row["_timestamp"], raw_value))
    return {timestamp: _aggregate_values(values, target) for timestamp, values in buckets.items()}


def _aggregate_values(
    timestamped_values: list[tuple[pd.Timestamp, object]],
    target: Target,
) -> float:
    if target.aggregation == Aggregation.count:
        return float(sum(value is not None and not pd.isna(value) for _, value in timestamped_values))
    if target.aggregation == Aggregation.count_distinct:
        return float(len({_hashable(value) for _, value in timestamped_values if not pd.isna(value)}))

    values = [_finite_number(value, target) for _, value in timestamped_values]
    if target.aggregation == Aggregation.none:
        if len(values) != 1:
            raise ChronosMCPError(
                ErrorCode.duplicate_period,
                f"Target {target.name} has duplicate observations in one period.",
                details={"target": target.name},
            )
        return values[0]
    if target.aggregation == Aggregation.sum:
        return float(sum(values))
    if target.aggregation == Aggregation.mean:
        return float(sum(values) / len(values))
    if target.aggregation == Aggregation.median:
        return float(pd.Series(values).median())
    if target.aggregation == Aggregation.min:
        return min(values)
    if target.aggregation == Aggregation.max:
        return max(values)
    if target.aggregation == Aggregation.last:
        return _finite_number(max(timestamped_values, key=lambda item: item[0])[1], target)
    raise ChronosMCPError(ErrorCode.invalid_argument, f"Unsupported aggregation: {target.aggregation}.")


def _hashable(value: object) -> object:
    try:
        hash(value)
    except TypeError:
        return str(value)
    return value


def _finite_number(value: object, target: Target) -> float:
    if isinstance(value, bool):
        raise ChronosMCPError(
            ErrorCode.column_type_mismatch,
            f"Target {target.name} must be numeric, not boolean.",
            field=target.column,
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ChronosMCPError(
            ErrorCode.column_type_mismatch,
            f"Target {target.name} must contain numeric values.",
            field=target.column,
        ) from exc
    if not math.isfinite(number):
        raise ChronosMCPError(
            ErrorCode.non_finite_value,
            f"Target {target.name} contains NaN or Infinity.",
            field=target.column,
        )
    return number


def _bucket_timestamp(timestamp: pd.Timestamp, request: ForecastRequest) -> pd.Timestamp:
    period_alias, boundary, _ = _PERIOD_SPECS[request.frequency]
    local = timestamp.tz_convert(request.timezone).tz_localize(None)
    period = local.to_period(period_alias)
    boundary_value = period.start_time if boundary == "start" else period.end_time.normalize()
    return boundary_value.tz_localize(request.timezone)


def _full_index(
    start: pd.Timestamp,
    end: pd.Timestamp,
    request: ForecastRequest,
) -> list[pd.Timestamp]:
    _, _, range_alias = _PERIOD_SPECS[request.frequency]
    return list(pd.date_range(start=start, end=end, freq=range_alias, tz=request.timezone))


def _future_index(last: pd.Timestamp, request: ForecastRequest) -> list[pd.Timestamp]:
    _, _, range_alias = _PERIOD_SPECS[request.frequency]
    return list(
        pd.date_range(
            start=last,
            periods=request.horizon + 1,
            freq=range_alias,
            tz=request.timezone,
        )[1:]
    )


def _fill_missing(
    values: list[float | None],
    policy: MissingPolicy,
    target: str,
) -> list[float]:
    if all(value is not None for value in values):
        return [float(value) for value in values if value is not None]
    if policy == MissingPolicy.error:
        raise ChronosMCPError(
            ErrorCode.missing_periods,
            f"Target {target} has missing periods.",
            details={"target": target, "missing_periods": sum(value is None for value in values)},
        )
    series = pd.Series(values, dtype="float64")
    if policy == MissingPolicy.zero:
        series = series.fillna(0.0)
    elif policy == MissingPolicy.forward_fill:
        series = series.ffill()
    elif policy == MissingPolicy.interpolate:
        series = series.interpolate(limit_area="inside")
    if series.isna().any():
        raise ChronosMCPError(
            ErrorCode.missing_periods,
            f"Policy {policy} could not fill every missing period for {target}.",
        )
    return [float(value) for value in series]


def _prepare_covariates(
    group: pd.DataFrame,
    history_index: list[pd.Timestamp],
    future_index: list[pd.Timestamp],
    request: ForecastRequest,
) -> tuple[dict[str, tuple[JsonScalar, ...]], dict[str, tuple[JsonScalar, ...]]]:
    if request.covariates is None:
        return {}, {}
    by_bucket = _rows_by_bucket(group, request)
    history_columns = [
        *request.covariates.past_columns,
        *request.covariates.future_columns,
    ]
    past = {
        column: tuple(
            _covariate_value(by_bucket, timestamp, column, future=False) for timestamp in history_index
        )
        for column in history_columns
    }
    future = {
        column: tuple(
            _covariate_value(by_bucket, timestamp, column, future=True) for timestamp in future_index
        )
        for column in request.covariates.future_columns
    }
    return past, future


def _rows_by_bucket(
    group: pd.DataFrame,
    request: ForecastRequest,
) -> dict[pd.Timestamp, pd.Series]:
    rows: dict[pd.Timestamp, pd.Series] = {}
    for _, row in group.sort_values("_timestamp").iterrows():
        rows[_bucket_timestamp(row["_timestamp"], request)] = row
    return rows


def _covariate_value(
    rows: dict[pd.Timestamp, pd.Series],
    timestamp: pd.Timestamp,
    column: str,
    *,
    future: bool,
) -> JsonScalar:
    row = rows.get(timestamp)
    value = None if row is None else row[column]
    if value is None or pd.isna(value):
        code = ErrorCode.future_covariates_incomplete if future else ErrorCode.missing_periods
        raise ChronosMCPError(
            code,
            f"Covariate {column} is missing for {timestamp.isoformat()}.",
            field=column,
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ChronosMCPError(
            ErrorCode.non_finite_value,
            f"Covariate {column} contains NaN or Infinity.",
            field=column,
        )
    return _json_scalar(value)


def _as_datetime(value: pd.Timestamp) -> datetime:
    return value.to_pydatetime()
