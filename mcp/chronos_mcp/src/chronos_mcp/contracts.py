from __future__ import annotations

import math
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

JsonScalar = str | int | float | bool | None
FiniteQuantile = Annotated[float, Field(gt=0.0, lt=1.0, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Frequency(StrEnum):
    minute = "minute"
    hour = "hour"
    day = "day"
    week_monday = "week_monday"
    week_sunday = "week_sunday"
    month_start = "month_start"
    month_end = "month_end"
    quarter_start = "quarter_start"
    quarter_end = "quarter_end"
    year_start = "year_start"
    year_end = "year_end"


class MissingPolicy(StrEnum):
    error = "error"
    zero = "zero"
    forward_fill = "forward_fill"
    interpolate = "interpolate"


class Aggregation(StrEnum):
    none = "none"
    sum = "sum"
    mean = "mean"
    median = "median"
    min = "min"
    max = "max"
    count = "count"
    count_distinct = "count_distinct"
    last = "last"


class PointForecast(StrEnum):
    median = "median"
    mean = "mean"


class MetricName(StrEnum):
    mae = "mae"
    rmse = "rmse"
    smape = "smape"
    mase = "mase"
    wql = "wql"


class InlineSource(StrictModel):
    kind: Literal["inline"] = "inline"
    rows: list[dict[str, JsonScalar]] = Field(min_length=2, max_length=50_000)


class TableSource(StrictModel):
    kind: Literal["table"] = "table"
    connection_id: str = Field(min_length=1, max_length=128)
    schema_name: str = Field(alias="schema", min_length=1, max_length=128)
    table: str = Field(min_length=1, max_length=128)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


Source = Annotated[InlineSource | TableSource, Field(discriminator="kind")]


class Target(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    column: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description=(
            "Exact source row field containing the target values; required for "
            "every aggregation except count."
        ),
        examples=["y"],
    )
    aggregation: Aggregation

    @model_validator(mode="after")
    def require_column_unless_counting_rows(self) -> Target:
        if self.aggregation != Aggregation.count and self.column is None:
            raise ValueError("column is required unless aggregation=count")
        return self


class PredicateOp(StrEnum):
    eq = "eq"
    ne = "ne"
    gt = "gt"
    gte = "gte"
    lt = "lt"
    lte = "lte"
    in_ = "in"
    not_in = "not_in"
    between = "between"
    is_null = "is_null"
    not_null = "not_null"


class LogicalOp(StrEnum):
    and_ = "and"
    or_ = "or"
    not_ = "not"


class FilterPredicate(StrictModel):
    column: str = Field(min_length=1, max_length=128)
    op: PredicateOp
    value: JsonScalar = None
    values: list[JsonScalar] | None = Field(default=None, max_length=100)
    lower: JsonScalar = None
    upper: JsonScalar = None

    @model_validator(mode="after")
    def validate_operands(self) -> FilterPredicate:
        scalar_ops = {
            PredicateOp.eq,
            PredicateOp.ne,
            PredicateOp.gt,
            PredicateOp.gte,
            PredicateOp.lt,
            PredicateOp.lte,
        }
        if self.op in scalar_ops and self.value is None:
            raise ValueError(f"{self.op} requires value")
        if self.op in {PredicateOp.in_, PredicateOp.not_in} and not self.values:
            raise ValueError(f"{self.op} requires non-empty values")
        if self.op == PredicateOp.between and (self.lower is None or self.upper is None):
            raise ValueError("between requires lower and upper")
        for operand in [self.value, self.lower, self.upper, *(self.values or [])]:
            if isinstance(operand, float) and not math.isfinite(operand):
                raise ValueError("filter operands must be finite")
        return self


class FilterGroup(StrictModel):
    op: LogicalOp
    args: list[FilterExpression] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_not_arity(self) -> FilterGroup:
        if self.op == LogicalOp.not_ and len(self.args) != 1:
            raise ValueError("not requires exactly one argument")
        return self


FilterExpression = FilterPredicate | FilterGroup
FilterGroup.model_rebuild()


class Covariates(StrictModel):
    past_columns: list[str] = Field(default_factory=list, max_length=32)
    future_columns: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def keep_column_roles_distinct(self) -> Covariates:
        overlap = set(self.past_columns) & set(self.future_columns)
        if overlap:
            raise ValueError(f"covariates cannot be both past and future: {sorted(overlap)}")
        return self


class ForecastOptions(StrictModel):
    point_forecast: PointForecast = PointForecast.median
    include_plot: bool = True
    max_history_points: int | None = Field(default=None, ge=2, le=100_000)
    cross_learning: bool = False


class ForecastRequest(StrictModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    source: Source
    time_column: str = Field(min_length=1, max_length=128)
    targets: list[Target] = Field(min_length=1, max_length=16)
    series_id_columns: list[str] = Field(default_factory=list, max_length=8)
    filter: FilterExpression | None = None
    history_start: date | datetime | None = None
    history_end: date | datetime | None = None
    horizon: int = Field(ge=1, le=1024)
    frequency: Frequency
    timezone: str = "UTC"
    missing_policy: MissingPolicy
    quantiles: list[FiniteQuantile] = Field(
        default_factory=lambda: [0.1, 0.5, 0.9],
        min_length=1,
        max_length=19,
    )
    model_alias: str | None = Field(default=None, min_length=1, max_length=128)
    covariates: Covariates | None = None
    options: ForecastOptions = Field(default_factory=ForecastOptions)

    @field_validator("quantiles")
    @classmethod
    def sort_unique_quantiles(cls, values: list[float]) -> list[float]:
        return sorted(set(values))

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value

    @model_validator(mode="after")
    def validate_source_specific_fields(self) -> ForecastRequest:
        if isinstance(self.source, InlineSource) and self.filter is not None:
            raise ValueError("filter is only supported for table source")
        if self.history_start and self.history_end and self.history_start > self.history_end:
            raise ValueError("history_start must not be after history_end")
        return self


class BacktestEvaluation(StrictModel):
    windows: int = Field(default=3, ge=1, le=20)
    step: int = Field(default=1, ge=1, le=1024)
    min_context_points: int = Field(default=2, ge=2, le=100_000)
    metrics: list[MetricName] = Field(
        default_factory=lambda: list(MetricName),
        min_length=1,
    )


class BacktestRequest(ForecastRequest):
    evaluation: BacktestEvaluation = Field(default_factory=BacktestEvaluation)


class WarningMessage(StrictModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ForecastRow(StrictModel):
    series_id: dict[str, JsonScalar]
    target: str
    ts: datetime
    prediction: float
    lower: float | None
    upper: float | None
    quantiles: dict[str, float]


class Interval(StrictModel):
    name: str
    lower_quantile: float
    upper_quantile: float
    nominal_coverage: float


class PlotPayload(StrictModel):
    format: Literal["plotly"] = "plotly"
    figure: dict[str, Any]


class ModelMeta(StrictModel):
    library: Literal["chronos-forecasting"] = "chronos-forecasting"
    library_version: str
    model_alias: str
    model_id: str
    model_revision: str
    family: str
    capabilities: list[str]
    context_points: int
    prediction_length: int
    inference_ms: int
    cached: bool


class QueryMeta(StrictModel):
    sql: str | None = None
    parameter_count: int = 0
    fingerprint: str | None = None


class DataMeta(StrictModel):
    source_kind: Literal["inline", "table"]
    connection_id: str | None
    table: str | None
    time_column: str
    targets: list[str]
    series_count: int
    input_row_count: int
    prepared_point_count: int
    history_start: datetime
    history_end: datetime
    frequency: Frequency
    timezone: str
    missing_periods: int
    missing_policy: MissingPolicy
    query: QueryMeta | None


class ForecastSuccess(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["ok"] = "ok"
    request_id: str
    rows: list[ForecastRow]
    intervals: list[Interval]
    plot: PlotPayload | None
    warnings: list[WarningMessage]
    model_meta: ModelMeta
    data_meta: DataMeta


class ErrorDetail(StrictModel):
    code: str
    message: str
    field: str | None
    retryable: bool
    details: dict[str, Any]


class ErrorResponse(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["error"] = "error"
    request_id: str
    error: ErrorDetail


class ForecastOutput(RootModel[ForecastSuccess | ErrorResponse]):
    pass


class BacktestRow(StrictModel):
    window: int
    cutoff: datetime
    series_id: dict[str, JsonScalar]
    target: str
    ts: datetime
    actual: float
    prediction: float
    lower: float | None
    upper: float | None


class BacktestMetrics(StrictModel):
    overall: dict[str, float | None]
    by_series: list[dict[str, Any]]


class BacktestSuccess(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["ok"] = "ok"
    request_id: str
    rows: list[BacktestRow]
    metrics: BacktestMetrics
    plot: PlotPayload | None
    warnings: list[WarningMessage]
    model_meta: ModelMeta
    data_meta: DataMeta


class BacktestOutput(RootModel[BacktestSuccess | ErrorResponse]):
    pass


class ModelCapability(StrictModel):
    alias: str
    model_id: str
    revision: str
    family: str
    capabilities: list[str]
    max_horizon: int
    max_context_points: int
    state: str


class CapabilitiesResponse(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["ok"] = "ok"
    library_name: Literal["chronos-forecasting"] = "chronos-forecasting"
    library_version: str
    models: list[ModelCapability]
    inline_source_available: bool = True
    table_source_available: bool
    table_dialects: list[str]
    max_inline_rows: int = 50_000
    max_result_rows: int = 50_000
