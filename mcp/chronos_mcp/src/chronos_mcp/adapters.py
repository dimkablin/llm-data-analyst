from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from chronos_mcp.application import ModelForecast, ModelMetadata, ModelPoint
from chronos_mcp.contracts import ModelCapability, PointForecast, QueryMeta
from chronos_mcp.errors import ChronosMCPError, ErrorCode
from chronos_mcp.preparation import (
    PreparedContext,
    PreparedForecast,
    TableReadPlan,
    TableReadResult,
)


@dataclass(frozen=True)
class ChronosSettings:
    model_alias: str = "chronos2-default"
    model_id: str = "amazon/chronos-2"
    model_revision: str = "main"
    family: str = "chronos2"
    device: str = "auto"
    dtype: str = "auto"
    max_horizon: int = 256
    max_context_points: int = 8192

    @classmethod
    def from_env(cls) -> ChronosSettings:
        return cls(
            model_alias=os.getenv("CHRONOS_MODEL_ALIAS", cls.model_alias),
            model_id=os.getenv("CHRONOS_MODEL_ID", cls.model_id),
            model_revision=os.getenv("CHRONOS_MODEL_REVISION", cls.model_revision),
            family=os.getenv("CHRONOS_MODEL_FAMILY", cls.family),
            device=os.getenv("CHRONOS_DEVICE", cls.device),
            dtype=os.getenv("CHRONOS_DTYPE", cls.dtype),
            max_horizon=_positive_int("CHRONOS_MAX_HORIZON", cls.max_horizon),
            max_context_points=_positive_int(
                "CHRONOS_MAX_CONTEXT_POINTS",
                cls.max_context_points,
            ),
        )


@dataclass(frozen=True)
class DataGatewaySettings:
    base_url: str
    token: str | None
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> DataGatewaySettings | None:
        base_url = os.getenv("CHRONOS_DATA_GATEWAY_URL", "").strip()
        if not base_url:
            return None
        return cls(
            base_url=base_url.rstrip("/"),
            token=os.getenv("CHRONOS_DATA_GATEWAY_TOKEN") or None,
            timeout_seconds=_positive_float("CHRONOS_QUERY_TIMEOUT_SEC", 30.0),
        )


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc


def _positive_float(name: str, default: float) -> float:
    try:
        return max(0.1, float(os.getenv(name, str(default))))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc


class ChronosRuntime:
    def __init__(
        self,
        settings: ChronosSettings,
        *,
        pipeline_loader: Callable[[ChronosSettings], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._pipeline_loader = pipeline_loader or _load_pipeline

    @property
    def library_version(self) -> str:
        try:
            return version("chronos-forecasting")
        except PackageNotFoundError:
            return "not-installed"

    def capabilities(self) -> list[ModelCapability]:
        capabilities = ["univariate", "batch", "quantiles"]
        if self._settings.family == "chronos2":
            capabilities.extend(["multitarget", "covariates", "cross_learning"])
        return [
            ModelCapability(
                alias=self._settings.model_alias,
                model_id=self._settings.model_id,
                revision=self._settings.model_revision,
                family=self._settings.family,
                capabilities=capabilities,
                max_horizon=self._settings.max_horizon,
                max_context_points=self._settings.max_context_points,
                state="ready" if "_pipeline" in self.__dict__ else "not_loaded",
            )
        ]

    def predict(self, request: PreparedForecast) -> ModelForecast:
        self._validate_request(request)
        pipeline = self._pipeline
        context_frame, future_frame, contexts_by_id = _model_frames(request)
        started = time.perf_counter()
        prediction_frame = self._predict_frame(
            pipeline,
            context_frame,
            future_frame,
            request,
        )
        points = _model_points(prediction_frame, contexts_by_id, request)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return ModelForecast(
            points=points,
            metadata=ModelMetadata(
                library_version=self.library_version,
                model_alias=self._settings.model_alias,
                model_id=self._settings.model_id,
                model_revision=self._settings.model_revision,
                family=self._settings.family,
                capabilities=self.capabilities()[0].capabilities,
                inference_ms=elapsed_ms,
                cached=True,
            ),
        )

    def _validate_request(self, request: PreparedForecast) -> None:
        if request.model_alias and request.model_alias != self._settings.model_alias:
            raise ChronosMCPError(
                ErrorCode.model_not_available,
                f"Model alias is not available: {request.model_alias}.",
                field="model_alias",
            )
        if request.horizon > self._settings.max_horizon:
            raise ChronosMCPError(
                ErrorCode.invalid_argument,
                f"horizon exceeds model limit {self._settings.max_horizon}.",
                field="horizon",
            )
        if request.cross_learning and self._settings.family != "chronos2":
            raise ChronosMCPError(
                ErrorCode.model_capability_mismatch,
                "cross_learning requires Chronos-2.",
                field="options.cross_learning",
            )
        if any(context.future_covariates for context in request.contexts):
            if self._settings.family != "chronos2":
                raise ChronosMCPError(
                    ErrorCode.model_capability_mismatch,
                    "Future covariates require Chronos-2.",
                    field="covariates.future_columns",
                )

    @cached_property
    def _pipeline(self) -> Any:
        return self._pipeline_loader(self._settings)

    @staticmethod
    def _predict_frame(
        pipeline: Any,
        context_frame: pd.DataFrame,
        future_frame: pd.DataFrame | None,
        request: PreparedForecast,
    ) -> pd.DataFrame:
        arguments: dict[str, Any] = {
            "prediction_length": request.horizon,
            "quantile_levels": list(request.quantiles),
            "id_column": "id",
            "timestamp_column": "timestamp",
            "target": "target",
        }
        if future_frame is not None:
            arguments["future_df"] = future_frame
        if request.cross_learning:
            arguments["cross_learning"] = True
        try:
            prediction = pipeline.predict_df(context_frame, **arguments)
        except ValueError as exc:
            raise ChronosMCPError(
                ErrorCode.model_input_rejected,
                f"Chronos rejected the prepared series: {exc}",
            ) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise ChronosMCPError(
                    ErrorCode.resource_exhausted,
                    "Chronos ran out of model memory.",
                    retryable=True,
                ) from exc
            raise ChronosMCPError(
                ErrorCode.model_input_rejected,
                f"Chronos inference failed: {exc}",
            ) from exc
        if not isinstance(prediction, pd.DataFrame):
            raise ChronosMCPError(
                ErrorCode.model_input_rejected,
                "Chronos returned a non-DataFrame result.",
            )
        return prediction


def _load_pipeline(settings: ChronosSettings) -> Any:
    try:
        from chronos import BaseChronosPipeline
    except ImportError as exc:
        raise ChronosMCPError(
            ErrorCode.model_not_available,
            "chronos-forecasting is not installed; install the 'chronos' extra.",
        ) from exc

    arguments: dict[str, Any] = {"revision": settings.model_revision}
    if settings.device != "auto":
        arguments["device_map"] = settings.device
    if settings.dtype != "auto":
        arguments["torch_dtype"] = settings.dtype
    try:
        return BaseChronosPipeline.from_pretrained(settings.model_id, **arguments)
    except OSError as exc:
        raise ChronosMCPError(
            ErrorCode.model_not_available,
            f"Could not load Chronos model {settings.model_id}: {exc}",
            retryable=True,
        ) from exc


def _model_frames(
    request: PreparedForecast,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, PreparedContext]]:
    context_rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    contexts_by_id: dict[str, PreparedContext] = {}
    for index, context in enumerate(request.contexts):
        item_id = f"series-{index}"
        contexts_by_id[item_id] = context
        context_rows.extend(_context_rows(item_id, context))
        future_rows.extend(_future_rows(item_id, context))
    context_frame = pd.DataFrame(context_rows)
    context_frame["timestamp"] = pd.to_datetime(context_frame["timestamp"], utc=True).dt.tz_localize(None)
    future_frame = pd.DataFrame(future_rows) if future_rows else None
    if future_frame is not None:
        future_frame["timestamp"] = pd.to_datetime(future_frame["timestamp"], utc=True).dt.tz_localize(None)
    return context_frame, future_frame, contexts_by_id


def _context_rows(item_id: str, context: PreparedContext) -> list[dict[str, Any]]:
    rows = []
    for index, (timestamp, value) in enumerate(zip(context.timestamps, context.values, strict=True)):
        row = {"id": item_id, "timestamp": timestamp, "target": value}
        for name, values in context.past_covariates.items():
            row[name] = values[index]
        rows.append(row)
    return rows


def _future_rows(item_id: str, context: PreparedContext) -> list[dict[str, Any]]:
    if not context.future_covariates:
        return []
    rows = []
    for index, timestamp in enumerate(context.future_timestamps):
        row = {"id": item_id, "timestamp": timestamp}
        for name, values in context.future_covariates.items():
            row[name] = values[index]
        rows.append(row)
    return rows


def _model_points(
    prediction_frame: pd.DataFrame,
    contexts_by_id: dict[str, PreparedContext],
    request: PreparedForecast,
) -> list[ModelPoint]:
    required = {"id"}
    missing = required - set(prediction_frame.columns)
    if missing:
        raise ChronosMCPError(
            ErrorCode.model_input_rejected,
            f"Chronos result is missing columns: {sorted(missing)}.",
        )
    points: list[ModelPoint] = []
    for item_id, context in contexts_by_id.items():
        rows = prediction_frame.loc[prediction_frame["id"].astype(str) == item_id]
        if "timestamp" in rows.columns:
            rows = rows.sort_values("timestamp")
        if len(rows) != request.horizon:
            raise ChronosMCPError(
                ErrorCode.model_input_rejected,
                f"Chronos returned {len(rows)} rows for {item_id}; expected {request.horizon}.",
            )
        for timestamp, (_, row) in zip(context.future_timestamps, rows.iterrows(), strict=True):
            quantiles = {level: _row_number(row, str(level)) for level in request.quantiles}
            prediction = _point_value(row, quantiles, request.point_forecast)
            points.append(
                ModelPoint(
                    series_id=context.series_id,
                    target=context.target,
                    timestamp=timestamp,
                    prediction=prediction,
                    quantiles=quantiles,
                )
            )
    return points


def _row_number(row: pd.Series, column: str) -> float:
    if column not in row.index:
        raise ChronosMCPError(
            ErrorCode.model_input_rejected,
            f"Chronos result is missing quantile column {column}.",
        )
    try:
        return float(row[column])
    except (TypeError, ValueError) as exc:
        raise ChronosMCPError(
            ErrorCode.model_input_rejected,
            f"Chronos returned a non-numeric value in column {column}.",
        ) from exc


def _point_value(
    row: pd.Series,
    quantiles: dict[float, float],
    point_forecast: PointForecast,
) -> float:
    if point_forecast == PointForecast.median:
        if 0.5 in quantiles:
            return quantiles[0.5]
        ordered = sorted(quantiles)
        return quantiles[ordered[len(ordered) // 2]]
    for column in ("predictions", "mean"):
        if column in row.index:
            return _row_number(row, column)
    raise ChronosMCPError(
        ErrorCode.model_input_rejected,
        "Chronos result does not contain a mean prediction column.",
    )


class HttpDataGateway:
    def __init__(
        self,
        settings: DataGatewaySettings,
        *,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self._settings = settings
        self._opener = opener

    def __call__(self, plan: TableReadPlan) -> TableReadResult:
        payload = {
            "connection_id": plan.connection_id,
            "schema": plan.schema_name,
            "table": plan.table,
            "columns": list(plan.columns),
            "filter": plan.filter_payload,
            "history_start": plan.history_start,
            "history_end": plan.history_end,
            "horizon": plan.horizon,
            "frequency": plan.frequency,
            "future_columns": list(plan.future_columns),
            "max_rows": plan.max_rows,
        }
        response = self._post_json("/v1/chronos/rows", payload)
        rows = response.get("rows")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ChronosMCPError(
                ErrorCode.source_unavailable,
                "Data Gateway returned an invalid rows payload.",
                retryable=True,
            )
        query_payload = response.get("query")
        query = QueryMeta.model_validate(query_payload) if query_payload else None
        return TableReadResult(rows=rows, query=query)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._settings.token:
            headers["Authorization"] = f"Bearer {self._settings.token}"
        request = Request(
            f"{self._settings.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._settings.timeout_seconds) as response:
                body = response.read()
        except HTTPError as exc:
            raise _gateway_http_error(exc) from exc
        except TimeoutError as exc:
            raise ChronosMCPError(
                ErrorCode.query_timeout,
                "Data Gateway request timed out.",
                retryable=True,
            ) from exc
        except URLError as exc:
            raise ChronosMCPError(
                ErrorCode.source_unavailable,
                f"Data Gateway is unavailable: {exc.reason}",
                retryable=True,
            ) from exc
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ChronosMCPError(
                ErrorCode.source_unavailable,
                "Data Gateway returned invalid JSON.",
                retryable=True,
            ) from exc
        if not isinstance(decoded, dict):
            raise ChronosMCPError(
                ErrorCode.source_unavailable,
                "Data Gateway returned a non-object response.",
                retryable=True,
            )
        return decoded


def _gateway_http_error(error: HTTPError) -> ChronosMCPError:
    body = error.read().decode("utf-8", errors="replace")[:500]
    if error.code in {401, 403}:
        return ChronosMCPError(
            ErrorCode.unauthorized_source,
            "Data Gateway denied access to the source.",
        )
    if error.code == 404:
        return ChronosMCPError(ErrorCode.source_not_found, "Data source was not found.")
    if error.code in {408, 504}:
        return ChronosMCPError(
            ErrorCode.query_timeout,
            "Data Gateway query timed out.",
            retryable=True,
        )
    if error.code == 422:
        return ChronosMCPError(
            ErrorCode.query_rejected,
            f"Data Gateway rejected the typed query: {body}",
        )
    return ChronosMCPError(
        ErrorCode.source_unavailable,
        f"Data Gateway returned HTTP {error.code}.",
        retryable=error.code >= 500,
    )
