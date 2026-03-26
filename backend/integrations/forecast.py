from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from backend.artifacts.artifact_meta import build_model_inference_recipe_step
from backend.integrations.contract import build_operation_meta, build_source_descriptor


def _clean_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _get_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce_positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _coerce_positive_float(value: object, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.1, parsed)


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


@dataclass(frozen=True)
class ForecastConfig:
    enabled: bool
    base_url: str
    predict_endpoint: str
    timeout_sec: float
    horizon_default: int
    source_type: str = "forecast"
    source_ref_id: str = "forecast"
    source_label: str = "Forecast"
    source_mode: str = "external"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "ForecastConfig":
        source_env = env or os.environ
        base_url = _clean_str(source_env.get("FORECAST_BACKEND_URL")) or ""
        enabled_default = bool(base_url)
        enabled = _get_bool(source_env, "FORECAST_ENABLED", enabled_default)
        return cls(
            enabled=enabled,
            base_url=base_url.rstrip("/"),
            predict_endpoint=_clean_str(source_env.get("FORECAST_PREDICT_ENDPOINT"))
            or "/v1/predict_ts_chronos",
            timeout_sec=_coerce_positive_float(
                source_env.get("FORECAST_TIMEOUT_SEC"),
                default=60.0,
            ),
            horizon_default=_coerce_positive_int(
                source_env.get("FORECAST_HORIZON_DEFAULT"),
                default=3,
            ),
            source_label=_clean_str(source_env.get("FORECAST_SOURCE_LABEL"))
            or "Forecast",
        )

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.base_url)


@dataclass(frozen=True)
class ForecastQueryResult:
    horizon: int
    time_col: str
    value_col: str
    input_point_count: int
    frequency: str | None
    target_name: str | None
    model_name: str | None
    summary: str | None
    forecast_rows: list[dict[str, Any]]
    warnings: list[str]
    request_params: dict[str, Any]

    @property
    def result_count(self) -> int:
        return len(self.forecast_rows)


class ForecastIntegrationError(RuntimeError):
    pass


ForecastTransport = Callable[[str, dict[str, Any], float], dict[str, Any]]


def _default_transport(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            raw_body = response.read()
    except HTTPError as exc:
        body_preview = exc.read().decode("utf-8", errors="replace")[:500]
        raise ForecastIntegrationError(
            f"Forecast backend returned HTTP {exc.code}: {body_preview}"
        ) from exc
    except URLError as exc:
        raise ForecastIntegrationError(
            f"Forecast backend is unavailable: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise ForecastIntegrationError("Forecast backend request timed out.") from exc

    if not raw_body:
        return {}
    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        preview = raw_body.decode("utf-8", errors="replace")[:500]
        raise ForecastIntegrationError(
            f"Forecast backend returned invalid JSON: {preview!r}"
        ) from exc
    if not isinstance(decoded, dict):
        raise ForecastIntegrationError(
            "Forecast backend returned a non-object JSON payload."
        )
    return decoded


class ForecastIntegrationService:
    def __init__(
        self,
        config: ForecastConfig,
        *,
        transport: ForecastTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport or _default_transport

    @classmethod
    def from_env(cls) -> "ForecastIntegrationService":
        return cls(ForecastConfig.from_env())

    @property
    def is_enabled(self) -> bool:
        return self.config.available

    def source_ref(self) -> dict[str, str]:
        return {
            "source_type": self.config.source_type,
            "source_ref_id": self.config.source_ref_id,
            "source_label": self.config.source_label,
            "source_mode": self.config.source_mode,
        }

    def source_descriptor(self) -> dict[str, Any]:
        return build_source_descriptor(
            source_type=self.config.source_type,
            source_ref_id=self.config.source_ref_id,
            source_label=self.config.source_label,
            display_name_ru="Прогноз",
            source_mode=self.config.source_mode,
            enabled=self.config.enabled,
            available=self.config.available,
            description="External forecasting integration.",
            description_ru="Прогнозирование по компактным временным рядам из данных сессии.",
            capabilities=["forecast", "time_series_forecast"],
            requires_session_data=True,
            timeout_hint_sec=self.config.timeout_sec,
        )

    @staticmethod
    def _artifact_name(value: str | None) -> str:
        text = str(value or "").strip()
        return text or "forecast_result"

    def _endpoint_url(self) -> str:
        if not self.config.base_url:
            raise ForecastIntegrationError(
                "Forecast integration is not configured. Set FORECAST_BACKEND_URL first."
            )
        return urljoin(f"{self.config.base_url}/", self.config.predict_endpoint.lstrip("/"))

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._transport(
                self._endpoint_url(),
                payload,
                self.config.timeout_sec,
            )
        except ForecastIntegrationError:
            raise
        except HTTPError as exc:
            body_preview = ""
            try:
                body_preview = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                body_preview = ""
            suffix = f": {body_preview}" if body_preview else ""
            raise ForecastIntegrationError(
                f"Forecast backend returned HTTP {exc.code}{suffix}"
            ) from exc
        except URLError as exc:
            raise ForecastIntegrationError(
                f"Forecast backend is unavailable: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ForecastIntegrationError("Forecast backend request timed out.") from exc

    @staticmethod
    def _normalize_time_value(value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        clean = _clean_str(_json_safe_value(value))
        if not clean:
            raise ForecastIntegrationError("Forecast input contains an empty time value.")
        return clean

    @staticmethod
    def _normalize_numeric_value(value: Any) -> float:
        if isinstance(value, bool):
            raise ForecastIntegrationError("Boolean values are not valid forecast targets.")
        try:
            numeric = float(_json_safe_value(value))
        except (TypeError, ValueError) as exc:
            raise ForecastIntegrationError(
                f"Forecast input contains a non-numeric target value: {value!r}"
            ) from exc
        return numeric

    @staticmethod
    def _records_from_rows(rows: object) -> list[dict[str, Any]]:
        if rows is None:
            return []
        if isinstance(rows, list):
            return [dict(item) for item in rows if isinstance(item, dict)]
        if isinstance(rows, tuple):
            return [dict(item) for item in rows if isinstance(item, dict)]
        to_dict = getattr(rows, "to_dict", None)
        if callable(to_dict):
            try:
                records = to_dict(orient="records")
            except TypeError:
                records = None
            if isinstance(records, list):
                return [dict(item) for item in records if isinstance(item, dict)]
        raise ForecastIntegrationError(
            "Forecast input rows must be a list of dicts or a DataFrame-like object."
        )

    def normalize_series_input(
        self,
        rows: object,
        *,
        time_col: str,
        value_col: str,
    ) -> list[dict[str, Any]]:
        records = self._records_from_rows(rows)
        if len(records) < 3:
            raise ForecastIntegrationError(
                "Forecast requires at least 3 history points in the input series."
            )

        clean_time_col = _clean_str(time_col)
        clean_value_col = _clean_str(value_col)
        if not clean_time_col or not clean_value_col:
            raise ForecastIntegrationError("time_col and value_col are required.")

        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(records, start=1):
            if clean_time_col not in row:
                raise ForecastIntegrationError(
                    f"Forecast input row {index} is missing time column '{clean_time_col}'."
                )
            if clean_value_col not in row:
                raise ForecastIntegrationError(
                    f"Forecast input row {index} is missing value column '{clean_value_col}'."
                )
            normalized.append(
                {
                    "ts": self._normalize_time_value(row.get(clean_time_col)),
                    "y": self._normalize_numeric_value(row.get(clean_value_col)),
                }
            )
        return normalized

    @staticmethod
    def _normalize_forecast_row(raw: object) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        ts = _clean_str(
            raw.get("ts")
            or raw.get("date")
            or raw.get("dt")
            or raw.get("period")
        )
        if not ts:
            return None

        value = raw.get("yhat")
        if value is None:
            value = raw.get("prediction")
        if value is None:
            value = raw.get("value")
        if value is None:
            return None

        row: dict[str, Any] = {
            "ts": ts,
            "yhat": _json_safe_value(value),
            "lower": _json_safe_value(
                raw.get("lower") if raw.get("lower") is not None else raw.get("lower_bound")
            ),
            "upper": _json_safe_value(
                raw.get("upper") if raw.get("upper") is not None else raw.get("upper_bound")
            ),
        }
        return row

    def run_forecast(
        self,
        rows: object,
        *,
        time_col: str,
        value_col: str,
        horizon: int | None = None,
        frequency: str | None = None,
        target_name: str | None = None,
    ) -> ForecastQueryResult:
        if not self.is_enabled:
            raise ForecastIntegrationError(
                "Forecast integration is disabled or not configured."
            )

        normalized_series = self.normalize_series_input(
            rows,
            time_col=time_col,
            value_col=value_col,
        )
        clean_frequency = _clean_str(frequency)
        clean_target_name = _clean_str(target_name)
        normalized_horizon = _coerce_positive_int(
            horizon,
            default=self.config.horizon_default,
        )

        request_params: dict[str, Any] = {
            "series": normalized_series,
            "horizon": normalized_horizon,
            "time_col": _clean_str(time_col),
            "value_col": _clean_str(value_col),
        }
        if clean_frequency:
            request_params["frequency"] = clean_frequency
        if clean_target_name:
            request_params["target_name"] = clean_target_name

        payload = self._request(request_params)

        raw_forecast = (
            payload.get("forecast")
            or payload.get("predictions")
            or payload.get("rows")
            or []
        )
        forecast_rows: list[dict[str, Any]] = []
        if isinstance(raw_forecast, Sequence) and not isinstance(raw_forecast, (str, bytes, bytearray)):
            for item in raw_forecast:
                normalized = self._normalize_forecast_row(item)
                if normalized is not None:
                    forecast_rows.append(normalized)

        warnings: list[str] = []
        raw_warnings = payload.get("warnings")
        if isinstance(raw_warnings, list):
            warnings.extend(
                clean for clean in (_clean_str(item) for item in raw_warnings) if clean
            )
        if not forecast_rows:
            warnings.append("Forecast backend returned no normalized forecast rows.")

        summary = _clean_str(
            payload.get("summary") or payload.get("forecast_summary") or payload.get("content")
        )
        model_name = _clean_str(payload.get("model_name") or payload.get("model"))

        return ForecastQueryResult(
            horizon=normalized_horizon,
            time_col=str(request_params["time_col"]),
            value_col=str(request_params["value_col"]),
            input_point_count=len(normalized_series),
            frequency=clean_frequency,
            target_name=clean_target_name,
            model_name=model_name,
            summary=summary,
            forecast_rows=forecast_rows,
            warnings=warnings,
            request_params=copy.deepcopy(request_params),
        )

    def build_artifact_payload(
        self,
        result: ForecastQueryResult,
        *,
        artifact_name: str = "forecast_result",
        tool_name: str = "forecast_tool",
    ) -> dict[str, Any]:
        return {
            "artifact_name": self._artifact_name(artifact_name),
            "rows": copy.deepcopy(result.forecast_rows),
            "source": self.source_ref(),
            "recipe": [
                build_model_inference_recipe_step(
                    source_type=self.config.source_type,
                    tool_name=tool_name,
                    title="Forecast run",
                    summary=result.summary or f"Forecast horizon={result.horizon}",
                    model_name=result.model_name,
                    params=result.request_params,
                    result_count=result.result_count,
                )
            ],
            "meta": {
                "forecast": build_operation_meta(
                    status="completed",
                    warnings=result.warnings,
                    request_params=result.request_params,
                    timeout_sec=self.config.timeout_sec,
                    extra={
                        "horizon": result.horizon,
                        "time_col": result.time_col,
                        "value_col": result.value_col,
                        "input_point_count": result.input_point_count,
                        "frequency": result.frequency,
                        "target_name": result.target_name,
                        "model_name": result.model_name,
                        "summary": result.summary,
                    },
                )
            },
        }


