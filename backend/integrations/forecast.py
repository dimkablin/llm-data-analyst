from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from backend.artifacts.artifact_meta import build_model_inference_recipe_step
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.integrations.contract import build_operation_meta, build_source_descriptor
from backend.integrations.predict_common import (
    PredictIntegrationError,
    build_db_payload,
    build_llm_payload,
    clean_str,
    post_json,
)


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


_FORECAST_TRACE_RE = re.compile(
    r"(forecast|prediction|predicted|yhat|прогноз|предикт)",
    flags=re.IGNORECASE,
)
_CONFIDENCE_TRACE_RE = re.compile(
    r"(confidence|interval|bound|lower|upper|ci|доверит|интервал)",
    flags=re.IGNORECASE,
)


def _sequence_has_points(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, dict) and {"dtype", "bdata"} <= set(value):
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) > 0
    return True


def _trace_has_points(trace: object) -> bool:
    if not isinstance(trace, Mapping):
        return False
    return _sequence_has_points(trace.get("x")) and _sequence_has_points(trace.get("y"))


def _forecast_rows_have_interval(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(row.get("lower") is not None and row.get("upper") is not None for row in rows)


def _plotly_figure_matches_forecast_contract(
    figure: object,
    rows: Sequence[Mapping[str, Any]],
) -> bool:
    if not isinstance(figure, Mapping):
        return False
    data = figure.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)) or not data:
        return False

    has_forecast_trace = False
    has_interval_trace = False
    for trace in data:
        if not isinstance(trace, Mapping) or not _trace_has_points(trace):
            continue
        name = str(trace.get("name") or "")
        fill = str(trace.get("fill") or "").lower()
        if _FORECAST_TRACE_RE.search(name):
            has_forecast_trace = True
        if _CONFIDENCE_TRACE_RE.search(name) or fill in {"tonexty", "tozeroy"}:
            has_interval_trace = True

    if not has_forecast_trace:
        return False
    return has_interval_trace if _forecast_rows_have_interval(rows) else not has_interval_trace


def _build_forecast_plotly_figure(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    x = [row.get("ts") for row in rows]
    yhat = [row.get("yhat") for row in rows]
    traces: list[dict[str, Any]] = []

    if _forecast_rows_have_interval(rows):
        upper = [row.get("upper") for row in rows]
        lower = [row.get("lower") for row in rows]
        traces.extend(
            [
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Upper bound",
                    "x": x,
                    "y": upper,
                    "line": {"color": "rgba(148, 163, 184, 0)"},
                    "showlegend": False,
                    "hoverinfo": "skip",
                },
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Confidence interval",
                    "x": x,
                    "y": lower,
                    "fill": "tonexty",
                    "fillcolor": "rgba(37, 99, 235, 0.14)",
                    "line": {"color": "rgba(148, 163, 184, 0)"},
                    "showlegend": True,
                    "hoverinfo": "skip",
                },
            ]
        )

    traces.append(
        {
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Forecast",
            "x": x,
            "y": yhat,
            "line": {"color": "#2563eb", "width": 3},
            "marker": {"color": "#2563eb", "size": 7},
        }
    )

    return {
        "data": traces,
        "layout": {
            "template": "plotly_white",
            "title": {"text": "Forecast"},
            "xaxis": {"title": {"text": "Period"}},
            "yaxis": {"title": {"text": "Forecast value"}},
            "showlegend": True,
            "legend": {"orientation": "h", "y": -0.2},
            "margin": {"l": 54, "r": 32, "t": 52, "b": 72},
        },
    }


class ForecastIntegrationError(PredictIntegrationError):
    pass


@dataclass(frozen=True)
class ForecastConfig:
    enabled: bool
    base_url: str
    predict_endpoint: str
    timeout_sec: float
    horizon_default: int
    backend_api_url: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    theme: str = "dark"
    source_type: str = "forecast"
    source_ref_id: str = "forecast"
    source_label: str = "Forecast"
    source_mode: str = "external"

    @classmethod
    def from_env(
        cls,
        settings=None,
        env: Mapping[str, str] | None = None,
    ) -> ForecastConfig:
        source_env = env or os.environ
        base_url = (
            clean_str(source_env.get("FORECAST_BACKEND_URL"))
            or clean_str(source_env.get("PREDICT_BACKEND_URL"))
            or ""
        )
        enabled_default = bool(base_url)
        enabled = _get_bool(source_env, "FORECAST_ENABLED", enabled_default)

        return cls(
            enabled=enabled,
            base_url=base_url.rstrip("/"),
            predict_endpoint=clean_str(source_env.get("FORECAST_PREDICT_ENDPOINT"))
            or "/v1/predict_ts_chronos",
            timeout_sec=_coerce_positive_float(
                source_env.get("FORECAST_TIMEOUT_SEC"),
                default=90.0,
            ),
            horizon_default=_coerce_positive_int(
                source_env.get("FORECAST_HORIZON_DEFAULT"),
                default=12,
            ),
            backend_api_url=(
                getattr(settings, "backend_public_api_url", None)
                or clean_str(source_env.get("BACKEND_PUBLIC_API_URL"))
                or "http://backend:8000/v1"
            ),
            llm_base_url=(
                getattr(settings, "llm_base_url", None)
                or clean_str(source_env.get("LLM_MODEL_API_URL"))
                or ""
            ),
            llm_api_key=(
                getattr(settings, "llm_api_key", None)
                or clean_str(source_env.get("LLM_API_KEY"))
                or ""
            ),
            llm_model=(
                getattr(settings, "llm_model", None)
                or clean_str(source_env.get("LLM_MODEL_NAME"))
                or ""
            ),
            theme=clean_str(source_env.get("PREDICT_THEME")) or "dark",
        )


@dataclass(frozen=True)
class ForecastQueryResult:
    question: str
    horizon: int
    model_name: str
    summary: str | None
    forecast_rows: list[dict[str, Any]]
    plotly_figure: dict[str, Any] | None
    warnings: list[str]
    request_params: dict[str, Any]


class ForecastIntegrationService:
    def __init__(self, config: ForecastConfig) -> None:
        self.config = config

    @classmethod
    def from_env(
        cls,
        settings=None,
        env: Mapping[str, str] | None = None,
    ) -> ForecastIntegrationService:
        return cls(ForecastConfig.from_env(settings=settings, env=env))

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled and bool(self.config.base_url)

    def _endpoint_url(self) -> str:
        if not self.config.base_url:
            raise ForecastIntegrationError("Forecast backend URL is not configured.")
        return self.config.base_url.rstrip("/") + "/" + self.config.predict_endpoint.lstrip("/")

    @staticmethod
    def prepare_question(question: str, *, catalog_hint: str = "") -> str:
        """Add backend-facing constraints for predict-service time-series preparation."""
        base = str(question or "").strip()
        parts = [
            base,
            "",
            "BACKEND INSTRUCTIONS FOR PREDICT-SERVICE:",
            "- Choose the most relevant table from the available tables.",
            "- If the user already named a table, use that table.",
            "- Prepare only the historical input dataset for forecasting.",
            "- The final SQL must return exactly two columns with exact aliases: dt, y.",
            "- dt must be a date/datetime/timestamp column.",
            "- y must be a numeric observed historical value.",
            "- Do not generate future rows in SQL.",
            "- Do not compute forecast values in SQL.",
        ]

        clean_catalog = str(catalog_hint or "").strip()
        if clean_catalog:
            parts.extend(["", clean_catalog])

        return "\n".join(parts).strip()

    def source_descriptor(self) -> dict[str, Any]:
        return build_source_descriptor(
            source_type=self.config.source_type,
            source_ref_id=self.config.source_ref_id,
            source_label=self.config.source_label,
            source_mode=self.config.source_mode,
            enabled=self.config.enabled,
            available=self.is_enabled,
            display_name_ru="Прогноз",
            description="Сервис прогноза временных рядов через predict backend.",
            description_ru="Сервис прогноза временных рядов через predict backend.",
            capabilities=["forecast", "time_series"],
            requires_session_data=True,
            timeout_hint_sec=self.config.timeout_sec,
        )

    @staticmethod
    def _normalize_row(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        ts = item.get("ts") or item.get("date") or item.get("dt") or item.get("period")
        if ts is None:
            return None

        return {
            "ts": ts,
            "yhat": item.get("yhat"),
            "lower": item.get("lower"),
            "upper": item.get("upper"),
        }

    def run_forecast(
        self,
        question: str,
        *,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
        csv_session_id: str | None = None,
        horizon: int | None = None,
    ) -> ForecastQueryResult:
        if not self.is_enabled:
            raise ForecastIntegrationError("Forecast integration is disabled or not configured.")

        clean_question = str(question or "").strip()
        if not clean_question:
            raise ForecastIntegrationError("Forecast question must not be empty.")

        normalized_horizon = _coerce_positive_int(horizon, default=self.config.horizon_default)
        request_params = {"message": clean_question, "fh": normalized_horizon}

        payload = {
            **request_params,
            "theme": self.config.theme,
            "llm": build_llm_payload(
                llm_base_url=self.config.llm_base_url,
                llm_api_key=self.config.llm_api_key,
                llm_model=self.config.llm_model,
            ),
            "db": build_db_payload(
                db_runtime_config=db_runtime_config,
                csv_session_id=csv_session_id,
                backend_api_url=self.config.backend_api_url,
            ),
        }

        response = post_json(self._endpoint_url(), payload, self.config.timeout_sec)

        rows: list[dict[str, Any]] = []
        raw_rows = response.get("forecast") or []
        if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes, bytearray)):
            for item in raw_rows:
                normalized = self._normalize_row(item)
                if normalized is not None:
                    rows.append(normalized)

        plotly_figure = response.get("plotly_figure")
        if not isinstance(plotly_figure, dict):
            plotly_figure = None

        warnings: list[str] = []
        raw_warnings = response.get("warnings")
        if isinstance(raw_warnings, list):
            warnings.extend(str(item).strip() for item in raw_warnings if str(item).strip())

        if not rows:
            raise ForecastIntegrationError("Predict backend returned no forecast rows.")
        if plotly_figure is None:
            warnings.append("Predict backend returned no plotly_figure.")

        return ForecastQueryResult(
            question=clean_question,
            horizon=normalized_horizon,
            model_name=clean_str(response.get("model_name") or response.get("model")) or "chronos",
            summary=clean_str(response.get("summary") or response.get("content")),
            forecast_rows=rows,
            plotly_figure=plotly_figure,
            warnings=warnings,
            request_params=request_params,
        )

    def build_artifact_payload(
        self,
        result: ForecastQueryResult,
        *,
        artifact_name: str = "forecast_result",
        plot_artifact_name: str = "forecast_chart",
        tool_name: str = "forecast_tool",
    ) -> dict[str, Any]:
        source = build_source_descriptor(
            source_type=self.config.source_type,
            source_ref_id=self.config.source_ref_id,
            source_label=self.config.source_label,
            source_mode=self.config.source_mode,
            enabled=self.config.enabled,
            available=self.is_enabled,
            display_name_ru="Прогноз",
            description="Сервис прогноза временных рядов через predict backend.",
            description_ru="Сервис прогноза временных рядов через predict backend.",
            capabilities=["forecast", "time_series"],
            requires_session_data=True,
            timeout_hint_sec=self.config.timeout_sec,
        )

        recipe = [
            build_model_inference_recipe_step(
                source_type=self.config.source_type,
                tool_name=tool_name,
                model_name=result.model_name,
                params=dict(result.request_params or {}),
                result_count=len(result.forecast_rows),
            )
        ]

        plotly_figure, built_fallback_plot = self._plotly_figure_for_result(result)
        warnings = list(result.warnings)
        if built_fallback_plot:
            warnings.append("Built forecast_chart from forecast rows to match forecast/CI artifact contract.")

        meta = {
            "forecast": build_operation_meta(
                status="completed",
                warnings=warnings,
                request_params=result.request_params,
                timeout_sec=self.config.timeout_sec,
                extra={
                    "summary": result.summary,
                    "model_name": result.model_name,
                    "row_count": len(result.forecast_rows),
                    "has_plot": bool(plotly_figure),
                },
            )
        }

        payload: dict[str, Any] = {
            "artifact_name": artifact_name,
            "rows": copy.deepcopy(result.forecast_rows),
            "source": source,
            "recipe": recipe,
            "meta": meta,
        }

        if plotly_figure:
            payload["plot"] = {plot_artifact_name: copy.deepcopy(plotly_figure)}

        return payload

    @staticmethod
    def _plotly_figure_for_result(result: ForecastQueryResult) -> tuple[dict[str, Any], bool]:
        if _plotly_figure_matches_forecast_contract(result.plotly_figure, result.forecast_rows):
            return copy.deepcopy(result.plotly_figure), False  # type: ignore[arg-type]
        return _build_forecast_plotly_figure(result.forecast_rows), True
