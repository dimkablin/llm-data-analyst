from __future__ import annotations

import copy
import os
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
            warnings.append("Predict backend returned no forecast rows.")
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

        meta = {
            "forecast": build_operation_meta(
                status="completed",
                warnings=result.warnings,
                request_params=result.request_params,
                timeout_sec=self.config.timeout_sec,
                extra={
                    "summary": result.summary,
                    "model_name": result.model_name,
                    "row_count": len(result.forecast_rows),
                    "has_plot": bool(result.plotly_figure),
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

        if result.plotly_figure:
            payload["plot"] = {plot_artifact_name: copy.deepcopy(result.plotly_figure)}

        return payload
