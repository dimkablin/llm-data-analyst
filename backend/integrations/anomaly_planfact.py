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


def _coerce_positive_float(value: object, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.1, parsed)


class AnomalyPlanfactIntegrationError(PredictIntegrationError):
    pass


@dataclass(frozen=True)
class AnomalyPlanfactConfig:
    enabled: bool
    base_url: str
    analyze_endpoint: str
    timeout_sec: float
    backend_api_url: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    theme: str = "dark"
    source_type: str = "anomaly_planfact"
    source_ref_id: str = "anomaly_planfact"
    source_label: str = "Anomaly / Plan-fact"
    source_mode: str = "external"

    @classmethod
    def from_env(
        cls,
        settings=None,
        env: Mapping[str, str] | None = None,
    ) -> AnomalyPlanfactConfig:
        source_env = env or os.environ
        base_url = (
            clean_str(source_env.get("ANOMALY_PLANFACT_BACKEND_URL"))
            or clean_str(source_env.get("PREDICT_BACKEND_URL"))
            or ""
        )
        enabled_default = bool(base_url)
        enabled = _get_bool(source_env, "ANOMALY_PLANFACT_ENABLED", enabled_default)

        return cls(
            enabled=enabled,
            base_url=base_url.rstrip("/"),
            analyze_endpoint=clean_str(source_env.get("ANOMALY_PLANFACT_ANALYZE_ENDPOINT"))
            or "/v1/anomaly_detect_planfact",
            timeout_sec=_coerce_positive_float(
                source_env.get("ANOMALY_PLANFACT_TIMEOUT_SEC"),
                default=90.0,
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
class AnomalyPlanfactQueryResult:
    question: str
    model_name: str
    summary: str | None
    anomaly_rows: list[dict[str, Any]]
    plotly_figure: dict[str, Any] | None
    warnings: list[str]
    request_params: dict[str, Any]


class AnomalyPlanfactIntegrationService:
    def __init__(self, config: AnomalyPlanfactConfig) -> None:
        self.config = config

    @classmethod
    def from_env(
        cls,
        settings=None,
        env: Mapping[str, str] | None = None,
    ) -> AnomalyPlanfactIntegrationService:
        return cls(AnomalyPlanfactConfig.from_env(settings=settings, env=env))

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled and bool(self.config.base_url)

    def _endpoint_url(self) -> str:
        if not self.config.base_url:
            raise AnomalyPlanfactIntegrationError("Anomaly backend URL is not configured.")
        return self.config.base_url.rstrip("/") + "/" + self.config.analyze_endpoint.lstrip("/")

    def source_descriptor(self) -> dict[str, Any]:
        return build_source_descriptor(
            source_type=self.config.source_type,
            source_ref_id=self.config.source_ref_id,
            source_label=self.config.source_label,
            source_mode=self.config.source_mode,
            enabled=self.config.enabled,
            available=self.is_enabled,
            display_name_ru="Аномалии / план-факт",
            description="Сервис поиска аномалий и анализа план-факт через predict backend.",
            description_ru="Сервис поиска аномалий и анализа план-факт через predict backend.",
            capabilities=["anomaly_detection", "plan_fact"],
            requires_session_data=True,
            timeout_hint_sec=self.config.timeout_sec,
        )

    @staticmethod
    def _normalize_row(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        ts = item.get("ts") or item.get("dt") or item.get("date") or item.get("period")
        if ts is None:
            return None

        # В predict-service сейчас planfact отдаёт y/yhat/lower/upper/severity/direction.
        # Здесь жёстко нормализуем именно в этот формат, чтобы дальше тул собирал
        # таблицу и график без пустых колонок.
        y = item.get("y")
        if y is None:
            y = item.get("fact")

        yhat = item.get("yhat")
        if yhat is None:
            yhat = item.get("plan")

        severity = item.get("severity")
        if severity is None:
            severity = item.get("anomaly_score")

        direction = item.get("direction")
        if (direction is None
                and y is not None
                and item.get("lower") is not None
                and item.get("upper") is not None):
            try:
                y_f = float(y)
                lo_f = float(item["lower"])
                up_f = float(item["upper"])
                if y_f < lo_f:
                    direction = "low"
                elif y_f > up_f:
                    direction = "high"
            except Exception:
                direction = None

        return {
            "ts": ts,
            "y": y,
            "yhat": yhat,
            "lower": item.get("lower"),
            "upper": item.get("upper"),
            "severity": severity,
            "direction": direction,
        }

    def run_analysis(
        self,
        question: str,
        *,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
        csv_session_id: str | None = None,
    ) -> AnomalyPlanfactQueryResult:
        if not self.is_enabled:
            raise AnomalyPlanfactIntegrationError("Anomaly integration is disabled or not configured.")

        clean_question = str(question or "").strip()
        if not clean_question:
            raise AnomalyPlanfactIntegrationError("Anomaly question must not be empty.")

        request_params = {
            "message": clean_question,
            "model": "PlanFact",
            "fraction": 0.2,
            "top_k": 50,
        }

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
        raw_rows = response.get("anomalies") or []
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
            warnings.append("Predict backend returned no anomaly rows.")
        if plotly_figure is None:
            warnings.append("Predict backend returned no plotly_figure.")

        return AnomalyPlanfactQueryResult(
            question=clean_question,
            model_name=clean_str(response.get("model_name") or response.get("model")) or "PlanFact",
            summary=clean_str(response.get("summary") or response.get("content")),
            anomaly_rows=rows,
            plotly_figure=plotly_figure,
            warnings=warnings,
            request_params=request_params,
        )

    def build_artifact_payload(
        self,
        result: AnomalyPlanfactQueryResult,
        *,
        artifact_name: str = "anomaly_planfact_result",
        plot_artifact_name: str = "anomaly_planfact_chart",
        tool_name: str = "anomaly_planfact_tool",
    ) -> dict[str, Any]:
        source = build_source_descriptor(
            source_type=self.config.source_type,
            source_ref_id=self.config.source_ref_id,
            source_label=self.config.source_label,
            source_mode=self.config.source_mode,
            enabled=self.config.enabled,
            available=self.is_enabled,
            display_name_ru="Аномалии / план-факт",
            description="Сервис поиска аномалий и анализа план-факт через predict backend.",
            description_ru="Сервис поиска аномалий и анализа план-факт через predict backend.",
            capabilities=["anomaly_detection", "plan_fact"],
            requires_session_data=True,
            timeout_hint_sec=self.config.timeout_sec,
        )

        recipe = [
            build_model_inference_recipe_step(
                source_type=self.config.source_type,
                tool_name=tool_name,
                model_name=result.model_name,
                params=dict(result.request_params or {}),
                result_count=len(result.anomaly_rows),
            )
        ]

        meta = {
            "anomaly_planfact": build_operation_meta(
                status="completed",
                warnings=result.warnings,
                request_params=result.request_params,
                timeout_sec=self.config.timeout_sec,
                extra={
                    "summary": result.summary,
                    "model_name": result.model_name,
                    "row_count": len(result.anomaly_rows),
                    "has_plot": bool(result.plotly_figure),
                },
            )
        }

        payload: dict[str, Any] = {
            "artifact_name": artifact_name,
            "rows": copy.deepcopy(result.anomaly_rows),
            "source": source,
            "recipe": recipe,
            "meta": meta,
        }

        if result.plotly_figure:
            payload["plot"] = {plot_artifact_name: copy.deepcopy(result.plotly_figure)}

        return payload
