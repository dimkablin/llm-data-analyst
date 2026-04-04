from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any
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
class AnomalyPlanfactConfig:
    enabled: bool
    base_url: str
    analyze_endpoint: str
    timeout_sec: float
    source_type: str = "anomaly_planfact"
    source_ref_id: str = "anomaly_planfact"
    source_label: str = "Anomaly / Plan-fact"
    source_mode: str = "external"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> AnomalyPlanfactConfig:
        source_env = env or os.environ
        base_url = _clean_str(source_env.get("ANOMALY_PLANFACT_BACKEND_URL")) or ""
        enabled_default = bool(base_url)
        enabled = _get_bool(
            source_env,
            "ANOMALY_PLANFACT_ENABLED",
            enabled_default,
        )
        return cls(
            enabled=enabled,
            base_url=base_url.rstrip("/"),
            analyze_endpoint=_clean_str(
                source_env.get("ANOMALY_PLANFACT_ANALYZE_ENDPOINT")
            )
            or "/v1/anomaly_detect_planfact",
            timeout_sec=_coerce_positive_float(
                source_env.get("ANOMALY_PLANFACT_TIMEOUT_SEC"),
                default=60.0,
            ),
            source_label=_clean_str(
                source_env.get("ANOMALY_PLANFACT_SOURCE_LABEL")
            )
            or "Anomaly / Plan-fact",
        )

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.base_url)


@dataclass(frozen=True)
class AnomalyPlanfactQueryResult:
    time_col: str
    plan_col: str
    fact_col: str
    input_point_count: int
    target_name: str | None
    model_name: str | None
    summary: str | None
    analysis_rows: list[dict[str, Any]]
    warnings: list[str]
    request_params: dict[str, Any]

    @property
    def result_count(self) -> int:
        return len(self.analysis_rows)

    @property
    def anomaly_count(self) -> int:
        return sum(1 for row in self.analysis_rows if row.get("is_anomaly") is True)


class AnomalyPlanfactIntegrationError(RuntimeError):
    pass


AnomalyPlanfactTransport = Callable[[str, dict[str, Any], float], dict[str, Any]]


def _default_transport(
    url: str,
    payload: dict[str, Any],
    timeout_sec: float,
) -> dict[str, Any]:
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
        raise AnomalyPlanfactIntegrationError(
            f"Anomaly / plan-fact backend returned HTTP {exc.code}: {body_preview}"
        ) from exc
    except URLError as exc:
        raise AnomalyPlanfactIntegrationError(
            f"Anomaly / plan-fact backend is unavailable: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise AnomalyPlanfactIntegrationError(
            "Anomaly / plan-fact backend request timed out."
        ) from exc

    if not raw_body:
        return {}
    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        preview = raw_body.decode("utf-8", errors="replace")[:500]
        raise AnomalyPlanfactIntegrationError(
            f"Anomaly / plan-fact backend returned invalid JSON: {preview!r}"
        ) from exc
    if not isinstance(decoded, dict):
        raise AnomalyPlanfactIntegrationError(
            "Anomaly / plan-fact backend returned a non-object JSON payload."
        )
    return decoded


class AnomalyPlanfactIntegrationService:
    def __init__(
        self,
        config: AnomalyPlanfactConfig,
        *,
        transport: AnomalyPlanfactTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport or _default_transport

    @classmethod
    def from_env(cls) -> AnomalyPlanfactIntegrationService:
        return cls(AnomalyPlanfactConfig.from_env())

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
            display_name_ru="План-факт и аномалии",
            source_mode=self.config.source_mode,
            enabled=self.config.enabled,
            available=self.config.available,
            description="External anomaly and plan-fact analysis integration.",
            description_ru="Анализ отклонений, план-факт и поиск аномалий по выровненным временным рядам.",
            capabilities=["anomaly_detection", "plan_fact_analysis"],
            requires_session_data=True,
            timeout_hint_sec=self.config.timeout_sec,
        )

    @staticmethod
    def _artifact_name(value: str | None) -> str:
        text = str(value or "").strip()
        return text or "anomaly_planfact_result"

    def _endpoint_url(self) -> str:
        if not self.config.base_url:
            raise AnomalyPlanfactIntegrationError(
                "Anomaly / plan-fact integration is not configured. "
                "Set ANOMALY_PLANFACT_BACKEND_URL first."
            )
        return urljoin(
            f"{self.config.base_url}/",
            self.config.analyze_endpoint.lstrip("/"),
        )

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._transport(
                self._endpoint_url(),
                payload,
                self.config.timeout_sec,
            )
        except AnomalyPlanfactIntegrationError:
            raise
        except HTTPError as exc:
            body_preview = ""
            try:
                body_preview = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                body_preview = ""
            suffix = f": {body_preview}" if body_preview else ""
            raise AnomalyPlanfactIntegrationError(
                f"Anomaly / plan-fact backend returned HTTP {exc.code}{suffix}"
            ) from exc
        except URLError as exc:
            raise AnomalyPlanfactIntegrationError(
                f"Anomaly / plan-fact backend is unavailable: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise AnomalyPlanfactIntegrationError(
                "Anomaly / plan-fact backend request timed out."
            ) from exc

    @staticmethod
    def _normalize_time_value(value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        clean = _clean_str(_json_safe_value(value))
        if not clean:
            raise AnomalyPlanfactIntegrationError(
                "Anomaly / plan-fact input contains an empty time value."
            )
        return clean

    @staticmethod
    def _normalize_numeric_value(value: Any, *, label: str) -> float:
        if isinstance(value, bool):
            raise AnomalyPlanfactIntegrationError(
                f"Boolean values are not valid for {label}."
            )
        try:
            numeric = float(_json_safe_value(value))
        except (TypeError, ValueError) as exc:
            raise AnomalyPlanfactIntegrationError(
                f"Anomaly / plan-fact input contains a non-numeric {label}: {value!r}"
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
        raise AnomalyPlanfactIntegrationError(
            "Anomaly / plan-fact input rows must be a list of dicts or a DataFrame-like object."
        )

    def normalize_series_input(
        self,
        rows: object,
        *,
        time_col: str,
        plan_col: str,
        fact_col: str,
    ) -> list[dict[str, Any]]:
        records = self._records_from_rows(rows)
        if len(records) < 2:
            raise AnomalyPlanfactIntegrationError(
                "Anomaly / plan-fact analysis requires at least 2 aligned points."
            )

        clean_time_col = _clean_str(time_col)
        clean_plan_col = _clean_str(plan_col)
        clean_fact_col = _clean_str(fact_col)
        if not clean_time_col or not clean_plan_col or not clean_fact_col:
            raise AnomalyPlanfactIntegrationError(
                "time_col, plan_col and fact_col are required."
            )

        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(records, start=1):
            if clean_time_col not in row:
                raise AnomalyPlanfactIntegrationError(
                    "Anomaly / plan-fact input row "
                    f"{index} is missing time column '{clean_time_col}'."
                )
            if clean_plan_col not in row:
                raise AnomalyPlanfactIntegrationError(
                    "Anomaly / plan-fact input row "
                    f"{index} is missing plan column '{clean_plan_col}'."
                )
            if clean_fact_col not in row:
                raise AnomalyPlanfactIntegrationError(
                    "Anomaly / plan-fact input row "
                    f"{index} is missing fact column '{clean_fact_col}'."
                )
            normalized.append(
                {
                    "ts": self._normalize_time_value(row.get(clean_time_col)),
                    "plan": self._normalize_numeric_value(
                        row.get(clean_plan_col),
                        label="plan value",
                    ),
                    "fact": self._normalize_numeric_value(
                        row.get(clean_fact_col),
                        label="fact value",
                    ),
                }
            )
        return normalized

    @staticmethod
    def _normalize_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        clean = _clean_str(value)
        if clean is None:
            return None
        normalized = clean.lower()
        if normalized in {"1", "true", "yes", "y", "anomaly"}:
            return True
        if normalized in {"0", "false", "no", "n", "normal"}:
            return False
        return None

    @staticmethod
    def _normalize_analysis_row(raw: object) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        ts = _clean_str(
            raw.get("ts")
            or raw.get("date")
            or raw.get("dt")
            or raw.get("period")
            or raw.get("time")
        )
        if not ts:
            return None

        plan = raw.get("plan")
        if plan is None:
            plan = raw.get("expected")
        if plan is None:
            plan = raw.get("baseline")

        fact = raw.get("fact")
        if fact is None:
            fact = raw.get("actual")
        if fact is None:
            fact = raw.get("value")

        if plan is None or fact is None:
            return None

        try:
            plan_value = float(_json_safe_value(plan))
            fact_value = float(_json_safe_value(fact))
        except (TypeError, ValueError):
            return None

        delta_abs = raw.get("delta_abs")
        if delta_abs is None:
            delta_abs = raw.get("absolute_deviation")
        if delta_abs is None:
            delta_abs = raw.get("abs_diff")
        if delta_abs is None:
            delta_abs = fact_value - plan_value

        delta_pct = raw.get("delta_pct")
        if delta_pct is None:
            delta_pct = raw.get("relative_deviation")
        if delta_pct is None:
            delta_pct = raw.get("pct_diff")
        if delta_pct is None and plan_value != 0:
            delta_pct = ((fact_value - plan_value) / abs(plan_value)) * 100.0

        anomaly_score = raw.get("anomaly_score")
        if anomaly_score is None:
            anomaly_score = raw.get("score")

        return {
            "ts": ts,
            "plan": plan_value,
            "fact": fact_value,
            "delta_abs": _json_safe_value(delta_abs),
            "delta_pct": _json_safe_value(delta_pct),
            "anomaly_score": _json_safe_value(anomaly_score),
            "is_anomaly": AnomalyPlanfactIntegrationService._normalize_bool(
                raw.get("is_anomaly")
                if raw.get("is_anomaly") is not None
                else raw.get("anomaly")
            ),
            "note": _clean_str(
                raw.get("note")
                or raw.get("reason")
                or raw.get("explanation")
                or raw.get("description")
            ),
        }

    def run_analysis(
        self,
        rows: object,
        *,
        time_col: str,
        plan_col: str,
        fact_col: str,
        target_name: str | None = None,
    ) -> AnomalyPlanfactQueryResult:
        if not self.is_enabled:
            raise AnomalyPlanfactIntegrationError(
                "Anomaly / plan-fact integration is disabled or not configured."
            )

        normalized_series = self.normalize_series_input(
            rows,
            time_col=time_col,
            plan_col=plan_col,
            fact_col=fact_col,
        )
        clean_target_name = _clean_str(target_name)
        request_params: dict[str, Any] = {
            "series": normalized_series,
            "time_col": _clean_str(time_col),
            "plan_col": _clean_str(plan_col),
            "fact_col": _clean_str(fact_col),
        }
        if clean_target_name:
            request_params["target_name"] = clean_target_name

        payload = self._request(request_params)

        raw_rows = (
            payload.get("analysis")
            or payload.get("anomalies")
            or payload.get("rows")
            or payload.get("results")
            or []
        )
        analysis_rows: list[dict[str, Any]] = []
        if isinstance(raw_rows, Sequence) and not isinstance(
            raw_rows,
            (str, bytes, bytearray),
        ):
            for item in raw_rows:
                normalized = self._normalize_analysis_row(item)
                if normalized is not None:
                    analysis_rows.append(normalized)

        warnings: list[str] = []
        raw_warnings = payload.get("warnings")
        if isinstance(raw_warnings, list):
            warnings.extend(
                clean for clean in (_clean_str(item) for item in raw_warnings) if clean
            )
        if not analysis_rows:
            warnings.append(
                "Anomaly / plan-fact backend returned no normalized analysis rows."
            )

        summary = _clean_str(
            payload.get("summary")
            or payload.get("analysis_summary")
            or payload.get("content")
        )
        model_name = _clean_str(payload.get("model_name") or payload.get("model"))

        return AnomalyPlanfactQueryResult(
            time_col=str(request_params["time_col"]),
            plan_col=str(request_params["plan_col"]),
            fact_col=str(request_params["fact_col"]),
            input_point_count=len(normalized_series),
            target_name=clean_target_name,
            model_name=model_name,
            summary=summary,
            analysis_rows=analysis_rows,
            warnings=warnings,
            request_params=copy.deepcopy(request_params),
        )

    def build_artifact_payload(
        self,
        result: AnomalyPlanfactQueryResult,
        *,
        artifact_name: str = "anomaly_planfact_result",
        tool_name: str = "anomaly_planfact_tool",
    ) -> dict[str, Any]:
        return {
            "artifact_name": self._artifact_name(artifact_name),
            "rows": copy.deepcopy(result.analysis_rows),
            "source": self.source_ref(),
            "recipe": [
                build_model_inference_recipe_step(
                    source_type=self.config.source_type,
                    tool_name=tool_name,
                    title="Anomaly / plan-fact analysis",
                    summary=result.summary
                    or (
                        f"Plan-fact analysis for {result.input_point_count} aligned points"
                    ),
                    model_name=result.model_name,
                    params=result.request_params,
                    result_count=result.result_count,
                )
            ],
            "meta": {
                "anomaly_planfact": build_operation_meta(
                    status="completed",
                    warnings=result.warnings,
                    request_params=result.request_params,
                    timeout_sec=self.config.timeout_sec,
                    extra={
                        "time_col": result.time_col,
                        "plan_col": result.plan_col,
                        "fact_col": result.fact_col,
                        "input_point_count": result.input_point_count,
                        "target_name": result.target_name,
                        "model_name": result.model_name,
                        "summary": result.summary,
                        "anomaly_count": result.anomaly_count,
                    },
                )
            },
        }


