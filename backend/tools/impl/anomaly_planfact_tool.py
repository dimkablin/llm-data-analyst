from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import PrivateAttr

from backend.agent.prompts import anomaly_planfact_tool_prompt
from backend.tools.impl.base_tool import BaseExecTool
from backend.tools.impl.db_helpers import DBAnalyticsHelper, DemoDBConnectionView
from backend.integrations.anomaly_planfact import (
    AnomalyPlanfactIntegrationError,
    AnomalyPlanfactIntegrationService,
)
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig


@dataclass
class AnomalyPlanfactToolHelper:
    service: AnomalyPlanfactIntegrationService
    tool_name: str = "anomaly_planfact_tool"

    def analyze(
        self,
        rows: object,
        *,
        time_col: str,
        plan_col: str,
        fact_col: str,
        target_name: str | None = None,
    ) -> dict[str, Any]:
        result = self.service.run_analysis(
            rows,
            time_col=time_col,
            plan_col=plan_col,
            fact_col=fact_col,
            target_name=target_name,
        )
        payload = self.service.build_artifact_payload(
            result,
            tool_name=self.tool_name,
        )
        anomaly_meta = payload["meta"]["anomaly_planfact"]
        return {
            "rows": payload["rows"],
            "source": payload["source"],
            "recipe": payload["recipe"],
            "meta": payload["meta"],
            "summary": anomaly_meta.get("summary"),
            "warnings": list(anomaly_meta.get("warnings", [])),
        }

    def analyze_result(
        self,
        rows: object,
        *,
        time_col: str,
        plan_col: str,
        fact_col: str,
        artifact_name: str = "anomaly_planfact_result",
        target_name: str | None = None,
    ) -> dict[str, Any]:
        result = self.service.run_analysis(
            rows,
            time_col=time_col,
            plan_col=plan_col,
            fact_col=fact_col,
            target_name=target_name,
        )
        payload = self.service.build_artifact_payload(
            result,
            artifact_name=artifact_name,
            tool_name=self.tool_name,
        )
        table = pd.DataFrame(
            payload["rows"],
            columns=[
                "ts",
                "plan",
                "fact",
                "delta_abs",
                "delta_pct",
                "anomaly_score",
                "is_anomaly",
                "note",
            ],
        )
        return {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {str(payload["artifact_name"]): table},
            "source": payload["source"],
            "recipe": payload["recipe"],
            "meta": payload["meta"],
        }


class AnomalyPlanfactTool(BaseExecTool):
    name: str = "anomaly_planfact_tool"
    artifact_name: str = "table"
    human_name: str = "anomaly_planfact"
    description: str = anomaly_planfact_tool_prompt
    allowed_libs: set[str] = {"pandas", "numpy"}
    allowed_artifact_types: tuple = (pd.DataFrame, pd.Series)
    _anomaly_planfact_service: AnomalyPlanfactIntegrationService = PrivateAttr()

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        anomaly_planfact_service: AnomalyPlanfactIntegrationService,
        execution_timeout_sec: float = 25.0,
        tool_cache_size: int = 48,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
    ) -> None:
        self._anomaly_planfact_service = anomaly_planfact_service
        super().__init__(
            df,
            execution_timeout_sec=execution_timeout_sec,
            include_plotly=False,
            tool_cache_size=tool_cache_size,
            db_runtime_config=db_runtime_config,
        )

    def get_execution_scope(self) -> dict[str, Any]:
        scope: dict[str, Any] = {
            "anomaly_planfact": AnomalyPlanfactToolHelper(
                service=self._anomaly_planfact_service,
                tool_name=self.name,
            )
        }
        if self._db_runtime_config is not None:
            scope["db_connection"] = DemoDBConnectionView(runtime=self._db_runtime_config)
            scope["db"] = DBAnalyticsHelper(
                runtime=self._db_runtime_config,
                timeout_sec=min(15.0, self.execution_timeout_sec),
            )
        return scope

    def _run(self, code: str) -> tuple[str, dict[str, object]]:
        if not self._anomaly_planfact_service.is_enabled:
            text = (
                "Ошибка anomaly_planfact_tool: integration недоступна. "
                "Проверь ANOMALY_PLANFACT_BACKEND_URL и ANOMALY_PLANFACT_ENABLED."
            )
            return text, {self.artifact_name: None, "text": text}

        try:
            text, payload = super()._run(code)
        except AnomalyPlanfactIntegrationError as exc:
            message = str(exc).strip() or "Anomaly / plan-fact integration failed."
            return message, {self.artifact_name: None, "text": message}

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            return text, payload
        anomaly_meta = meta.get("anomaly_planfact")
        if not isinstance(anomaly_meta, dict):
            return text, payload

        summary = str(anomaly_meta.get("summary", "")).strip()
        model_name = str(anomaly_meta.get("model_name", "")).strip()
        point_count = anomaly_meta.get("input_point_count")
        anomaly_count = anomaly_meta.get("anomaly_count")
        lines: list[str] = []
        if isinstance(point_count, int):
            lines.append(f"Input points: {point_count}")
        if isinstance(anomaly_count, int):
            lines.append(f"Flagged rows: {anomaly_count}")
        if model_name:
            lines.append(f"Model: {model_name}")
        if summary:
            lines.append(summary)
        enriched_text = "\n".join(lines).strip()
        if enriched_text:
            payload["text"] = enriched_text
            return enriched_text, payload
        return text, payload


