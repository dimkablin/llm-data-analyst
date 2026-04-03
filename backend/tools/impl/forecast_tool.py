from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import PrivateAttr

from backend.agent.prompts import forecast_tool_prompt
from backend.tools.impl.base_tool import BaseExecTool
from backend.tools.impl.db_helpers import DBAnalyticsHelper, DemoDBConnectionView
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.integrations.forecast import (
    ForecastIntegrationError,
    ForecastIntegrationService,
)


@dataclass
class ForecastToolHelper:
    service: ForecastIntegrationService
    tool_name: str = "forecast_tool"

    def forecast(
        self,
        rows: object,
        *,
        time_col: str,
        value_col: str,
        horizon: int | None = None,
        frequency: str | None = None,
        target_name: str | None = None,
    ) -> dict[str, Any]:
        result = self.service.run_forecast(
            rows,
            time_col=time_col,
            value_col=value_col,
            horizon=horizon,
            frequency=frequency,
            target_name=target_name,
        )
        payload = self.service.build_artifact_payload(
            result,
            tool_name=self.tool_name,
        )
        forecast_meta = payload["meta"]["forecast"]
        return {
            "rows": payload["rows"],
            "source": payload["source"],
            "recipe": payload["recipe"],
            "meta": payload["meta"],
            "summary": forecast_meta.get("summary"),
            "warnings": list(forecast_meta.get("warnings", [])),
        }

    def forecast_result(
        self,
        rows: object,
        *,
        time_col: str,
        value_col: str,
        artifact_name: str = "forecast_result",
        horizon: int | None = None,
        frequency: str | None = None,
        target_name: str | None = None,
    ) -> dict[str, Any]:
        result = self.service.run_forecast(
            rows,
            time_col=time_col,
            value_col=value_col,
            horizon=horizon,
            frequency=frequency,
            target_name=target_name,
        )
        payload = self.service.build_artifact_payload(
            result,
            artifact_name=artifact_name,
            tool_name=self.tool_name,
        )
        table = pd.DataFrame(
            payload["rows"],
            columns=["ts", "yhat", "lower", "upper"],
        )
        return {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {str(payload["artifact_name"]): table},
            "source": payload["source"],
            "recipe": payload["recipe"],
            "meta": payload["meta"],
        }


class ForecastTool(BaseExecTool):
    name: str = "forecast_tool"
    artifact_name: str = "table"
    human_name: str = "прогнозов"
    description: str = forecast_tool_prompt
    allowed_libs: set[str] = {"pandas", "numpy"}
    allowed_artifact_types: tuple = (pd.DataFrame, pd.Series)
    _forecast_service: ForecastIntegrationService = PrivateAttr()

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        forecast_service: ForecastIntegrationService,
        execution_timeout_sec: float = 25.0,
        tool_cache_size: int = 48,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
        sandbox: object | None = None,
    ) -> None:
        self._forecast_service = forecast_service
        super().__init__(
            df,
            execution_timeout_sec=execution_timeout_sec,
            include_plotly=False,
            tool_cache_size=tool_cache_size,
            db_runtime_config=db_runtime_config,
            sandbox=sandbox,
        )

    def get_execution_scope(self) -> dict[str, Any]:
        scope: dict[str, Any] = {
            "forecast": ForecastToolHelper(
                service=self._forecast_service,
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
        if not self._forecast_service.is_enabled:
            text = (
                "Ошибка forecast_tool: forecast integration недоступна. "
                "Проверь FORECAST_BACKEND_URL и FORECAST_ENABLED."
            )
            return text, {self.artifact_name: None, "text": text}

        try:
            text, payload = super()._run(code)
        except ForecastIntegrationError as exc:
            message = str(exc).strip() or "Forecast integration failed."
            return message, {self.artifact_name: None, "text": message}

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            return text, payload
        forecast_meta = meta.get("forecast")
        if not isinstance(forecast_meta, dict):
            return text, payload

        horizon = forecast_meta.get("horizon")
        summary = str(forecast_meta.get("summary", "")).strip()
        model_name = str(forecast_meta.get("model_name", "")).strip()
        point_count = forecast_meta.get("input_point_count")
        lines: list[str] = []
        if isinstance(horizon, int):
            lines.append(f"Forecast horizon: {horizon}")
        if isinstance(point_count, int):
            lines.append(f"Input points: {point_count}")
        if model_name:
            lines.append(f"Model: {model_name}")
        if summary:
            lines.append(summary)
        enriched_text = "\n".join(lines).strip()
        if enriched_text:
            payload["text"] = enriched_text
            return enriched_text, payload
        return text, payload


