from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.integrations.forecast import ForecastIntegrationService
from backend.integrations.predict_common import PredictIntegrationError
from backend.tools.impl.db_helpers import DBAnalyticsHelper
from backend.tools.instructions import tool_description


class ForecastToolArgs(BaseModel):
    question: str = Field(description="Forecast question including the target metric and period.")
    horizon: int | None = Field(
        default=None,
        ge=1,
        description="Number of future periods. Omit only when the user did not specify it.",
    )
    artifact_name: str = Field(default="forecast_result")
    plot_artifact_name: str = Field(default="forecast_chart")


class ForecastTool(BaseTool):
    name: str = "forecast_tool"
    artifact_name: str = "table"
    description: str = tool_description("forecast_tool")
    args_schema: type[BaseModel] = ForecastToolArgs
    response_format: str = "content_and_artifact"
    parallel_safe: ClassVar[bool] = False

    _forecast_service: ForecastIntegrationService = PrivateAttr()
    _db_runtime_config: RuntimeDBConnectionConfig | None = PrivateAttr(default=None)
    _csv_session_id: str | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        forecast_service: ForecastIntegrationService,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
        csv_session_id: str | None = None,
    ) -> None:
        super().__init__()
        self._forecast_service = forecast_service
        self._db_runtime_config = db_runtime_config
        self._csv_session_id = str(csv_session_id or "").strip() or None

    def _catalog_hint(self) -> str:
        if self._db_runtime_config is None:
            return ""
        try:
            rows = DBAnalyticsHelper(
                self._db_runtime_config,
                timeout_sec=10.0,
            ).list_tables_with_columns()
        except Exception:
            return ""

        lines: list[str] = []
        for row in rows[:20]:
            qualified_name = row.get("qualified_name") or row.get("table_name")
            columns = row.get("columns") or []
            lines.append(
                f"- {qualified_name}: {', '.join(map(str, columns[:30]))}"
            )
        if not lines:
            return ""
        return "Available tables and columns:\n" + "\n".join(lines)

    def _run(
        self,
        question: str,
        horizon: int | None = None,
        artifact_name: str = "forecast_result",
        plot_artifact_name: str = "forecast_chart",
        **kwargs: Any,
    ) -> tuple[str, dict[str, object]]:
        for key in ("run_manager", "callbacks", "tags", "metadata", "config"):
            kwargs.pop(key, None)

        question = str(question or "").strip()
        if not question:
            text = "Forecast error: question must not be empty."
            return text, {self.artifact_name: None, "text": text}

        try:
            prepared_question = self._forecast_service.prepare_question(
                question,
                catalog_hint=self._catalog_hint(),
            )
            result = self._forecast_service.run_forecast(
                prepared_question,
                db_runtime_config=self._db_runtime_config,
                csv_session_id=self._csv_session_id,
                horizon=horizon,
            )
            integration_payload = self._forecast_service.build_artifact_payload(
                result,
                artifact_name=artifact_name,
                plot_artifact_name=plot_artifact_name,
                tool_name=self.name,
            )
        except PredictIntegrationError as exc:
            text = f"Forecast error: {exc}"
            return text, {self.artifact_name: None, "text": text}
        except Exception as exc:
            text = f"Forecast error: {exc}"
            return text, {self.artifact_name: None, "text": text}

        rows = integration_payload.get("rows") or []
        table = pd.DataFrame(
            [
                {
                    "ts": row.get("ts"),
                    "yhat": row.get("yhat"),
                    "lower": row.get("lower"),
                    "upper": row.get("upper"),
                }
                for row in rows
            ],
            columns=["ts", "yhat", "lower", "upper"],
        )
        item_name = str(integration_payload["artifact_name"])
        text = (
            f"Forecast created for {result.horizon} period(s). "
            f"Use artifact `{item_name}` for metric-specific interpretation."
        )
        payload: dict[str, object] = {
            "text": text,
            self.artifact_name: {item_name: table},
        }
        for key in ("source", "recipe", "meta", "plot"):
            if key in integration_payload:
                payload[key] = integration_payload[key]
        return text, payload
