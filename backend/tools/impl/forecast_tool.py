from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import PrivateAttr

from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.integrations.forecast import (
    ForecastIntegrationError,
    ForecastIntegrationService,
)
from backend.tools.impl.base_tool import BaseExecTool
from backend.tools.impl.db_helpers import DBAnalyticsHelper, DemoDBConnectionView
from backend.tools.instructions import tool_description


@dataclass
class ForecastToolHelper:
    service: ForecastIntegrationService
    tool_name: str = "forecast_tool"
    db_runtime_config: RuntimeDBConnectionConfig | None = None
    csv_session_id: str | None = None

    def forecast(
        self,
        question: str,
        *,
        horizon: int | None = None,
    ) -> dict[str, Any]:
        return self.forecast_result(question, horizon=horizon)

    def forecast_result(
        self,
        question: str,
        *,
        artifact_name: str = "forecast_result",
        plot_artifact_name: str = "forecast_chart",
        horizon: int | None = None,
    ) -> dict[str, Any]:
        result = self.service.run_forecast(
            self._prepare_question(question),
            db_runtime_config=self.db_runtime_config,
            csv_session_id=self.csv_session_id,
            horizon=horizon,
        )
        payload = self.service.build_artifact_payload(
            result,
            artifact_name=artifact_name,
            plot_artifact_name=plot_artifact_name,
            tool_name=self.tool_name,
        )

        rows = payload["rows"] or []
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            row = dict(row or {})
            normalized_rows.append(
                {
                    "ts": row.get("ts"),
                    "yhat": row.get("yhat"),
                    "lower": row.get("lower"),
                    "upper": row.get("upper"),
                }
            )

        table = pd.DataFrame(
            normalized_rows,
            columns=["ts", "yhat", "lower", "upper"],
        )

        artifact: dict[str, Any] = {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {str(payload["artifact_name"]): table},
            "source": payload["source"],
            "recipe": payload["recipe"],
            "meta": payload["meta"],
        }

        if "plot" in payload:
            artifact["plot"] = payload["plot"]

        return artifact

    def _catalog_hint(self) -> str:
        if self.db_runtime_config is None:
            return ""

        try:
            db = DBAnalyticsHelper(self.db_runtime_config, timeout_sec=10.0)
            rows = db.list_tables_with_columns()
        except Exception:
            return ""

        lines = []
        for row in rows[:20]:
            qname = row.get("qualified_name") or row.get("table_name")
            cols = row.get("columns") or []
            cols_preview = ", ".join(map(str, cols[:30]))
            lines.append(f"- {qname}: {cols_preview}")

        if not lines:
            return ""

        return "Доступные таблицы и колонки:\n" + "\n".join(lines)

    def _prepare_question(self, question: str) -> str:
        base = str(question or "").strip()
        catalog = self._catalog_hint()

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

        if catalog:
            parts.extend(["", catalog])

        return "\n".join(parts).strip()


class ForecastTool(BaseExecTool):
    name: str = "forecast_tool"
    artifact_name: str = "table"
    human_name: str = "прогнозов"
    description: str = tool_description("forecast_tool")
    allowed_libs: set[str] = {"pandas", "numpy"}
    allowed_artifact_types: tuple = (pd.DataFrame, pd.Series)

    _forecast_service: ForecastIntegrationService = PrivateAttr()
    _db_runtime_config: RuntimeDBConnectionConfig | None = PrivateAttr(default=None)
    _csv_session_id: str | None = PrivateAttr(default=None)

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        forecast_service: ForecastIntegrationService,
        execution_timeout_sec: float = 25.0,
        tool_cache_size: int = 48,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
        csv_session_id: str | None = None,
        sandbox: object | None = None,
    ) -> None:
        super().__init__(
            df,
            execution_timeout_sec=execution_timeout_sec,
            include_plotly=False,
            tool_cache_size=tool_cache_size,
            db_runtime_config=db_runtime_config,
            sandbox=sandbox,
        )
        self._forecast_service = forecast_service
        self._db_runtime_config = db_runtime_config
        self._csv_session_id = str(csv_session_id or "").strip() or None

    def get_execution_scope(self) -> dict[str, Any]:
        scope = super().get_execution_scope()
        scope["forecast"] = ForecastToolHelper(
            service=self._forecast_service,
            tool_name=self.name,
            db_runtime_config=self._db_runtime_config,
            csv_session_id=self._csv_session_id,
        )
        if self._db_runtime_config is not None:
            scope["db_connection"] = DemoDBConnectionView(self._db_runtime_config)
            scope["db"] = DBAnalyticsHelper(self._db_runtime_config)
        return scope

    def user_facing_error(self, exc: Exception) -> str:
        if isinstance(exc, ForecastIntegrationError):
            return f"❌ Ошибка прогноза: {exc}"
        return super().user_facing_error(exc)

    def _try_run_once(self, code: str) -> tuple[bool, str, dict[str, object]]:
        try:
            tool_result = self._execute_in_sandbox(code)
            artifact_hints = self._extract_payload_hints(tool_result)

            if tool_result is None:
                return False, "Не найдена переменная `tool_result`", {}

            normalized_result, contract_message = self._validate_tool_contract(tool_result)
            if normalized_result is None:
                return False, contract_message, {}

            normalized_result = self.post_process_tool_result(normalized_result)
            if not isinstance(normalized_result, dict) or not normalized_result:
                return False, "post_process_tool_result вернул пустой или неверный результат.", {}

            valid, validate_message = self.validate_tool_result(normalized_result)
            if not valid:
                return False, validate_message, {}

            text = (
                f"✅ Создано через {self.name} - {len(normalized_result)} {self.human_name}: "
                f"{', '.join(normalized_result.keys())}"
            )

            payload: dict[str, object] = {"text": text, "code": code}
            if artifact_hints:
                payload.update(artifact_hints)

            payload[self.artifact_name] = normalized_result

            if isinstance(tool_result, dict):
                plot_payload = tool_result.get("plot")
                if isinstance(plot_payload, dict) and plot_payload:
                    payload["plot"] = copy.deepcopy(plot_payload)

            return True, text, payload

        except SyntaxError as e:
            code_lines = code.splitlines()
            error_line = (
                code_lines[e.lineno - 1]
                if e.lineno and e.lineno <= len(code_lines)
                else ""
            )
            return False, f"SyntaxError: {e.msg}\n{error_line}", {}
        except Exception as e:
            return False, str(e) or e.__class__.__name__, {}
