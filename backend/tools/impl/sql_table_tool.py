from __future__ import annotations

from typing import TYPE_CHECKING, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.data_access.sql_table_service import SQLTableService

if TYPE_CHECKING:
    from backend.tools.sandbox import SessionSandbox


class SQLTableToolArgs(BaseModel):
    question: str = Field(
        ...,
        description="Естественно-языковой аналитический вопрос по доступным таблицам.",
    )


class SQLTableTool(BaseTool):
    name: str = "sql_table_tool"
    description: str = (
        "Основной инструмент табличной аналитики по БД и/или CSV в DuckDB-сессии. "
        "Выбирает таблицу, генерирует безопасный SELECT и возвращает результат как табличный артефакт."
    )
    args_schema: Type[BaseModel] = SQLTableToolArgs
    response_format: str = "content_and_artifact"

    _service: SQLTableService = PrivateAttr()
    _sandbox: SessionSandbox | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        llm_base_url: str,
        llm_model: str,
        llm_api_key: str | None,
        llm_enable_thinking: bool = False,
        llm_chat_template_kwargs_enabled: bool = True,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
        csv_loaded: bool = False,
        csv_session_id: str | None = None,
        max_rows: int = 200,
        sandbox: SessionSandbox | None = None,
    ) -> None:
        super().__init__()
        self._service = SQLTableService(
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_enable_thinking=llm_enable_thinking,
            llm_chat_template_kwargs_enabled=llm_chat_template_kwargs_enabled,
            db_runtime_config=db_runtime_config,
            csv_loaded=csv_loaded,
            csv_session_id=csv_session_id,
            max_rows=max_rows,
        )
        self._sandbox = sandbox

    def _run(self, question: str) -> tuple[str, dict[str, object]]:
        payload = self._service.build_table_artifact(question)
        item_names = ", ".join(payload["items"].keys())
        text = f"✅ Выполнен sql_table_tool: {item_names}"

        result: dict[str, object] = {
            "text": text,
            "table": payload["items"],
        }
        if "source" in payload:
            result["source"] = payload["source"]
        if "recipe" in payload:
            result["recipe"] = payload["recipe"]
        if "meta" in payload:
            result["meta"] = payload["meta"]
        if "schema_version" in payload:
            result["schema_version"] = payload["schema_version"]
        if "artifact_type" in payload:
            result["artifact_type"] = payload["artifact_type"]

        self._log_to_notebook(question, payload)
        return text, result

    def _log_to_notebook(self, question: str, payload: dict) -> None:
        if self._sandbox is None:
            return
        try:
            import pandas as pd

            items = payload.get("items", {})
            parts = []
            for name, data in items.items():
                if isinstance(data, pd.DataFrame):
                    parts.append(f"{name}: {data.shape[0]}x{data.shape[1]}")
                else:
                    parts.append(str(name))
            result_summary = ", ".join(parts) or "—"

            recipe = payload.get("recipe")
            sql = recipe.get("sql", "") if isinstance(recipe, dict) else ""
            code = f"-- {question}\n{sql}" if sql else f"-- {question}"

            self._sandbox.log_code_entry(
                tool_name="sql_table_tool",
                code=code,
                result_summary=result_summary,
            )
        except Exception:
            pass
