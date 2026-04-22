from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.data_access.sql_table_service import SQLTableService

if TYPE_CHECKING:
    from backend.tools.sandbox import SessionSandbox


class SQLToolArgs(BaseModel):
    question: str = Field(
        ...,
        description="Естественно-языковой аналитический вопрос по доступным таблицам.",
    )
    artifact_name: str | None = Field(
        default=None,
        description=(
            "Необязательное желаемое имя Python-переменной для результата "
            "в snake_case, например women_by_district или monthly_sales."
        ),
    )


class SQLTool(BaseTool):
    name: str = "sql_tool"
    description: str = (
        "Основной инструмент табличной аналитики по БД и/или CSV в DuckDB-сессии. "
        "Выбирает таблицу, генерирует безопасный SELECT и возвращает результат как табличный артефакт. "
        "Можно передать artifact_name — желаемое имя результата для дальнейшего использования "
        "в plotly_tool/pandas_tool."
    )
    args_schema: type[BaseModel] = SQLToolArgs
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
        llm_provider: str = "",
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
            llm_provider=llm_provider,
            db_runtime_config=db_runtime_config,
            csv_loaded=csv_loaded,
            csv_session_id=csv_session_id,
            max_rows=max_rows,
        )
        self._sandbox = sandbox

    def _sanitize_artifact_name(self, value: str | None) -> str | None:
        text = str(value or "").strip().lower()
        if not text:
            return None

        text = re.sub(r"\W+", "_", text, flags=re.UNICODE)
        text = re.sub(r"_+", "_", text).strip("_")

        if not text:
            return None

        if text[0].isdigit():
            text = f"result_{text}"

        return text[:80]

    def _run(
        self,
        question: str,
        artifact_name: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        try:
            return self._run_query(question, artifact_name=artifact_name)
        except Exception as exc:
            error_text = f"❌ Ошибка sql_tool: {exc}"
            return error_text, {"text": error_text}

    def _run_query(
        self,
        question: str,
        artifact_name: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        import pandas as pd

        clean_artifact_name = self._sanitize_artifact_name(artifact_name)

        payload = self._service.build_table_artifact(
            question,
            artifact_name=clean_artifact_name,
        )
        item_names = ", ".join(payload["items"].keys())

        # Inject result DataFrames into sandbox scope so subsequent tools
        # (plotly_tool, pandas_tool) can reference them by variable name.
        injected: list[str] = []
        if self._sandbox is not None:
            for name, data in payload["items"].items():
                if isinstance(data, pd.DataFrame):
                    self._sandbox.put(name, data)
                    injected.append(name)

        if injected:
            vars_hint = ", ".join(f"`{v}`" for v in injected)
            text = (
                f"✅ Выполнен sql_tool: {item_names}. "
                f"Результаты доступны как Python-переменные: {vars_hint}. "
                f"Используй эти имена напрямую в plotly_tool/pandas_tool."
            )
        else:
            text = f"✅ Выполнен sql_tool: {item_names}"

        result: dict[str, object] = {
            "text": text,
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": payload["items"],
        }
        if "source" in payload:
            result["source"] = payload["source"]
        if "recipe" in payload:
            result["recipe"] = payload["recipe"]
        if "meta" in payload:
            result["meta"] = payload["meta"]

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

            self._sandbox.log_code_entry(
                tool_name="sql_tool",
                language="sql",
                question=question,
                code=sql,
                result_summary=result_summary,
            )
        except Exception:
            pass
