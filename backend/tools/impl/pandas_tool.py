from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from backend.tools.impl.base_tool import BaseExecTool
from backend.tools.instructions import tool_description

if TYPE_CHECKING:
    from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig


class PandasTool(BaseExecTool):
    """
    Инструмент для анализа данных с помощью Pandas (создание таблиц).

    Attributes:
        name (str): Имя инструмента.
        artifact_name (str): Ключ результата.
        human_name (str): Человеко-понятное имя.
        description (str): Описание инструмента.
        _locals (dict): Локальные переменные для exec.
        allowed_libs (set[str]): Разрешённые библиотеки.
        allowed_artifact_types (tuple): Разрешённые типы артефактов.
    """

    name: str = "pandas_tool"
    artifact_name: str = "table"
    human_name: str = "таблиц"
    description: str = tool_description("pandas_tool")
    allowed_libs: set[str] = {"pandas", "numpy", "re", "datetime"}
    allowed_artifact_types: tuple = (pd.DataFrame, pd.Series)
    def __init__(
        self,
        df: pd.DataFrame,
        execution_timeout_sec: float = 25.0,
        tool_cache_size: int = 48,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
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

    @staticmethod
    def _round_numeric_table(table: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        if isinstance(table, pd.Series):
            series = table.copy()
            if pd.api.types.is_numeric_dtype(series.dtype):
                return series.round(4)
            return series

        rounded = table.copy()
        numeric_columns = rounded.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            rounded.loc[:, numeric_columns] = rounded.loc[:, numeric_columns].round(4)
        return rounded

    def _validate_tool_contract(
        self,
        tool_result: object,
    ) -> tuple[dict[str, object] | None, str]:
        if isinstance(tool_result, str) and tool_result.strip():
            return {"output": pd.DataFrame({"output": tool_result.splitlines()})}, ""
        return super()._validate_tool_contract(tool_result)

    def post_process_tool_result(self, tool_result: dict[str, object]) -> dict[str, object]:
        base = super().post_process_tool_result(tool_result)
        processed: dict[str, object] = {}
        for name, value in base.items():
            # Normalize common LLM payload slips into tabular objects accepted by pandas_tool.
            if isinstance(value, list):
                if all(isinstance(item, dict) for item in value):
                    value = pd.DataFrame(value)
                elif len(value) == 0:
                    value = pd.DataFrame()
            elif isinstance(value, dict):
                try:
                    value = pd.DataFrame([value])
                except Exception:
                    # Keep original value; validator will report precise type mismatch.
                    pass
            if isinstance(value, (pd.DataFrame, pd.Series)):
                processed[name] = self._round_numeric_table(value)
            else:
                processed[name] = value
        return processed
