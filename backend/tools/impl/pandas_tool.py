from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pandas as pd

from backend.agent.prompts import pandas_tool_prompt
from backend.tools.impl.base_tool import BaseExecTool

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
    description: str = pandas_tool_prompt
    allowed_libs: set[str] = {"pandas", "numpy"}
    allowed_artifact_types: tuple = (pd.DataFrame, pd.Series)
    TOOL_ENABLE_THINKING: ClassVar[bool] = False  # deterministic, temp=0

    def __init__(
        self,
        df: pd.DataFrame,
        execution_timeout_sec: float = 25.0,
        tool_cache_size: int = 48,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
        sandbox: object | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_enable_thinking: bool = False,
        llm_chat_template_kwargs_enabled: bool = True,
        llm_provider: str = "",
        code_fix_max_retries: int = 3,
    ) -> None:
        super().__init__(
            df,
            execution_timeout_sec=execution_timeout_sec,
            include_plotly=False,
            tool_cache_size=tool_cache_size,
            db_runtime_config=db_runtime_config,
            sandbox=sandbox,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_enable_thinking=llm_enable_thinking,
            llm_chat_template_kwargs_enabled=llm_chat_template_kwargs_enabled,
            llm_provider=llm_provider,
            code_fix_max_retries=code_fix_max_retries,
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

    def post_process_tool_result(self, tool_result: dict[str, object]) -> dict[str, object]:
        base = super().post_process_tool_result(tool_result)
        processed: dict[str, object] = {}
        for name, value in base.items():
            if isinstance(value, (pd.DataFrame, pd.Series)):
                processed[name] = self._round_numeric_table(value)
            else:
                processed[name] = value
        return processed
