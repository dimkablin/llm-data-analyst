from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from agent.prompts import plotly_tool_prompt
from agent.tools.base_tool import BaseExecTool
from agent.tools.db_tool import DBAnalyticsHelper, DemoDBConnectionView
from backend.artifact_meta import build_chart_recipe_step, normalize_recipe_steps
from backend.db_runtime_service import RuntimeDBConnectionConfig


@dataclass
class ChartArtifactHelper:
    tool_name: str = "plotly_tool"

    def result(
        self,
        fig: Any,
        *,
        artifact_name: str = "chart",
        recipe: list[dict[str, Any]] | None = None,
        source: dict[str, Any] | None = None,
        summary: str | None = None,
        title: str | None = None,
        meta: dict[str, Any] | None = None,
        depends_on: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "artifact_type": "plot",
            "items": {str(artifact_name or 'chart'): fig},
        }
        if source:
            payload["source"] = dict(source)

        normalized_recipe = normalize_recipe_steps(recipe)
        normalized_recipe.append(
            build_chart_recipe_step(
                tool_name=self.tool_name,
                summary=summary,
                title=title,
                depends_on=depends_on,
            )
        )
        payload["recipe"] = normalized_recipe
        if isinstance(meta, dict) and meta:
            payload["meta"] = dict(meta)
        return payload


class PlotlyTool(BaseExecTool):
    """
    Инструмент для построения графиков с помощью Plotly.

    Attributes:
        name (str): Имя инструмента.
        artifact_name (str): Ключ результата.
        human_name (str): Человеко-понятное имя.
        description (str): Описание инструмента.
        _locals (dict): Локальные переменные для exec.
        allowed_libs (set[str]): Разрешённые библиотеки.
        allowed_artifact_types (tuple): Разрешённые типы артефактов.
    """

    name: str = "plotly_tool"
    artifact_name: str = "plot"
    human_name: str = "графиков"
    description: str = plotly_tool_prompt
    allowed_libs: set[str] = {"plotly", "pandas", "numpy"}
    allowed_artifact_types: tuple = (go.Figure,)

    def __init__(
        self,
        df: pd.DataFrame,
        execution_timeout_sec: float = 25.0,
        tool_cache_size: int = 48,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
    ) -> None:
        """
        Инициализация инструмента с DataFrame и Plotly.

        Args:
            df (pd.DataFrame): Исходный DataFrame.
        """
        _ = (px, go)  # imports остаются для явной зависимости инструмента
        super().__init__(
            df,
            execution_timeout_sec=execution_timeout_sec,
            include_plotly=True,
            tool_cache_size=tool_cache_size,
            db_runtime_config=db_runtime_config,
        )

    def get_execution_scope(self) -> dict[str, Any]:
        scope: dict[str, Any] = {
            "chart": ChartArtifactHelper(tool_name=self.name),
        }
        if self._db_runtime_config is not None:
            scope["db_connection"] = DemoDBConnectionView(runtime=self._db_runtime_config)
            scope["db"] = DBAnalyticsHelper(
                runtime=self._db_runtime_config,
                timeout_sec=min(15.0, self.execution_timeout_sec),
            )
        return scope
