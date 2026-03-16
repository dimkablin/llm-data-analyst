import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from agent.prompts import plotly_tool_prompt
from agent.tools.base_tool import BaseExecTool


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
        )
