from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from backend.agent.prompts import plotly_tool_prompt
from backend.artifacts.artifact_meta import build_chart_recipe_step, normalize_recipe_steps
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.tools.impl.base_tool import BaseExecTool

# Cohesive palette + dark layout aligned with frontend `ArtifactSurface` iframe (#09090b).
_CHART_COLORWAY: tuple[str, ...] = (
    "#2563eb",  # blue
    "#7c3aed",  # violet
    "#0f766e",  # teal
    "#ea580c",  # orange
)


def apply_default_chart_style(fig: Any) -> Any:
    """Polish Plotly figures before serialization: neutral layout + cohesive palette."""
    if not isinstance(fig, go.Figure):
        return fig

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(
            family="system-ui, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif",
            size=12,
            color="#18181b",
        ),
        colorway=list(_CHART_COLORWAY),
        margin=dict(l=54, r=32, t=52, b=52),
        title=dict(
            font=dict(size=15, color="#18181b", family="system-ui, sans-serif"),
            x=0.02,
            xanchor="left",
            pad=dict(t=8, b=8),
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.98)",
            bordercolor="rgba(15,23,42,0.12)",
            font=dict(size=12, color="#0f172a"),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=11, color="#475569"),
        ),
    )

    axis_grid = dict(
        showgrid=True,
        gridcolor="rgba(15,23,42,0.10)",
        zeroline=False,
        linecolor="rgba(15,23,42,0.18)",
        tickfont=dict(color="#64748b", size=11),
        title_font=dict(color="#475569", size=12),
    )
    fig.update_xaxes(**axis_grid)
    fig.update_yaxes(**axis_grid)

    fig.update_traces(
        marker_line_width=0,
        selector=dict(type="bar"),
    )
    fig.update_traces(
        line=dict(width=2.4),
        selector=dict(type="scatter"),
    )

    return fig


@dataclass
class ChartArtifactHelper:
    tool_name: str = "plotly_tool"
    _items: dict[str, Any] = field(default_factory=dict)
    _recipe_steps: list[dict[str, Any]] = field(default_factory=list)
    _source: dict[str, Any] = field(default_factory=dict)
    _meta: dict[str, Any] = field(default_factory=dict)

    def result(
        self,
        fig: Any,
        artifact_name: str = "chart",
        *,
        recipe: list[dict[str, Any]] | None = None,
        source: dict[str, Any] | None = None,
        summary: str | None = None,
        title: str | None = None,
        meta: dict[str, Any] | None = None,
        depends_on: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        styled = apply_default_chart_style(fig)
        self._items[str(artifact_name or "chart")] = styled
        if source:
            self._source.update(dict(source))

        normalized_recipe = normalize_recipe_steps(recipe)
        step = build_chart_recipe_step(
            tool_name=self.tool_name,
            summary=summary,
            title=title,
            depends_on=depends_on,
        )
        normalized_recipe.append(step)
        self._recipe_steps.extend(normalized_recipe)
        if isinstance(meta, dict) and meta:
            self._meta.update(dict(meta))

        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "artifact_type": "plot",
            "items": dict(self._items),
        }
        if self._source:
            payload["source"] = dict(self._source)
        merged_recipe = normalize_recipe_steps(self._recipe_steps)
        if merged_recipe:
            payload["recipe"] = merged_recipe
        if self._meta:
            payload["meta"] = dict(self._meta)
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
    allowed_libs: set[str] = {
        "plotly", "pandas", "numpy",
        "datetime", "math", "statistics", "calendar", "collections", "itertools", "re",
    }
    allowed_artifact_types: tuple = (go.Figure,)
    TOOL_ENABLE_THINKING: ClassVar[bool] = False  # deterministic, temp=0

    def __init__(
        self,
        df: pd.DataFrame,
        execution_timeout_sec: float = 25.0,
        tool_cache_size: int = 48,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
        sandbox: Any | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_enable_thinking: bool = False,
        llm_chat_template_kwargs_enabled: bool = True,
        llm_provider: str = "",
        code_fix_max_retries: int = 3,
    ) -> None:
        _ = (px, go)  # imports остаются для явной зависимости инструмента
        super().__init__(
            df,
            execution_timeout_sec=execution_timeout_sec,
            include_plotly=True,
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

    def get_execution_scope(self) -> dict[str, Any]:
        scope: dict[str, Any] = {
            "chart": ChartArtifactHelper(tool_name=self.name),
        }
        return scope
