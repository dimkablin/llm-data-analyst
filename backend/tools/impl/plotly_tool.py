import base64
import inspect
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from backend.artifacts.artifact_meta import build_chart_recipe_step, normalize_recipe_steps
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.tools.impl.base_tool import BaseExecTool
from backend.tools.instructions import tool_description

_BASE_FORBIDDEN_CODE_PATTERNS: tuple[tuple[str, str], ...] = tuple(
    BaseExecTool.model_fields["forbidden_code_patterns"].default
)
_PLOTLY_FORBIDDEN_CODE_PATTERNS: tuple[tuple[str, str], ...] = tuple(
    (
        (
            pattern,
            "В plotly_tool нельзя использовать matplotlib. "
            "Построй Plotly Figure через px или go.",
        )
        if pattern == r"\bmatplotlib\b|\bplt\."
        else (pattern, message)
    )
    for pattern, message in _BASE_FORBIDDEN_CODE_PATTERNS
)

# Cohesive palette + dark layout aligned with frontend `ArtifactSurface`.
CHART_COLORWAY: tuple[str, ...] = (
    "#2563eb",  # blue
    "#7c3aed",  # violet
    "#0f766e",  # teal
    "#ea580c",  # orange
)
_CHART_COLORWAY = CHART_COLORWAY
_MAX_BAR_VALUE_LABELS = 6
_AXIS_SPAN_COMPAT_NOTE = (
    "# plotly_tool normalized axis-spanning annotation through Plotly `shape.label`."
)


def _plotly_json_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_plotly_json"):
        raw = value.to_plotly_json()
        return dict(raw) if isinstance(raw, dict) else {}
    return {"text": str(value)}


def _shape_label_position(raw: Any) -> str | None:
    tokens = str(raw or "").lower().split()
    tokens = [token for token in tokens if token not in {"inside", "outside"}]
    if not tokens:
        return "middle center"
    if tokens == ["middle"] or tokens[0] in {"start", "middle", "end"}:
        return " ".join(tokens)

    vertical = next((token for token in tokens if token in {"top", "middle", "bottom"}), None)
    horizontal = next((token for token in tokens if token in {"left", "center", "right"}), None)
    if vertical and horizontal:
        return f"{vertical} {horizontal}"
    if vertical:
        return f"{vertical} center"
    if horizontal:
        return f"middle {horizontal}"
    return None


def _axis_span_label_from_annotation(call_kwargs: dict[str, Any]) -> dict[str, Any]:
    raw = _plotly_json_dict(call_kwargs.get("annotation"))
    for key, value in call_kwargs.items():
        if key.startswith("annotation_"):
            raw[key.removeprefix("annotation_")] = value

    label: dict[str, Any] = {}
    for key in ("text", "font", "textangle", "xanchor", "yanchor"):
        if raw.get(key) is not None:
            label[key] = raw[key]
    if raw.get("position") is not None:
        position = _shape_label_position(raw["position"])
        if position:
            label["textposition"] = position

    existing = _plotly_json_dict(call_kwargs.get("label"))
    return {**label, **existing}


def _axis_span_call_kwargs(
    signature: inspect.Signature,
    self: go.Figure,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    bound = signature.bind_partial(self, *args, **kwargs)
    call_kwargs: dict[str, Any] = {}
    for key, value in bound.arguments.items():
        if key == "self":
            continue
        if key == "kwargs" and isinstance(value, dict):
            call_kwargs.update(value)
        else:
            call_kwargs[key] = value
    return call_kwargs


def _has_axis_span_annotation(call_kwargs: dict[str, Any]) -> bool:
    return call_kwargs.get("annotation") is not None or any(
        key.startswith("annotation_") for key in call_kwargs
    )


def _mark_axis_span_compat(fig: go.Figure) -> None:
    notes = list(getattr(fig, "_llm_data_analyst_tool_notes", []))
    if _AXIS_SPAN_COMPAT_NOTE not in notes:
        notes.append(_AXIS_SPAN_COMPAT_NOTE)
    setattr(fig, "_llm_data_analyst_tool_notes", notes)


def _install_axis_span_annotation_compat() -> None:
    for method_name in ("add_vline", "add_hline", "add_vrect", "add_hrect"):
        current = getattr(go.Figure, method_name)
        if getattr(current, "_llm_data_analyst_axis_span_compat", False):
            continue

        original = current
        signature = inspect.signature(original)

        def _safe_axis_span(
            self: go.Figure,
            *args: Any,
            _original: Any = original,
            _signature: inspect.Signature = signature,
            **kwargs: Any,
        ) -> Any:
            try:
                return _original(self, *args, **kwargs)
            except TypeError:
                call_kwargs = _axis_span_call_kwargs(_signature, self, args, kwargs)
                if not _has_axis_span_annotation(call_kwargs):
                    raise
                # ponytail: native shape.label avoids Plotly annotation mean/min/max bugs.
                label = _axis_span_label_from_annotation(call_kwargs)
                clean_kwargs = {
                    key: value
                    for key, value in call_kwargs.items()
                    if key != "annotation" and not key.startswith("annotation_")
                }
                if label:
                    clean_kwargs["label"] = label
                result = _original(self, **clean_kwargs)
                _mark_axis_span_compat(self)
                return result

        _safe_axis_span._llm_data_analyst_axis_span_compat = True  # type: ignore[attr-defined]
        setattr(go.Figure, method_name, _safe_axis_span)


_install_axis_span_annotation_compat()


def _plotly_sequence(value: Any) -> list[Any]:
    """Convert Plotly trace axis data to a plain list (numpy-safe; no ``value or []``)."""
    if value is None:
        return []
    if isinstance(value, dict) and {"dtype", "bdata"} <= set(value):
        try:
            raw = base64.b64decode(str(value["bdata"]))
            return np.frombuffer(raw, dtype=np.dtype(str(value["dtype"]))).tolist()
        except Exception:
            return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return pd.Series(value).tolist()
    except Exception:
        try:
            return list(value)
        except TypeError:
            return [value]


def _format_compact_number(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f} млрд".replace(".0 млрд", " млрд")
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f} млн".replace(".0 млн", " млн")
    if abs_value >= 10_000:
        return f"{value / 1_000:.1f} тыс".replace(".0 тыс", " тыс")
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 1:
        text = f"{value:.1f}".rstrip("0").rstrip(".")
        return text or "0"
    return f"{value:.2f}".rstrip("0").rstrip(".") or "0"


def _numeric_bar_values(trace: Any) -> list[float]:
    orientation = str(getattr(trace, "orientation", "") or "").lower()
    axis_attr = "x" if orientation == "h" else "y"
    raw = _plotly_sequence(getattr(trace, axis_attr, None))
    values: list[float] = []
    for item in raw:
        try:
            values.append(float(pd.to_numeric(item, errors="coerce")))
        except Exception:
            values.append(0.0)
    return values


def _bar_label_texts(trace: Any) -> list[str] | None:
    count = _bar_point_count(trace)
    if count > _MAX_BAR_VALUE_LABELS:
        return None
    values = _numeric_bar_values(trace)
    if not values or sum(values) <= 0:
        return None

    total = float(sum(values))
    labels: list[str] = []
    for value in values:
        if count <= 5 and total > 100:
            labels.append(f"{100.0 * value / total:.1f}%")
        elif total <= 100.5 and max(values) <= 100:
            labels.append(f"{value:.1f}%")
        else:
            labels.append(_format_compact_number(value))
    return labels


def _bar_point_count(trace: Any) -> int:
    orientation = str(getattr(trace, "orientation", "") or "").lower()
    axis_attr = "x" if orientation == "h" else "y"
    try:
        return max(1, len(_plotly_sequence(getattr(trace, axis_attr, None))))
    except Exception:
        return 1


def _enhance_pie_chart_traces(fig: go.Figure) -> None:
    for trace_index, trace in enumerate(fig.data):
        if str(getattr(trace, "type", "") or "").lower() != "pie":
            continue
        try:
            label_count = max(1, len(_plotly_sequence(getattr(trace, "labels", None))))
        except Exception:
            label_count = 1
        colors = [
            _CHART_COLORWAY[(trace_index + point_index) % len(_CHART_COLORWAY)]
            for point_index in range(label_count)
        ]
        marker: dict[str, Any] = trace.marker if isinstance(trace.marker, dict) else {}
        trace.marker = {**marker, "colors": colors, "line": {"color": "#ffffff", "width": 1}}  # type: ignore[attr-defined]
        trace.textfont = dict(size=11, color="#475569")  # type: ignore[attr-defined]


def _enhance_bar_chart_traces(fig: go.Figure) -> bool:
    """Per-bar palette colors, optional value labels. Returns True if any labels added."""
    labeled = False
    for trace_index, trace in enumerate(fig.data):
        if str(getattr(trace, "type", "") or "").lower() != "bar":
            continue
        count = _bar_point_count(trace)
        colors = [
            _CHART_COLORWAY[(trace_index + point_index) % len(_CHART_COLORWAY)]
            for point_index in range(count)
        ]
        marker: dict[str, Any] = {
            "color": colors if count > 1 else colors[0],
            "line": {"width": 0},
            "opacity": 0.9,
            "cornerradius": 6,
        }
        trace.marker = marker  # type: ignore[attr-defined]

        label_texts = _bar_label_texts(trace)
        if label_texts:
            trace.text = label_texts  # type: ignore[attr-defined]
            trace.texttemplate = "%{text}"  # type: ignore[attr-defined]
            trace.textposition = "outside"  # type: ignore[attr-defined]
            trace.textfont = dict(size=10, color="#475569")  # type: ignore[attr-defined]
            trace.cliponaxis = False  # type: ignore[attr-defined]
            labeled = True
        else:
            trace.text = None  # type: ignore[attr-defined]
            trace.texttemplate = None  # type: ignore[attr-defined]

    fig.update_layout(bargap=0.32, bargroupgap=0.14)
    return labeled


def _apply_legend_policy(fig: go.Figure, *, has_bar_labels: bool) -> None:
    multi_series = len(fig.data) > 1
    if multi_series:
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0)",
                borderwidth=0,
                font=dict(size=11, color="#475569"),
            ),
            margin=dict(
                l=54,
                r=40 if has_bar_labels else 32,
                t=64 if has_bar_labels else 52,
                b=72,
            ),
        )
    else:
        fig.update_layout(
            showlegend=False,
            margin=dict(
                l=54,
                r=40 if has_bar_labels else 32,
                t=64 if has_bar_labels else 52,
                b=52,
            ),
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

    has_bar_labels = _enhance_bar_chart_traces(fig)
    _enhance_pie_chart_traces(fig)
    _apply_legend_policy(fig, has_bar_labels=has_bar_labels)

    fig.update_traces(
        line=dict(width=2.6),
        selector=dict(type="scatter"),
    )
    fig.update_traces(
        marker=dict(size=7, line=dict(width=0)),
        selector=dict(type="scatter", mode="markers"),
    )
    fig.update_traces(
        fillcolor="rgba(37, 99, 235, 0.12)",
        selector=dict(type="scatter", fill="tozeroy"),
    )
    fig.update_traces(
        fillcolor="rgba(124, 58, 237, 0.10)",
        selector=dict(type="scatter", fill="tonexty"),
    )
    if len(fig.data) > 1:
        fig.update_traces(showlegend=True)
    else:
        fig.update_traces(showlegend=False, selector=dict(type="bar"))

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
        notes = list(getattr(fig, "_llm_data_analyst_tool_notes", []))
        if notes:
            payload["tool_result_note"] = "\n".join(str(note) for note in notes if note)
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
    description: str = tool_description("plotly_tool")
    forbidden_code_patterns: ClassVar[tuple[tuple[str, str], ...]] = (
        *_PLOTLY_FORBIDDEN_CODE_PATTERNS,
        (
            r"\.plot\.(?:bar|line|pie|scatter|area|hist)\s*\(",
            "pandas .plot.* запрещён. Используй px.bar / px.pie / go.Bar.",
        ),
    )
    allowed_libs: set[str] = {
        "plotly", "pandas", "numpy",
        "datetime", "math", "statistics", "calendar", "collections", "itertools", "re",
    }
    allowed_artifact_types: tuple = (go.Figure,)
    def __init__(
        self,
        df: pd.DataFrame,
        execution_timeout_sec: float = 25.0,
        tool_cache_size: int = 48,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
        sandbox: Any | None = None,
    ) -> None:
        _ = (px, go)  # imports остаются для явной зависимости инструмента
        super().__init__(
            df,
            execution_timeout_sec=execution_timeout_sec,
            include_plotly=True,
            tool_cache_size=tool_cache_size,
            db_runtime_config=db_runtime_config,
            sandbox=sandbox,
        )

    def get_execution_scope(self) -> dict[str, Any]:
        scope: dict[str, Any] = {
            "chart": ChartArtifactHelper(tool_name=self.name),
        }
        return scope
