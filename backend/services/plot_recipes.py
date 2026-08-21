"""Autogen plot recipe registry: question intent + table profile → plot specs.

Dataset-agnostic: column roles come from ``chart_type_selector`` and name patterns.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from backend.data_access.dataframe_utils import (
    column_nunique,
    column_series,
    deduplicate_dataframe_columns,
)
from backend.data_access.segment_aggregation import (
    SegmentMetric,
    compute_segment_shares,
    resolve_segment_metric,
)
from backend.services.chart_type_selector import (
    ChartKind,
    PlotIntent,
    TabularPlotProfile,
    infer_tabular_plot_profile,
    is_plausible_time_column,
    pick_chart_kind,
    pick_metric_pair,
    score_dataframe_for_plot,
)

logger = logging.getLogger(__name__)

PlotSpec = tuple[str, Any, dict[str, object]]

PLOT_VALUE_PATTERNS: tuple[str, ...] = (
    "value_mln",
    "price",
    "sales",
    "revenue",
    "amount",
    "turnover",
    "qty",
    "quantity",
    "count",
    "sum",
    "total",
)

PLOT_SEGMENT_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("account_type", "account", "счет", "счёт"),
    ("channel", "канал"),
    ("region", "регион"),
    ("category", "категор"),
    ("brand", "бренд"),
    ("month", "period", "date", "дата"),
)

_TIME_COLUMN_RE = re.compile(
    r"(date|time|month|year|week|месяц|дата|период|dt\b|timestamp)",
    re.IGNORECASE,
)
_CHANNEL_COLUMN_RE = re.compile(
    r"(channel|канал|segment|сегмент|region|регион|brand|бренд|category|категор)",
    re.IGNORECASE,
)


def pick_column(columns: list[str], patterns: tuple[str, ...]) -> str | None:
    lowered = {col: col.lower() for col in columns}
    for pattern in patterns:
        for col, col_lower in lowered.items():
            if pattern in col_lower:
                return col
    return None


def sandbox_var_name(frame_name: str) -> str:
    cleaned = re.sub(r"\W+", "_", str(frame_name or "").strip(), flags=re.UNICODE).strip("_")
    if not cleaned:
        return "df"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned if cleaned.isidentifier() else "df"


def prepare_plot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return deduplicate_dataframe_columns(df)


def pick_time_column(df: pd.DataFrame) -> str | None:
    profile = infer_tabular_plot_profile(df)
    if profile.time_columns:
        return profile.time_columns[0]
    for col in df.columns:
        if is_plausible_time_column(str(col), column_series(df, str(col))):
            return str(col)
    return None


def pick_segment_column(df: pd.DataFrame, *, exclude: set[str]) -> str | None:
    profile = infer_tabular_plot_profile(df)
    for col in profile.dimension_columns:
        if col not in exclude:
            return col
    for col in df.columns:
        name = str(col)
        if name in exclude:
            continue
        if _CHANNEL_COLUMN_RE.search(name):
            return name
    for col in df.columns:
        name = str(col)
        if name in exclude:
            continue
        if df[col].dtype == object or str(df[col].dtype) == "string":
            if 1 < int(df[col].nunique(dropna=True)) <= 24:
                return name
    return None


def pick_plot_segment_columns(df: pd.DataFrame) -> list[str]:
    columns = [str(c) for c in df.columns]
    picked: list[str] = []
    if len(picked) < 2:
        profile = infer_tabular_plot_profile(df)
        for col in profile.dimension_columns:
            if col not in picked:
                picked.append(col)
            if len(picked) >= 4:
                break
    if len(picked) < 2:
        for patterns in PLOT_SEGMENT_PATTERNS:
            col = pick_column(columns, patterns)
            if col and col not in picked and column_nunique(df, col) <= 24:
                picked.append(col)
            if len(picked) >= 4:
                break
    if not picked:
        for col in columns:
            series = column_series(df, col)
            if is_numeric_dtype(series) or is_datetime64_any_dtype(series):
                continue
            if column_nunique(df, col) <= 16:
                picked.append(col)
            if len(picked) >= 3:
                break
    return picked[:4]


def infer_chart_segment_columns(df: pd.DataFrame) -> list[str]:
    picked = pick_plot_segment_columns(prepare_plot_dataframe(df))
    if picked:
        return picked[:4]
    profile = infer_tabular_plot_profile(df)
    if profile.dimension_columns:
        return list(profile.dimension_columns[:4])
    columns = [str(column) for column in df.columns]
    fallback: list[str] = []
    for column in columns:
        series = df[column]
        if is_numeric_dtype(series) or is_datetime64_any_dtype(series):
            continue
        if int(series.nunique(dropna=False)) <= 24:
            fallback.append(column)
        if len(fallback) >= 4:
            break
    return fallback


def pick_row_label_column(df: pd.DataFrame, value_col: str) -> str | None:
    for column in df.columns:
        name = str(column)
        if name == value_col:
            continue
        if is_numeric_dtype(df[column]) or is_datetime64_any_dtype(df[column]):
            continue
        return name
    return None


def aggregate_segment_values(
    df: pd.DataFrame,
    *,
    segment_col: str,
    value_col: str,
    limit: int = 12,
) -> pd.DataFrame | None:
    if segment_col not in df.columns or value_col not in df.columns:
        return None
    agg = df.groupby(segment_col, dropna=False)[value_col].sum().reset_index()
    agg[value_col] = pd.to_numeric(agg[value_col], errors="coerce").fillna(0)
    agg = agg.sort_values(value_col, ascending=False).head(limit)
    if agg.empty:
        return None
    if _metric_allows_signed_values(value_col):
        return agg
    if float(agg[value_col].sum()) <= 0:
        return None
    return agg


def segment_value_breakdown(
    df: pd.DataFrame,
    *,
    segment_col: str,
    value_col: str,
    limit: int = 12,
) -> list[dict[str, object]]:
    """Breakdown with unified SUM-based shares (see ``resolve_segment_metric``)."""
    if segment_col not in df.columns:
        return []
    metric = resolve_segment_metric(df)
    if metric is None and value_col in df.columns:
        metric = SegmentMetric(column=value_col, kind="absolute")
    if metric is None:
        return []
    rows = compute_segment_shares(df, segment_col, metric=metric, limit=limit)
    return [
        {"label": row.label, "value": row.value, "share_pct": row.share_pct}
        for row in rows
    ]


def breakdown_from_plot_figure(fig: Any) -> list[dict[str, object]]:
    import plotly.graph_objects as go

    from backend.tools.impl.plotly_tool import _plotly_sequence

    if not isinstance(fig, go.Figure):
        return []
    rows: list[dict[str, object]] = []
    for trace in fig.data:
        trace_type = str(getattr(trace, "type", "") or "").lower()
        if trace_type == "pie":
            labels = _plotly_sequence(getattr(trace, "labels", None))
            values = pd.to_numeric(
                pd.Series(_plotly_sequence(getattr(trace, "values", None))),
                errors="coerce",
            ).fillna(0)
        elif trace_type == "bar":
            orientation = str(getattr(trace, "orientation", "") or "").lower()
            if orientation == "h":
                labels = _plotly_sequence(getattr(trace, "y", None))
                values = pd.to_numeric(
                    pd.Series(_plotly_sequence(getattr(trace, "x", None))),
                    errors="coerce",
                ).fillna(0)
            else:
                labels = _plotly_sequence(getattr(trace, "x", None))
                values = pd.to_numeric(
                    pd.Series(_plotly_sequence(getattr(trace, "y", None))),
                    errors="coerce",
                ).fillna(0)
        else:
            continue
        total = float(values.sum())
        if total <= 0:
            continue
        for label, value in zip(labels, values, strict=False):
            amount = float(value)
            rows.append(
                {
                    "label": str(label),
                    "value": amount,
                    "share_pct": round(100.0 * amount / total, 2),
                }
            )
    return rows


def build_single_segment_bar_figure(
    df: pd.DataFrame,
    *,
    value_col: str,
    segment_col: str,
    title: str | None = None,
) -> Any | None:
    import plotly.graph_objects as go

    from backend.tools.impl.plotly_tool import apply_default_chart_style

    agg = aggregate_segment_values(df, segment_col=segment_col, value_col=value_col)
    if agg is None:
        return None
    fig = go.Figure(
        go.Bar(
            x=agg[segment_col].astype(str),
            y=agg[value_col],
            name=segment_col,
        )
    )
    fig.update_layout(
        title=title or f"Распределение: {segment_col}",
        height=380,
    )
    fig.update_xaxes(tickangle=-25)
    return apply_default_chart_style(fig)


def build_single_segment_pie_figure(
    df: pd.DataFrame,
    *,
    value_col: str,
    segment_col: str,
    title: str | None = None,
) -> Any | None:
    import plotly.graph_objects as go

    from backend.tools.impl.plotly_tool import CHART_COLORWAY, apply_default_chart_style

    agg = aggregate_segment_values(df, segment_col=segment_col, value_col=value_col)
    if agg is None:
        return None
    labels = agg[segment_col].astype(str).tolist()
    values = agg[value_col].tolist()
    colors = [CHART_COLORWAY[index % len(CHART_COLORWAY)] for index in range(len(labels))]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.35 if len(labels) <= 5 else 0.0,
            marker={"colors": colors},
            textinfo="label+percent",
            textposition="outside",
            sort=False,
        )
    )
    fig.update_layout(
        title=title or f"Доли: {segment_col}",
        height=400,
        margin=dict(t=64, b=32, l=24, r=24),
    )
    return apply_default_chart_style(fig)


def build_single_segment_horizontal_bar_figure(
    df: pd.DataFrame,
    *,
    value_col: str,
    segment_col: str,
    title: str | None = None,
) -> Any | None:
    import plotly.graph_objects as go

    from backend.tools.impl.plotly_tool import apply_default_chart_style

    agg = aggregate_segment_values(df, segment_col=segment_col, value_col=value_col)
    if agg is None:
        return None
    fig = go.Figure(
        go.Bar(
            x=agg[value_col],
            y=agg[segment_col].astype(str),
            orientation="h",
            name=segment_col,
        )
    )
    fig.update_layout(
        title=title or f"Топ: {segment_col}",
        height=400,
        yaxis={"categoryorder": "total ascending"},
    )
    return apply_default_chart_style(fig)


def build_metric_scatter_figure(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    title: str | None = None,
) -> Any | None:
    import plotly.graph_objects as go

    from backend.tools.impl.plotly_tool import apply_default_chart_style

    if x_col not in df.columns or y_col not in df.columns:
        return None
    work = df.copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=[x_col, y_col])
    if work.empty:
        return None
    fig = go.Figure()
    if color_col and color_col in work.columns:
        for seg_value, group in work.groupby(color_col, dropna=False):
            fig.add_trace(
                go.Scatter(
                    x=group[x_col],
                    y=group[y_col],
                    mode="markers",
                    name=str(seg_value),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=work[x_col],
                y=work[y_col],
                mode="markers",
                name=f"{x_col} vs {y_col}",
            )
        )
    fig.update_layout(
        title=title or f"Связь: {x_col} и {y_col}",
        height=420,
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return apply_default_chart_style(fig)


def build_segment_autogen_figure(
    df: pd.DataFrame,
    *,
    value_col: str,
    segment_col: str,
    chart_kind: ChartKind,
    title: str | None = None,
) -> Any | None:
    if chart_kind == "pie":
        return build_single_segment_pie_figure(
            df,
            value_col=value_col,
            segment_col=segment_col,
            title=title,
        )
    if chart_kind == "bar_h":
        return build_single_segment_horizontal_bar_figure(
            df,
            value_col=value_col,
            segment_col=segment_col,
            title=title,
        )
    if chart_kind == "scatter":
        axes = pick_metric_pair(df)
        if axes is None:
            return build_single_segment_bar_figure(
                df,
                value_col=value_col,
                segment_col=segment_col,
                title=title,
            )
        x_col, y_col = axes
        color_col = segment_col if segment_col in df.columns else None
        return build_metric_scatter_figure(
            df,
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
            title=title or "Сравнение метрик",
        )
    return build_single_segment_bar_figure(
        df,
        value_col=value_col,
        segment_col=segment_col,
        title=title,
    )


def build_structure_plot_figure(
    df: pd.DataFrame,
    *,
    value_col: str,
    segment_cols: list[str],
) -> Any | None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from backend.tools.impl.plotly_tool import apply_default_chart_style

    present = [c for c in segment_cols if c in df.columns]
    if not present:
        return None
    ncols = min(4, len(present))
    titles = present[:ncols]
    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles)
    for idx, seg in enumerate(titles, start=1):
        agg = df.groupby(seg, dropna=False)[value_col].sum().reset_index()
        agg[value_col] = pd.to_numeric(agg[value_col], errors="coerce").fillna(0)
        agg = agg.sort_values(value_col, ascending=False).head(12)
        fig.add_trace(
            go.Bar(
                x=agg[seg].astype(str),
                y=agg[value_col],
                name=seg,
                showlegend=False,
            ),
            row=1,
            col=idx,
        )
    fig.update_layout(title="Структура по разрезам", height=460)
    fig.update_xaxes(tickangle=-25)
    return apply_default_chart_style(fig)


def build_dynamics_line_figure(
    df: pd.DataFrame,
    *,
    time_col: str,
    value_col: str,
    segment_col: str | None = None,
) -> Any | None:
    import plotly.graph_objects as go

    from backend.tools.impl.plotly_tool import apply_default_chart_style

    if time_col not in df.columns or value_col not in df.columns:
        return None
    if not is_plausible_time_column(time_col, column_series(df, time_col)):
        return None
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[time_col, value_col])
    if work.empty or work[time_col].nunique(dropna=True) < 2:
        return None
    fig = go.Figure()
    if segment_col and segment_col in work.columns:
        for seg_value, group in work.groupby(segment_col, dropna=False):
            agg = (
                group.groupby(time_col, dropna=False)[value_col]
                .sum()
                .reset_index()
                .sort_values(time_col)
            )
            fig.add_trace(
                go.Scatter(
                    x=agg[time_col].astype(str),
                    y=agg[value_col],
                    mode="lines+markers",
                    name=str(seg_value),
                )
            )
    else:
        agg = (
            work.groupby(time_col, dropna=False)[value_col]
            .sum()
            .reset_index()
            .sort_values(time_col)
        )
        fig.add_trace(
            go.Scatter(
                x=agg[time_col].astype(str),
                y=agg[value_col],
                mode="lines+markers",
                name=value_col,
            )
        )
    if len(fig.data) == 0:
        return None
    fig.update_layout(title="Динамика котировок", height=420)
    fig.update_xaxes(title=time_col)
    fig.update_yaxes(title=value_col)
    return apply_default_chart_style(fig)


def append_row_composition_chart_specs(
    specs: list[PlotSpec],
    *,
    df: pd.DataFrame,
    value_col: str,
    base_meta: dict[str, object],
    intent: PlotIntent,
) -> None:
    if specs or len(df) > 24 or value_col not in df.columns:
        return
    label_col = pick_row_label_column(df, value_col)
    if not label_col:
        return
    breakdown = segment_value_breakdown(
        df,
        segment_col=label_col,
        value_col=value_col,
        limit=12,
    )
    chart_kind = pick_chart_kind(
        intent=intent,
        segment_index=0,
        df=df,
        segment_col=label_col,
    )
    fig = build_segment_autogen_figure(
        df,
        value_col=value_col,
        segment_col=label_col,
        chart_kind=chart_kind,
        title="Структура по сегментам",
    )
    if fig is None:
        return
    specs.append(
        (
            "structure_composition_chart",
            fig,
            {
                **base_meta,
                "segment": label_col,
                "chart_kind": chart_kind,
                "breakdown": breakdown,
            },
        )
    )


def build_autogen_plot_specs(
    df: pd.DataFrame,
    *,
    value_col: str,
    segment_cols: list[str],
    source_table: str,
) -> list[PlotSpec]:
    """Build plot specs from dataframe shape and types."""
    df = prepare_plot_dataframe(df)
    intent: PlotIntent = "generic"
    specs: list[PlotSpec] = []
    base_meta: dict[str, object] = {
        "autogen": True,
        "source_table": source_table,
        "intent": intent,
    }
    if not segment_cols:
        segment_cols = infer_chart_segment_columns(df)

    if segment_cols:
        primary_seg = segment_cols[0]
        chart_kind = pick_chart_kind(
            intent=intent,
            segment_index=0,
            df=df,
            segment_col=primary_seg,
        )
        breakdown = segment_value_breakdown(
            df,
            segment_col=primary_seg,
            value_col=value_col,
        )
        fig = build_segment_autogen_figure(
            df,
            value_col=value_col,
            segment_col=primary_seg,
            chart_kind=chart_kind,
            title=f"Обзор: {primary_seg}",
        )
        if fig is not None:
            specs.append(
                (
                    "structure_breakdown_chart",
                    fig,
                    {
                        **base_meta,
                        "segments": segment_cols,
                        "chart_kind": chart_kind,
                        "breakdown": breakdown,
                    },
                )
            )
            if len(segment_cols) >= 2 and chart_kind != "bar":
                secondary = segment_cols[1]
                secondary_kind = pick_chart_kind(
                    intent="structure",
                    segment_index=1,
                    df=df,
                    segment_col=secondary,
                )
                secondary_fig = build_segment_autogen_figure(
                    df,
                    value_col=value_col,
                    segment_col=secondary,
                    chart_kind=secondary_kind,
                    title=f"Обзор: {secondary}",
                )
                if secondary_fig is not None:
                    safe = sandbox_var_name(secondary)
                    specs.append(
                        (
                            f"structure_by_{safe}_chart",
                            secondary_fig,
                            {
                                **base_meta,
                                "segment": secondary,
                                "chart_kind": secondary_kind,
                                "breakdown": segment_value_breakdown(
                                    df,
                                    segment_col=secondary,
                                    value_col=value_col,
                                ),
                            },
                        )
                    )

    if not specs:
        append_row_composition_chart_specs(
            specs,
            df=df,
            value_col=value_col,
            base_meta=base_meta,
            intent=intent,
        )
    return specs


def pick_value_column(df: pd.DataFrame) -> str | None:
    value_col = pick_column([str(c) for c in df.columns], PLOT_VALUE_PATTERNS)
    if value_col:
        return value_col
    profile = infer_tabular_plot_profile(df)
    if profile.metric_columns:
        return profile.metric_columns[0]
    numeric_cols = [
        str(c)
        for c in df.columns
        if is_numeric_dtype(column_series(df, str(c)))
        and "pct" not in str(c).lower()
    ]
    return numeric_cols[0] if numeric_cols else None


def _is_metadata_frame_name(frame_name: str) -> bool:
    lowered = str(frame_name or "").lower()
    if lowered.startswith("columns_") or lowered.startswith("describe_"):
        return True
    return any(
        token in lowered
        for token in ("db_tables", "describe_", "preview_", "db_schemas")
    )


def _is_lookup_frame_name(frame_name: str) -> bool:
    lowered = str(frame_name or "").lower()
    return "lookup" in lowered


def _is_snapshot_frame(frame_name: str, frame_df: pd.DataFrame) -> bool:
    lowered = str(frame_name or "").lower()
    return bool(len(frame_df) <= 2 and "snapshot" in lowered)


def _frame_suitable_for_timeseries_line(frame_name: str, frame_df: pd.DataFrame) -> bool:
    if _is_metadata_frame_name(frame_name):
        return False
    if _is_snapshot_frame(frame_name, frame_df):
        return False
    profile = infer_tabular_plot_profile(frame_df)
    if not profile.time_columns:
        return False
    time_col = profile.time_columns[0]
    if not is_plausible_time_column(time_col, column_series(frame_df, time_col)):
        return False
    parsed = pd.to_datetime(column_series(frame_df, time_col), errors="coerce").dropna()
    if parsed.empty or int(parsed.nunique()) < 2:
        return False
    if int(parsed.dt.year.min()) < 1990:
        return False
    return bool((parsed.max() - parsed.min()).days >= 1)


def _pick_timeseries_value_column(df: pd.DataFrame, profile: TabularPlotProfile) -> str | None:
    return pick_value_column(df) or (profile.metric_columns[0] if profile.metric_columns else None)


def _slug_artifact_prefix(frame_name: str) -> str:
    clean = re.sub(r"[^a-z0-9_]+", "_", str(frame_name or "").lower()).strip("_")
    return clean or "chart"


def _entity_label_column(df: pd.DataFrame) -> str | None:
    profile = infer_tabular_plot_profile(df)
    for column in profile.dimension_columns:
        if column_nunique(df, column) <= 30:
            return column
    for column in df.columns:
        name = str(column)
        if not is_numeric_dtype(column_series(df, name)) and column_nunique(df, name) <= 30:
            return name
    return None


def _timeseries_signature(frame_df: pd.DataFrame, time_col: str, value_col: str) -> tuple[object, ...]:
    parsed = pd.to_datetime(column_series(frame_df, time_col), errors="coerce")
    values = pd.to_numeric(column_series(frame_df, value_col), errors="coerce")
    mask = parsed.notna() & values.notna()
    if not mask.any():
        return ()
    ordered = values[mask].reset_index(drop=True)
    return (
        int(mask.sum()),
        round(float(ordered.iloc[0]), 4),
        round(float(ordered.iloc[-1]), 4),
    )


def _metric_allows_signed_values(value_col: str) -> bool:
    lowered = str(value_col or "").lower()
    return any(token in lowered for token in ("return", "change", "delta", "pnl", "score"))


def _cross_section_chart_title(value_col: str) -> str:
    return f"Comparison by {value_col}"


def _pick_cross_section_value_column(df: pd.DataFrame) -> str | None:
    return pick_value_column(df)

def _build_timeseries_line_specs(
    usable: list[tuple[str, pd.DataFrame]],
    *,
    intent: PlotIntent,
    base_meta: dict[str, object],
    max_lines: int,
) -> list[PlotSpec]:
    if max_lines <= 0:
        return []
    candidates: list[tuple[int, str, Any, dict[str, object]]] = []
    seen_signatures: set[tuple[object, ...]] = set()
    for frame_name, frame_df in usable:
        if not _frame_suitable_for_timeseries_line(frame_name, frame_df):
            continue
        profile = infer_tabular_plot_profile(frame_df)
        value_col = _pick_timeseries_value_column(frame_df, profile)
        if not value_col or not profile.time_columns:
            continue
        time_col = profile.time_columns[0]
        signature = _timeseries_signature(frame_df, time_col, value_col)
        if signature and signature in seen_signatures:
            continue
        if signature:
            seen_signatures.add(signature)
        segment_col = next(
            None,
        )
        fig = build_dynamics_line_figure(
            frame_df,
            time_col=time_col,
            value_col=value_col,
            segment_col=segment_col,
        )
        if fig is None:
            continue
        prefix = _slug_artifact_prefix(frame_name)
        meta = {
            **base_meta,
            "source_table": frame_name,
            "time_col": time_col,
            "value_col": value_col,
            "chart_kind": "line",
        }
        score = score_dataframe_for_plot(frame_name, frame_df, intent=intent)
        candidates.append((score, f"{prefix}_line", fig, meta))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(name, fig, meta) for _, name, fig, meta in candidates[:max_lines]]


def _build_cross_section_comparison_specs(
    usable: list[tuple[str, pd.DataFrame]],
    *,
    intent: PlotIntent,
    base_meta: dict[str, object],
) -> list[PlotSpec]:
    min_rows = 2
    candidates: list[tuple[int, int, str, pd.DataFrame, str, str]] = []
    for frame_name, frame_df in usable:
        if (
            _is_metadata_frame_name(frame_name)
            or _frame_suitable_for_timeseries_line(frame_name, frame_df)
        ):
            continue
        label_col = _entity_label_column(frame_df)
        if not label_col:
            continue
        value_col = _pick_cross_section_value_column(frame_df)
        if not value_col:
            continue
        numeric = pd.to_numeric(frame_df[value_col], errors="coerce")
        valid = frame_df.loc[numeric.notna()]
        if len(valid) < min_rows:
            continue
        score = score_dataframe_for_plot(frame_name, frame_df, intent="comparison")
        priority = 0
        priority += len(valid)
        candidates.append((priority, score, frame_name, frame_df, label_col, value_col))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, frame_name, frame_df, label_col, value_col = candidates[0]
    plot_df = frame_df.copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col])
    if len(plot_df) < min_rows:
        return []
    if label_col in plot_df.columns:
        plot_df = plot_df.sort_values(value_col, ascending=False)
    chart_kind = pick_chart_kind(
        intent=intent,
        segment_index=0,
        df=plot_df,
        segment_col=label_col,
    )
    title = _cross_section_chart_title(value_col)
    fig = build_segment_autogen_figure(
        plot_df,
        value_col=value_col,
        segment_col=label_col,
        chart_kind=chart_kind,
        title=title,
    )
    if fig is None:
        return []
    prefix = _slug_artifact_prefix(frame_name)
    return [
        (
            f"{prefix}_compare",
            fig,
            {
                **base_meta,
                "source_table": frame_name,
                "segment": label_col,
                "value_col": value_col,
                "chart_kind": chart_kind,
            },
        )
    ]


def build_autogen_plot_specs_for_frames(
    frames: list[tuple[str, pd.DataFrame]],
) -> list[PlotSpec]:
    """Pick the best generic table(s) and run the recipe registry."""
    if not frames:
        return []
    intent: PlotIntent = "generic"
    usable = [
        (name, prepare_plot_dataframe(df))
        for name, df in frames
        if not _is_metadata_frame_name(name)
    ]
    if not usable:
        return []

    specs: list[PlotSpec] = []
    base_meta: dict[str, object] = {"autogen": True, "intent": intent}

    wants_timeseries = any(
        _frame_suitable_for_timeseries_line(name, frame)
        for name, frame in usable
    )
    if wants_timeseries:
        specs.extend(
            _build_timeseries_line_specs(
                usable,
                intent=intent,
                base_meta=base_meta,
                max_lines=1,
            )
        )

    if specs:
        return specs[:3]

    best_name, best_df = max(
        usable,
        key=lambda item: score_dataframe_for_plot(item[0], item[1], intent=intent),
    )
    resolved_metric = resolve_segment_metric(best_df)
    value_col = resolved_metric.column if resolved_metric else pick_value_column(best_df)
    if not value_col:
        logger.info("autogen plots skipped: no numeric value column in %s", best_name)
        return []

    segment_cols = infer_chart_segment_columns(best_df)
    try:
        return build_autogen_plot_specs(
            best_df,
            value_col=value_col,
            segment_cols=segment_cols,
            source_table=best_name,
        )
    except Exception as exc:
        logger.warning("autogen plot build failed: %s", exc, exc_info=True)
        return []
