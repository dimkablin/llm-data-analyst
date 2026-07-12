from __future__ import annotations

import copy
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from docx import Document
from docx.shared import Inches, Pt
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.llm_client import make_reasoning_llm
from backend.core.config import settings

_ARTIFACT_RE = re.compile(
    r"\[\s*artifact\s*:\s*([A-Za-z0-9_\-]+)\s*\]",
    re.IGNORECASE,
)
_BOLD_HEADING_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")
_INLINE_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")

# Same idea as frontend ArtifactSurface / PlotArtifact normalization.
# DOCX export uses light theme by default because Word pages are white.
_PLOTLY_COLORWAY: tuple[str, ...] = (
    "#2563eb",  # blue
    "#7c3aed",  # violet
    "#0f766e",  # teal
    "#ea580c",  # orange
)


@dataclass
class ReportBuildResult:
    file_name: str
    file_path: str
    download_url: str


def _artifact_name_from_message_refs(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _is_plotly_format(value: str) -> bool:
    normalized = str(value or "").strip().lower().replace("_", "-")
    return normalized == "plotly-json"


def _safe_file_stem(value: str, fallback: str = "chart") -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")
    return stem or fallback


def _build_artifact_index(artifacts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for art in artifacts:
        if not isinstance(art, dict):
            continue

        art_type = str(art.get("type") or art.get("presentation_type") or "").strip().lower()
        title = str(art.get("text") or art.get("title") or "").strip()
        data = art.get("data") if isinstance(art.get("data"), dict) else {}
        data_format = str(data.get("format") or "").strip().lower()
        payload = data.get("data")
        meta = art.get("meta") if isinstance(art.get("meta"), dict) else {}

        artifact_name = ""
        if isinstance(meta.get("artifact_name"), str) and str(meta.get("artifact_name")).strip():
            artifact_name = str(meta["artifact_name"]).strip()
        elif title:
            artifact_name = title

        if not artifact_name:
            continue

        result[artifact_name] = {
            "artifact_name": artifact_name,
            "artifact_type": art_type or "unknown",
            "title": title or artifact_name,
            "data_format": data_format,
            "payload": payload,
        }
    return result


def _plot_context_block(artifact: dict[str, Any]) -> str:
    payload = artifact.get("payload")
    fig = None
    try:
        if isinstance(payload, dict):
            fig = go.Figure(payload)
    except Exception:
        fig = None

    chart_title = artifact.get("title") or artifact.get("artifact_name") or "-"
    x_title = "-"
    y_title = "-"
    trace_types = "unknown"
    series_names = "-"

    if fig is not None:
        try:
            layout = fig.layout
            chart_title = getattr(getattr(layout, "title", None), "text", "") or chart_title
            x_title = getattr(getattr(getattr(layout, "xaxis", None), "title", None), "text", "") or "-"
            y_title = getattr(getattr(getattr(layout, "yaxis", None), "title", None), "text", "") or "-"
            trace_types = ", ".join(sorted({getattr(trace, "type", "unknown") for trace in fig.data})) or "unknown"  # noqa: E501
            names = [str(getattr(trace, "name", "")).strip() for trace in fig.data]
            names = [x for x in names if x]
            if names:
                series_names = ", ".join(names)
        except Exception:
            pass

    return (
        f"- artifact_name: {artifact['artifact_name']}\n"
        f"  artifact_type: plot\n"
        f"  chart_title: {chart_title}\n"
        f"  x_title: {x_title}\n"
        f"  y_title: {y_title}\n"
        f"  trace_types: {trace_types}\n"
        f"  series_names: {series_names}"
    )


def _build_prompt(chat_history: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> str:
    artifact_index = _build_artifact_index(artifacts)

    lines: list[str] = []
    lines.append("[ИСТОРИЯ ЧАТА]")
    for idx, msg in enumerate(chat_history, start=1):
        if not isinstance(msg, dict):
            continue

        role = str(msg.get("role") or "").strip()
        content = str(msg.get("content") or "").strip()
        if not content:
            continue

        lines.append(f"{idx}. role={role}")
        lines.append(content)

        linked_names = _artifact_name_from_message_refs(msg.get("artifacts"))
        linked_plot_names = [name for name in linked_names if name in artifact_index]

        if linked_plot_names:
            lines.append("Связанные графики:")
            for name in linked_plot_names:
                art = artifact_index[name]
                if _is_plotly_format(str(art.get("data_format") or "")):
                    lines.append(_plot_context_block(art))
        lines.append("")

    lines.append("[ВСЕ ДОСТУПНЫЕ ГРАФИКИ]")
    for art in artifact_index.values():
        if not _is_plotly_format(str(art.get("data_format") or "")):
            continue
        lines.append(_plot_context_block(art))
        lines.append("")

    return "\n".join(lines).strip()


def _build_llm():
    return make_reasoning_llm(
        provider=settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        enable_thinking=False,
        temperature=getattr(settings, "llm_temperature_chat", 0.3),
        max_tokens=getattr(settings, "llm_max_tokens_default", 4096),
        streaming=False,
        timeout=max(60.0, float(getattr(settings, "agent_step_timeout_sec", 60.0))),
        top_p=1.0,
        top_k=getattr(settings, "llm_top_k", 0),
        num_ctx=getattr(settings, "llm_num_ctx", 0),
        presence_penalty=0.0,
        chat_template_kwargs_enabled=getattr(settings, "llm_chat_template_kwargs_enabled", False),
    )


_SYSTEM_PROMPT = """
Ты пишешь деловой отчет по истории чата и связанным графикам.

Правила:
1. Пиши отчет на русском языке.
2. Используй markdown-подобный формат.
3. Заголовки разделов выделяй строго так: **Текст заголовка**
4. Для вставки графиков используй только формат: [artifact:artifact_name]
5. Используй только реально доступные artifact_name из контекста.
6. Не придумывай новые артефакты.
7. Не вставляй таблицы.
8. Структура: **Название отчета**, краткое введение, основные выводы, рекомендации.
9. Возвращай только текст отчета без пояснений.
""".strip()


def _validate_artifact_refs(report_text: str, artifacts: list[dict[str, Any]]) -> None:
    index = _build_artifact_index(artifacts)
    allowed = set(index.keys())

    found = [name.strip() for name in _ARTIFACT_RE.findall(report_text)]
    invented = sorted({name for name in found if name not in allowed})

    if invented:
        raise ValueError(f"LLM invented artifact names: {invented}")


def _with_alpha(hex_color: str, alpha: float) -> str:
    normalized = str(hex_color or "").replace("#", "").strip()
    if len(normalized) == 3:
        normalized = "".join(ch * 2 for ch in normalized)

    try:
        value = int(normalized, 16)
    except ValueError:
        return f"rgba(148, 163, 184, {alpha})"

    r = (value >> 16) & 255
    g = (value >> 8) & 255
    b = value & 255
    return f"rgba({r}, {g}, {b}, {alpha})"


def _resolve_trace_color(name_raw: Any, index: int) -> str:
    name = str(name_raw or "").lower()

    if re.search(r"(anomaly|outlier|alert|аномал)", name):
        return "#dc2626"

    if re.search(r"(forecast|prediction|pred|yhat|plan|expected|прогноз|план)", name):
        return "#7c3aed"

    if re.search(r"(fact|actual|real|observed|факт)", name) or name == "y":
        return "#2563eb"

    if re.search(r"(lower|upper|bound|interval|confidence|band|ci)", name):
        return "#94a3b8"

    return _PLOTLY_COLORWAY[index % len(_PLOTLY_COLORWAY)]


def _normalize_plotly_trace(raw_trace: Any, index: int, *, is_dark: bool = False) -> dict[str, Any]:
    trace = dict(raw_trace or {}) if isinstance(raw_trace, dict) else {}

    name = str(trace.get("name") or "").lower()
    trace_type = str(trace.get("type") or "scatter").lower()
    mode = str(trace.get("mode") or "").lower()
    fill = str(trace.get("fill") or "").lower()

    color = _resolve_trace_color(trace.get("name"), index)

    is_band = (
        re.search(r"(lower|upper|bound|interval|confidence|band|ci)", name) is not None
        or fill in {"tonexty", "tozeroy"}
    )

    if trace_type in {"scatter", "scattergl", ""}:
        base_line = trace.get("line") if isinstance(trace.get("line"), dict) else {}
        trace["line"] = {
            **base_line,
            "color": color,
            "width": 1.6 if is_band else 2.4,
        }

        if "markers" in mode:
            base_marker = trace.get("marker") if isinstance(trace.get("marker"), dict) else {}
            trace["marker"] = {
                **base_marker,
                "color": color,
                "line": {"width": 0},
            }

        if is_band:
            trace["fillcolor"] = _with_alpha(color, 0.18 if is_dark else 0.12)

    elif trace_type in {"bar", "histogram"}:
        base_marker = trace.get("marker") if isinstance(trace.get("marker"), dict) else {}
        trace["marker"] = {
            **base_marker,
            "color": color,
            "line": {"width": 0},
        }

    return trace


def _normalize_axis_layout(
    raw_axis: Any,
    *,
    text: str,
    muted: str,
    grid: str,
    zero: str,
) -> dict[str, Any]:
    axis = dict(raw_axis or {}) if isinstance(raw_axis, dict) else {}

    tickfont = axis.get("tickfont") if isinstance(axis.get("tickfont"), dict) else {}
    title = axis.get("title") if isinstance(axis.get("title"), dict) else {}
    title_font = title.get("font") if isinstance(title.get("font"), dict) else {}

    return {
        **axis,
        "automargin": True,
        "gridcolor": grid,
        "zerolinecolor": zero,
        "tickfont": {
            **tickfont,
            "color": muted,
        },
        "title": {
            **title,
            "font": {
                **title_font,
                "color": text,
            },
        },
    }


def _normalize_annotations(raw_annotations: Any, *, text: str) -> Any:
    if not isinstance(raw_annotations, list):
        return raw_annotations

    normalized: list[dict[str, Any]] = []
    for raw_ann in raw_annotations:
        if not isinstance(raw_ann, dict):
            continue

        font = raw_ann.get("font") if isinstance(raw_ann.get("font"), dict) else {}
        normalized.append({
            **raw_ann,
            "font": {
                **font,
                "color": text,
            },
        })

    return normalized


def _style_plotly_payload_for_report(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Apply frontend-like Plotly styling before static PNG export.

    Frontend renders charts inside a responsive card, so it can remove fixed
    width/height. DOCX export needs deterministic image dimensions, therefore
    width/height are set here explicitly.
    """
    is_dark = False

    frame_bg = "#09090b" if is_dark else "#ffffff"
    plot_bg = frame_bg
    text = "#fafafa" if is_dark else "#18181b"
    muted = "#a1a1aa" if is_dark else "#717182"
    grid = "rgba(63,63,70,0.5)" if is_dark else "rgba(0,0,0,0.10)"
    zero = "#3f3f46" if is_dark else "rgba(0,0,0,0.18)"

    normalized = copy.deepcopy(payload)

    raw_data = normalized.get("data")
    traces = raw_data if isinstance(raw_data, list) else []
    normalized["data"] = [
        _normalize_plotly_trace(trace, index, is_dark=is_dark)
        for index, trace in enumerate(traces)
    ]

    base_layout = normalized.get("layout")
    base_layout = dict(base_layout or {}) if isinstance(base_layout, dict) else {}

    base_font = base_layout.get("font") if isinstance(base_layout.get("font"), dict) else {}

    base_title = base_layout.get("title") if isinstance(base_layout.get("title"), dict) else {}
    base_title_font = base_title.get("font") if isinstance(base_title.get("font"), dict) else {}

    base_legend = base_layout.get("legend") if isinstance(base_layout.get("legend"), dict) else {}
    base_legend_font = base_legend.get("font") if isinstance(base_legend.get("font"), dict) else {}

    base_legend_title = base_legend.get("title") if isinstance(base_legend.get("title"), dict) else {}
    base_legend_title_font = (
        base_legend_title.get("font")
        if isinstance(base_legend_title.get("font"), dict)
        else {}
    )

    base_margin = base_layout.get("margin") if isinstance(base_layout.get("margin"), dict) else {}

    normalized["layout"] = {
        **base_layout,
        "width": 1200,
        "height": 720,
        "autosize": False,
        "colorway": list(_PLOTLY_COLORWAY),
        "paper_bgcolor": frame_bg,
        "plot_bgcolor": plot_bg,
        "font": {
            **base_font,
            "color": text,
            "family": "ui-sans-serif, system-ui, sans-serif",
        },
        "title": {
            **base_title,
            "font": {
                **base_title_font,
                "color": text,
                "family": "ui-sans-serif, system-ui, sans-serif",
            },
        },
        "legend": {
            **base_legend,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {
                **base_legend_font,
                "color": muted,
            },
            "title": {
                **base_legend_title,
                "font": {
                    **base_legend_title_font,
                    "color": muted,
                },
            },
        },
        "xaxis": _normalize_axis_layout(
            base_layout.get("xaxis"),
            text=text,
            muted=muted,
            grid=grid,
            zero=zero,
        ),
        "yaxis": _normalize_axis_layout(
            base_layout.get("yaxis"),
            text=text,
            muted=muted,
            grid=grid,
            zero=zero,
        ),
        "annotations": _normalize_annotations(
            base_layout.get("annotations"),
            text=text,
        ),
        "margin": {
            "l": 44,
            "r": 28,
            "t": 44,
            "b": 44,
            **base_margin,
        },
    }

    return normalized


def _render_plot_png(artifact: dict[str, Any], export_dir: Path) -> Path:
    payload = artifact.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact {artifact['artifact_name']} has no plotly-json payload")

    styled_payload = _style_plotly_payload_for_report(payload)
    fig = go.Figure(styled_payload)

    safe_name = _safe_file_stem(str(artifact.get("artifact_name") or "chart"))
    out_path = export_dir / f"{safe_name}.png"

    fig.write_image(
        str(out_path),
        format="png",
        width=1200,
        height=720,
        scale=2,
    )
    return out_path


def _add_markdownish_paragraph(doc: Document, line: str) -> None:
    line = str(line or "").strip()
    if not line:
        return

    # Full-line bold => heading
    heading_match = _BOLD_HEADING_RE.match(line)
    if heading_match:
        p = doc.add_paragraph()
        run = p.add_run(heading_match.group(1).strip())
        run.bold = True
        run.font.size = Pt(15)
        return

    # Inline **bold** inside normal paragraph
    p = doc.add_paragraph()
    pos = 0

    for match in _INLINE_BOLD_RE.finditer(line):
        if match.start() > pos:
            p.add_run(line[pos:match.start()])

        bold_run = p.add_run(match.group(1))
        bold_run.bold = True

        pos = match.end()

    if pos < len(line):
        p.add_run(line[pos:])


def _render_docx(report_text: str, artifacts: list[dict[str, Any]], out_path: Path) -> None:
    artifact_index = _build_artifact_index(artifacts)
    export_dir = out_path.parent / f"{out_path.stem}_assets"
    export_dir.mkdir(parents=True, exist_ok=True)

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(12)

    for raw_line in report_text.splitlines():
        line = raw_line.strip()

        if not line:
            doc.add_paragraph("")
            continue

        match = _ARTIFACT_RE.fullmatch(line)
        if match:
            artifact_name = match.group(1).strip()
            artifact = artifact_index.get(artifact_name)

            if artifact is None:
                p = doc.add_paragraph()
                run = p.add_run(f"[График не найден: {artifact_name}]")
                run.italic = True
                run.font.size = Pt(10)
                continue

            if not _is_plotly_format(str(artifact.get("data_format") or "")):
                p = doc.add_paragraph()
                run = p.add_run(f"[Артефакт не является графиком: {artifact_name}]")
                run.italic = True
                run.font.size = Pt(10)
                continue

            png_path = _render_plot_png(artifact, export_dir)
            doc.add_picture(str(png_path), width=Inches(6.5))
            continue

        _add_markdownish_paragraph(doc, line)

    doc.save(str(out_path))


def build_report_docx(
    *,
    session_id: str,
    chat_history: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    output_dir: Path,
    base_download_url: str,
) -> ReportBuildResult:
    llm = _build_llm()
    prompt = _build_prompt(chat_history, artifacts)

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    report_text = str(getattr(response, "content", "") or "").strip()
    if not report_text:
        raise ValueError("LLM returned empty report")

    _validate_artifact_refs(report_text, artifacts)

    file_name = f"report_{session_id}_{uuid.uuid4().hex[:8]}.docx"
    file_path = output_dir / file_name
    output_dir.mkdir(parents=True, exist_ok=True)

    _render_docx(report_text, artifacts, file_path)

    return ReportBuildResult(
        file_name=file_name,
        file_path=str(file_path),
        download_url=f"{base_download_url.rstrip('/')}/{file_name}",
    )
