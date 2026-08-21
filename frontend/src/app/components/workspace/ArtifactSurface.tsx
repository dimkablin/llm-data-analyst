import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { useTheme } from "next-themes";
import type { ArtifactPayload } from "../../lib/backend-types";
import { formatNumber } from "../../lib/format";
import { plotlySequence } from "../../lib/plotly-data";
import { MarkdownBlock } from "../MarkdownBlock";

function formatCellValue(value: unknown): string {
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "string") {
    return value.trim() || "—";
  }
  return String(value);
}

type ArtifactVariant = "default" | "board";

function ValueArtifact({
  artifact,
  variant = "default",
}: {
  artifact: ArtifactPayload;
  variant?: ArtifactVariant;
}) {
  const values = artifact.data.data as Record<string, unknown>;
  const isBoard = variant === "board";
  return (
    <dl className={isBoard ? "divide-y divide-border/15" : "grid gap-2"}>
      {Object.entries(values).map(([key, value]) => (
        <div
          key={`${artifact.id}-${key}`}
          className={
            isBoard
              ? "grid grid-cols-[1fr_auto] gap-3 px-1 py-2"
              : "grid grid-cols-[1fr_auto] gap-3 rounded-xl border border-border/40 bg-background/30 px-3 py-2"
          }
        >
          <dt className="truncate text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            {key}
          </dt>
          <dd className="text-sm font-bold">{formatCellValue(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function TableArtifact({
  artifact,
  variant = "default",
}: {
  artifact: ArtifactPayload;
  variant?: ArtifactVariant;
}) {
  const raw = artifact.data.data as { columns?: unknown[]; index?: unknown[]; data?: unknown[][] };
  const columns = Array.isArray(raw.columns) ? raw.columns.map(String) : [];
  const index = Array.isArray(raw.index) ? raw.index : [];
  const rows = Array.isArray(raw.data) ? raw.data : [];
  const isBoard = variant === "board";
  const [sort, setSort] = useState<{ column: number; direction: "asc" | "desc" } | null>(null);
  const isPlanfactVarianceTable =
    artifact.meta?.source_type === "planfact" &&
    ["planfact_first_look_cfo_summary", "planfact_first_look_article_summary"].includes(artifact.id);
  const statusColumn = columns.indexOf("Статус");
  const hiddenBoardColumns = isBoard && artifact.id === "planfact_first_look_article_summary"
    ? new Set(["Содержание услуги", "Контрагент план", "Контрагент факт", "Договор"])
    : new Set<string>();
  const visibleColumnIndexes = columns
    .map((column, columnIndex) => ({ column, columnIndex }))
    .filter(({ column }) => !hiddenBoardColumns.has(column));
  const sortedRows = useMemo(() => {
    const paired = rows.map((row, rowIndex) => ({ row, rowIndex }));
    if (!isBoard || !sort) return paired;
    const sortValue = (value: unknown): string | number => {
      if (typeof value === "number") return value;
      const text = String(value ?? "").trim();
      const match = text.replace(/\s/g, "").replace(",", ".").match(/^([+-]?\d+(?:\.\d+)?)/);
      if (!match) return text.toLocaleLowerCase("ru");
      const scale = /млрд/i.test(text) ? 1_000_000_000 : /млн/i.test(text) ? 1_000_000 : /тыс/i.test(text) ? 1_000 : 1;
      return Number(match[1]) * scale;
    };
    return paired.sort((left, right) => {
      const leftValue = sortValue(sort.column < 0 ? index[left.rowIndex] ?? left.rowIndex : left.row[sort.column]);
      const rightValue = sortValue(sort.column < 0 ? index[right.rowIndex] ?? right.rowIndex : right.row[sort.column]);
      const result = typeof leftValue === "number" && typeof rightValue === "number"
        ? leftValue - rightValue
        : String(leftValue).localeCompare(String(rightValue), "ru", { numeric: true });
      return sort.direction === "asc" ? result : -result;
    });
  }, [index, isBoard, rows, sort]);

  function toggleSort(column: number): void {
    if (!isBoard) return;
    setSort((current) => current?.column === column
      ? { column, direction: current.direction === "asc" ? "desc" : "asc" }
      : { column, direction: "asc" });
  }

  function sortLabel(column: number): string {
    if (sort?.column !== column) return "";
    return sort.direction === "asc" ? " ↑" : " ↓";
  }

  return (
    <div className={isBoard ? "overflow-x-auto" : "overflow-x-auto rounded-2xl border border-border/40"}>
      <table className="w-full min-w-[480px] border-collapse text-left text-sm">
        <thead className={isBoard ? "border-b border-border/20" : "bg-secondary/30"}>
          <tr>
            <th
              className="px-3 py-2 text-xs font-bold uppercase tracking-wider text-muted-foreground"
              aria-sort={sort?.column === -1 ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}
            >
              <button type="button" onClick={() => toggleSort(-1)} className={isBoard ? "cursor-pointer whitespace-nowrap hover:text-foreground" : "cursor-default"}>
                #{sortLabel(-1)}
              </button>
            </th>
            {visibleColumnIndexes.map(({ column, columnIndex }) => (
              <th
                key={column}
                className="px-3 py-2 text-xs font-bold uppercase tracking-wider text-muted-foreground"
                aria-sort={sort?.column === columnIndex ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}
              >
                <button type="button" onClick={() => toggleSort(columnIndex)} className={isBoard ? "cursor-pointer whitespace-nowrap hover:text-foreground" : "cursor-default"}>
                  {column}{sortLabel(columnIndex)}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border/30">
          {sortedRows.map(({ row, rowIndex }) => (
            <tr
              key={`${artifact.id}-${rowIndex}`}
              className={
                isPlanfactVarianceTable && statusColumn >= 0
                  ? ["Превышение", "Факт без плана"].includes(String(row[statusColumn] ?? ""))
                    ? "bg-rose-500/5"
                    : ["Экономия", "План без факта"].includes(String(row[statusColumn] ?? ""))
                      ? "bg-emerald-500/5"
                      : ""
                  : ""
              }
            >
              <td className="px-3 py-2 text-muted-foreground">
                {formatCellValue(index[rowIndex] ?? rowIndex)}
              </td>
              {visibleColumnIndexes.map(({ columnIndex }) => (
                <td key={`${artifact.id}-${rowIndex}-${columnIndex}`} className="px-3 py-2">
                  {formatCellValue(row[columnIndex])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function JsonArtifact({
  artifact,
  variant = "default",
}: {
  artifact: ArtifactPayload;
  variant?: ArtifactVariant;
}) {
  const data = artifact.data.data as Record<string, unknown>;
  const answer = typeof data?.answer === "string" ? data.answer.trim() : null;
  const query = typeof data?.query === "string" ? data.query.trim() : null;
  const results = Array.isArray(data?.results) ? (data.results as Array<Record<string, unknown>>) : null;
  const references = Array.isArray(data?.references) ? (data.references as string[]) : null;
  const sources = Array.isArray(data?.sources) ? (data.sources as string[]) : null;

  const isSearchResult = results !== null;
  const isRagResult = references !== null && !isSearchResult;

  if (isSearchResult || isRagResult) {
    return (
      <div className="space-y-3 text-sm">
        {query && (
          <p className="text-xs leading-relaxed text-muted-foreground">
            <span className="font-semibold uppercase tracking-wider text-foreground/70">Запрос: </span>
            {query}
          </p>
        )}
        {answer && (
          <div className="text-sm font-medium leading-relaxed text-foreground">
            {answer}
          </div>
        )}
        {results && results.length > 0 && (
          <ol className="divide-y divide-border/20 border-t border-border/15">
            {results.slice(0, 8).map((r, i) => (
              <li
                key={i}
                className="py-2.5 text-sm"
              >
                <a
                  href={typeof r.url === "string" ? r.url : undefined}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-semibold text-primary hover:underline"
                >
                  {typeof r.title === "string" ? r.title : `Результат ${i + 1}`}
                </a>
                {typeof r.snippet === "string" && r.snippet && (
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground line-clamp-2">{r.snippet}</p>
                )}
                {typeof r.source_name === "string" && r.source_name && (
                  <p className="mt-0.5 text-xs text-muted-foreground/60">{r.source_name}</p>
                )}
              </li>
            ))}
          </ol>
        )}
        {(references ?? sources ?? []).length > 0 && (
          <div className="space-y-1 border-t border-border/15 pt-2">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Источники
            </p>
            <ul className="space-y-1">
              {(references ?? sources ?? []).map((ref, i) => (
                <li key={i} className="text-xs">
                  <a
                    href={ref}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline break-all"
                  >
                    {ref}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  }

  return (
    <pre
      className={
        variant === "board"
          ? "overflow-auto bg-transparent p-1 text-xs"
          : "overflow-auto rounded-2xl border border-border/40 bg-background/40 p-4 text-xs"
      }
    >
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

const PLOTLY_COLORWAY = [
  "#2563eb", // blue
  "#7c3aed", // violet
  "#0f766e", // teal
  "#ea580c", // orange
] as const;

const MAX_BAR_VALUE_LABELS = 6;

function formatCompactNumber(value: number): string {
  const absValue = Math.abs(value);
  if (absValue >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(1).replace(/\.0$/, "")} млрд`;
  }
  if (absValue >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1).replace(/\.0$/, "")} млн`;
  }
  if (absValue >= 10_000) {
    return `${(value / 1_000).toFixed(1).replace(/\.0$/, "")} тыс`;
  }
  if (absValue >= 100) {
    return value.toFixed(0);
  }
  if (absValue >= 1) {
    return value.toFixed(1).replace(/\.0$/, "");
  }
  return value.toFixed(2).replace(/\.00$/, "");
}

function buildBarLabelTexts(values: number[]): string[] | null {
  if (values.length > MAX_BAR_VALUE_LABELS) {
    return null;
  }
  const total = values.reduce((sum, value) => sum + value, 0);
  if (total <= 0) {
    return null;
  }
  const maxValue = Math.max(...values);
  return values.map((value) => {
    if (values.length <= 5 && total > 100) {
      return `${((100 * value) / total).toFixed(1)}%`;
    }
    if (total <= 100.5 && maxValue <= 100) {
      return `${value.toFixed(1)}%`;
    }
    return formatCompactNumber(value);
  });
}

function numericBarValues(trace: PlotlyTraceLike): number[] {
  const orientation = String(trace.orientation ?? "").toLowerCase();
  const raw = orientation === "h" ? trace.x : trace.y;
  return plotlySequence(raw).map((item) => {
    const numeric = Number(item);
    return Number.isFinite(numeric) ? numeric : 0;
  });
}

type PlotlyTraceLike = Record<string, unknown> & {
  name?: unknown;
  type?: unknown;
  mode?: unknown;
  fill?: unknown;
  line?: Record<string, unknown>;
  marker?: Record<string, unknown>;
  fillcolor?: unknown;
};

function withAlpha(hex: string, alpha: number): string {
  const normalized = hex.replace("#", "");
  const full =
    normalized.length === 3
      ? normalized
          .split("")
          .map((c) => c + c)
          .join("")
      : normalized;

  const value = Number.parseInt(full, 16);
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;

  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function departmentAbbreviation(value: unknown): string {
  const clean = String(value ?? "").trim();
  const words = clean.match(/[A-Za-zА-Яа-яЁё0-9]+/g) ?? [];
  const stopWords = new Set(["и", "в", "во", "для", "на", "по", "с", "со"]);
  const significant = words.filter((word) => !stopWords.has(word.toLowerCase()));
  return significant.length > 1 && clean.length > 12
    ? significant.map((word) => word[0]).join("").toUpperCase()
    : clean;
}

function resolveTraceColor(nameRaw: unknown, index: number): string {
  const name = String(nameRaw ?? "").toLowerCase();

  if (/(anomaly|outlier|alert|аномал)/.test(name)) {
    return "#dc2626";
  }

  if (/(forecast|prediction|pred|yhat|plan|expected|прогноз|план)/.test(name)) {
    return "#7c3aed";
  }

  if (/(fact|actual|real|observed|факт)/.test(name) || name === "y") {
    return "#2563eb";
  }

  if (/(lower|upper|bound|interval|confidence|band|ci)/.test(name)) {
    return "#94a3b8";
  }

  return PLOTLY_COLORWAY[index % PLOTLY_COLORWAY.length];
}

function normalizePlotlyTraces(
  traces: unknown[],
  isDark: boolean,
): { traces: Plotly.Data[]; hasBarLabels: boolean } {
  let hasBarLabels = false;
  const normalized = traces.map((raw, index) => {
    const trace = { ...((raw as PlotlyTraceLike) ?? {}) };
    const name = String(trace.name ?? "").toLowerCase();
    const traceType = String(trace.type ?? "scatter").toLowerCase();
    const fill = String(trace.fill ?? "").toLowerCase();
    const color = resolveTraceColor(trace.name, index);

    const isBand =
      /(lower|upper|bound|interval|confidence|band|ci)/.test(name) ||
      fill === "tonexty" ||
      fill === "tozeroy";

    if (traceType === "scatter" || traceType === "") {
      trace.line = {
        ...(trace.line ?? {}),
        color,
        width: isBand ? 1.6 : 2.4,
      };

      if (String(trace.mode ?? "").includes("markers")) {
        trace.marker = {
          ...(trace.marker ?? {}),
          color,
          line: { width: 0 },
        };
      }

      if (isBand) {
        trace.fillcolor = withAlpha(color, isDark ? 0.18 : 0.12);
      }
    } else if (traceType === "bar" || traceType === "histogram") {
      const barOrientation = String(trace.orientation ?? "").toLowerCase();
      const primary = barOrientation === "h" ? trace.x : trace.y;
      const pointCount = Math.max(plotlySequence(primary).length, 1);
      const barColors = Array.from({ length: Math.max(pointCount, 1) }, (_, barIndex) =>
        resolveTraceColor(trace.name, index + barIndex),
      );
      trace.marker = {
        ...(trace.marker ?? {}),
        color: barColors.length > 1 ? barColors : barColors[0],
        line: { width: 0 },
        opacity: 0.9,
        cornerradius: 6,
      };
      const labelTexts = buildBarLabelTexts(numericBarValues(trace));
      if (labelTexts) {
        trace.text = labelTexts;
        trace.textposition = "outside";
        trace.textfont = { size: 10, color: isDark ? "#d4d4d8" : "#475569" };
        trace.cliponaxis = false;
        hasBarLabels = true;
      }
    }

    return trace as Plotly.Data;
  });
  return { traces: normalized, hasBarLabels };
}

function PlotArtifact({
  artifact,
  contentHeightPx,
  variant = "default",
}: {
  artifact: ArtifactPayload;
  contentHeightPx?: number;
  variant?: ArtifactVariant;
}) {
  const { resolvedTheme } = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const plotlyRef = useRef<any>(null);
  const isDark = resolvedTheme === "dark";

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const payload = artifact.data.data as {
      data?: unknown[];
      layout?: Record<string, unknown>;
      config?: Record<string, unknown>;
    };

    let legacyWaterfallTicks: { tickvals: string[]; ticktext: string[] } | null = null;
    const rawTraces: PlotlyTraceLike[] = [];
    for (const raw of Array.isArray(payload.data) ? payload.data : []) {
      const trace = { ...((raw as PlotlyTraceLike) ?? {}) };
      const values = plotlySequence(trace.x).map(String);
      if (String(trace.type).toLowerCase() !== "waterfall" || new Set(values).size === values.length) {
        rawTraces.push(trace);
        continue;
      }
      const tickvals = values.map((_, index) => `step-${index}`);
      legacyWaterfallTicks = { tickvals, ticktext: values };
      rawTraces.push({ ...trace, x: tickvals });
    }
    const { traces, hasBarLabels } = normalizePlotlyTraces(rawTraces, isDark);
    const multiSeries = traces.length > 1;

    const frameBg = isDark ? "#09090b" : "#ffffff";
    const plotBg = frameBg;
    const text = isDark ? "#fafafa" : "#18181b";
    const muted = isDark ? "#a1a1aa" : "#717182";
    const grid = isDark ? "rgba(63,63,70,0.5)" : "rgba(0,0,0,0.10)";
    const zero = isDark ? "#3f3f46" : "rgba(0,0,0,0.18)";

    const baseLayout = (payload.layout || {}) as Record<string, unknown>;
    const waterfallTicktext = artifact.id === "planfact_first_look_plan_to_fact_waterfall"
      ? plotlySequence(rawTraces[0]?.customdata).map((value) => {
          const fullLabel = String(value);
          if (fullLabel === "План" || fullLabel === "Факт") return fullLabel;
          if (fullLabel === "Прочие статьи") return "Прочие";
          return departmentAbbreviation(fullLabel.split(" · ")[0]);
        })
      : [];
    const showLegend = artifact.id === "planfact_first_look_variance_donut"
      || multiSeries
      || baseLayout.showlegend === true;
    const titleOverride = artifact.id === "planfact_first_look_plan_to_fact_waterfall"
      ? "Вклад статей в отклонение"
      : null;
    const layout = {
      ...baseLayout,
      width: undefined,
      height: undefined,
      colorway: [...PLOTLY_COLORWAY],
      paper_bgcolor: frameBg,
      plot_bgcolor: plotBg,
      font: {
        ...((baseLayout.font as object) || {}),
        color: text,
        family: "ui-sans-serif, system-ui, sans-serif",
      },
      title: {
        ...((baseLayout.title as object) || {}),
        ...(titleOverride ? { text: titleOverride } : {}),
        font: {
          ...(((baseLayout.title as Record<string, unknown>)?.font as object) || {}),
          color: text,
          family: "ui-sans-serif, system-ui, sans-serif",
        },
      },
      showlegend: showLegend,
      legend: showLegend
        ? {
            ...((baseLayout.legend as object) || {}),
            ...(multiSeries
              ? { orientation: "h", yanchor: "top", y: -0.2, xanchor: "center", x: 0.5 }
              : {}),
            bgcolor: "transparent",
            font: {
              ...(((baseLayout.legend as Record<string, unknown>)?.font as object) || {}),
              color: muted,
            },
          }
        : { ...(baseLayout.legend as object), traceorder: "normal" },
      xaxis: {
        ...((baseLayout.xaxis as object) || {}),
        ...(legacyWaterfallTicks
          ? { tickmode: "array", tickvals: legacyWaterfallTicks.tickvals, ticktext: legacyWaterfallTicks.ticktext }
          : {}),
        ...(waterfallTicktext.length ? { ticktext: waterfallTicktext } : {}),
        automargin: true,
        gridcolor: grid,
        zerolinecolor: zero,
        tickfont: {
          ...(((baseLayout.xaxis as Record<string, unknown>)?.tickfont as object) || {}),
          color: muted,
        },
        title: {
          ...((((baseLayout.xaxis as Record<string, unknown>)?.title as Record<string, unknown>) || {})),
          font: {
            ...((((((baseLayout.xaxis as Record<string, unknown>)?.title as Record<string, unknown>) || {})
              .font as object) || {})),
            color: text,
          },
        },
      },
      yaxis: {
        ...((baseLayout.yaxis as object) || {}),
        automargin: true,
        gridcolor: grid,
        zerolinecolor: zero,
        tickfont: {
          ...(((baseLayout.yaxis as Record<string, unknown>)?.tickfont as object) || {}),
          color: muted,
        },
        title: {
          ...((((baseLayout.yaxis as Record<string, unknown>)?.title as Record<string, unknown>) || {})),
          font: {
            ...((((((baseLayout.yaxis as Record<string, unknown>)?.title as Record<string, unknown>) || {})
              .font as object) || {})),
            color: text,
          },
        },
      },
      annotations: Array.isArray((baseLayout as Record<string, unknown>).annotations)
        ? ((baseLayout as Record<string, unknown>).annotations as Array<Record<string, unknown>>).map(
            (ann) => ({
              ...ann,
              font: {
                ...((ann.font as object) || {}),
                color: text,
              },
            }),
          )
        : (baseLayout as Record<string, unknown>).annotations,
      autosize: true,
      margin: {
        l: 44,
        r: hasBarLabels ? 40 : 28,
        t: hasBarLabels ? 64 : 44,
        b: multiSeries ? 72 : 44,
        ...(baseLayout.margin as object || {}),
      },
    };

    let cancelled = false;
    let resizeObserver: ResizeObserver | null = null;

    import("plotly.js-dist-min").then((Plotly) => {
      if (cancelled || !containerRef.current) return;

      plotlyRef.current = Plotly;

      Plotly.newPlot(container, traces, layout as Partial<Plotly.Layout>, {
        responsive: true,
        displaylogo: false,
        scrollZoom: true,
        ...(payload.config || {}),
      }).then(() => {
        if (cancelled || !containerRef.current || !plotlyRef.current) return;

        resizeObserver = new ResizeObserver(() => {
          if (!containerRef.current || !plotlyRef.current) return;
          plotlyRef.current.Plots.resize(containerRef.current);
        });

        resizeObserver.observe(containerRef.current);
        plotlyRef.current.Plots.resize(containerRef.current);
      });
    });

    return () => {
      cancelled = true;
      resizeObserver?.disconnect();

      if (containerRef.current && plotlyRef.current) {
        plotlyRef.current.purge(containerRef.current);
      }
    };
  }, [artifact.id, isDark]);

  return (
    <div
      ref={containerRef}
      className={
        variant === "board"
          ? "w-full min-w-0 overflow-hidden"
          : "w-full min-w-0 overflow-hidden rounded-2xl border border-border/40"
      }
      style={{ height: `${contentHeightPx ?? 420}px` }}
    />
  );
}

function ArtifactContent({
  artifact,
  contentHeightPx,
  variant = "default",
}: {
  artifact: ArtifactPayload;
  contentHeightPx?: number;
  variant?: ArtifactVariant;
}) {
  const effectiveNonPlotHeight = contentHeightPx ?? 380;
  const nonPlotStyle = { maxHeight: `${effectiveNonPlotHeight}px` };
  const nonPlotClass = "overflow-auto pr-1";

  return (
    <>
      {artifact.type === "plot" && artifact.data.format === "plotly-json" ? (
        <PlotArtifact artifact={artifact} contentHeightPx={contentHeightPx} variant={variant} />
      ) : null}
      {artifact.type === "table" && artifact.data.format === "split" ? (
        <div className={nonPlotClass} style={nonPlotStyle}>
          <TableArtifact artifact={artifact} variant={variant} />
        </div>
      ) : null}
      {artifact.type === "value" && artifact.data.format === "value" ? (
        <div className={nonPlotClass} style={nonPlotStyle}>
          <ValueArtifact artifact={artifact} variant={variant} />
        </div>
      ) : null}
      {artifact.type === "json" && artifact.data.format === "json" ? (
        <div className={nonPlotClass} style={nonPlotStyle}>
          <JsonArtifact artifact={artifact} variant={variant} />
        </div>
      ) : null}
      {artifact.type === "note" && artifact.data.format === "markdown" ? (
        <div className={nonPlotClass} style={nonPlotStyle}>
          {(() => {
            const noteContent = String(
              (artifact.data.data as { content?: unknown })?.content ?? "",
            );
            return (
              <div
                className={
                  variant === "board"
                    ? "px-1 py-1 text-sm leading-relaxed"
                    : "rounded-2xl border border-border/40 bg-background p-4 text-sm leading-relaxed"
                }
              >
                <MarkdownBlock content={noteContent} className="management-note" />
              </div>
            );
          })()}
        </div>
      ) : null}
      {!(
        (artifact.type === "plot" && artifact.data.format === "plotly-json") ||
        (artifact.type === "table" && artifact.data.format === "split") ||
        (artifact.type === "value" && artifact.data.format === "value") ||
        (artifact.type === "json" && artifact.data.format === "json") ||
        (artifact.type === "note" && artifact.data.format === "markdown")
      ) ? (
        <div className={nonPlotClass} style={nonPlotStyle}>
          <pre
            className={
              variant === "board"
                ? "overflow-auto bg-transparent p-1 text-xs"
                : "overflow-auto rounded-2xl border border-border/40 bg-background/40 p-4 text-xs"
            }
          >
            {JSON.stringify(artifact.data.data, null, 2)}
          </pre>
        </div>
      ) : null}
    </>
  );
}

function ArtifactCodeBlock({ code }: { code: string }) {
  return (
    <details className="mt-3 rounded-2xl bg-background/30 p-4">
      <summary className="cursor-pointer text-sm font-semibold">Код артефакта</summary>
      <pre className="mt-3 overflow-auto text-xs">{code}</pre>
    </details>
  );
}

export function ArtifactSurface({
  artifact,
  showCode = false,
  contentHeightPx,
  variant = "default",
  headerAction,
}: {
  artifact: ArtifactPayload;
  showCode?: boolean;
  contentHeightPx?: number;
  variant?: "default" | "board";
  headerAction?: ReactNode;
}) {
  const code = typeof artifact.meta?.code === "string" ? artifact.meta.code : "";

  if (variant === "board") {
    return (
      <div className="min-w-0">
        <div className="overflow-hidden rounded-2xl border border-border/40 bg-background/20">
          <ArtifactContent artifact={artifact} contentHeightPx={contentHeightPx} variant={variant} />
        </div>
        {showCode && code ? <ArtifactCodeBlock code={code} /> : null}
      </div>
    );
  }

  return (
    <article className="overflow-hidden rounded-3xl border border-border/50 bg-card p-5 shadow-sm">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h4 className="text-base font-bold tracking-tight">{artifact.text || artifact.type}</h4>
          <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
            {artifact.type}
          </p>
        </div>
        {headerAction ? <div className="shrink-0">{headerAction}</div> : null}
      </div>

      <ArtifactContent artifact={artifact} contentHeightPx={contentHeightPx} variant={variant} />

      {showCode && code ? <ArtifactCodeBlock code={code} /> : null}
    </article>
  );
}
