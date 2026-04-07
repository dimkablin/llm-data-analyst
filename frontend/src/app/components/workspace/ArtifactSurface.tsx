import { useEffect, useRef } from "react";
import { useTheme } from "next-themes";
import type { ArtifactPayload } from "../../lib/backend-types";
import { formatNumber } from "../../lib/format";

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

function ValueArtifact({ artifact }: { artifact: ArtifactPayload }) {
  const values = artifact.data.data as Record<string, unknown>;
  return (
    <dl className="grid gap-2">
      {Object.entries(values).map(([key, value]) => (
        <div key={`${artifact.id}-${key}`} className="grid grid-cols-[1fr_auto] gap-3 rounded-xl border border-border/40 bg-background/30 px-3 py-2">
          <dt className="truncate text-xs font-semibold uppercase tracking-wider text-muted-foreground">{key}</dt>
          <dd className="text-sm font-bold">{formatCellValue(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function TableArtifact({ artifact }: { artifact: ArtifactPayload }) {
  const raw = artifact.data.data as { columns?: unknown[]; index?: unknown[]; data?: unknown[][] };
  const columns = Array.isArray(raw.columns) ? raw.columns.map(String) : [];
  const index = Array.isArray(raw.index) ? raw.index : [];
  const rows = Array.isArray(raw.data) ? raw.data : [];

  return (
    <div className="overflow-x-auto rounded-2xl border border-border/40">
      <table className="w-full min-w-[480px] border-collapse text-left text-sm">
        <thead className="bg-secondary/30">
          <tr>
            <th className="px-3 py-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">#</th>
            {columns.map((column) => (
              <th key={column} className="px-3 py-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border/30">
          {rows.map((row, rowIdx) => (
            <tr key={`${artifact.id}-${rowIdx}`}>
              <td className="px-3 py-2 text-muted-foreground">{formatCellValue(index[rowIdx] ?? rowIdx)}</td>
              {row.map((cell, cellIdx) => (
                <td key={`${artifact.id}-${rowIdx}-${cellIdx}`} className="px-3 py-2">
                  {formatCellValue(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function JsonArtifact({ artifact }: { artifact: ArtifactPayload }) {
  const data = artifact.data.data as Record<string, unknown>;
  const answer = typeof data?.answer === "string" ? data.answer.trim() : null;
  const query = typeof data?.query === "string" ? data.query.trim() : null;
  const results = Array.isArray(data?.results) ? data.results as Array<Record<string, unknown>> : null;
  const references = Array.isArray(data?.references) ? data.references as string[] : null;
  const sources = Array.isArray(data?.sources) ? data.sources as string[] : null;

  const isSearchResult = results !== null;
  const isRagResult = references !== null && !isSearchResult;

  if (isSearchResult || isRagResult) {
    return (
      <div className="space-y-3">
        {query && (
          <p className="text-xs text-muted-foreground">
            <span className="font-semibold uppercase tracking-wider">Запрос: </span>{query}
          </p>
        )}
        {answer && (
          <div className="rounded-xl border border-border/40 bg-background/30 px-3 py-2 text-sm">
            {answer}
          </div>
        )}
        {results && results.length > 0 && (
          <ol className="space-y-2">
            {results.slice(0, 8).map((r, i) => (
              <li key={i} className="rounded-xl border border-border/30 bg-background/20 px-3 py-2 text-sm">
                <a
                  href={typeof r.url === "string" ? r.url : undefined}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-semibold text-primary hover:underline"
                >
                  {typeof r.title === "string" ? r.title : `Результат ${i + 1}`}
                </a>
                {typeof r.snippet === "string" && r.snippet && (
                  <p className="mt-0.5 text-xs text-muted-foreground line-clamp-2">{r.snippet}</p>
                )}
                {typeof r.source_name === "string" && r.source_name && (
                  <p className="mt-0.5 text-xs text-muted-foreground/60">{r.source_name}</p>
                )}
              </li>
            ))}
          </ol>
        )}
        {(references ?? sources ?? []).length > 0 && (
          <div className="space-y-1">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Источники</p>
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
    <pre className="overflow-auto rounded-2xl border border-border/40 bg-background/40 p-4 text-xs">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

function PlotArtifact({ artifact }: { artifact: ArtifactPayload }) {
  const { resolvedTheme } = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const isDark = resolvedTheme === "dark";

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const payload = artifact.data.data as {
      data?: unknown[];
      layout?: Record<string, unknown>;
      config?: Record<string, unknown>;
    };

    const frameBg = isDark ? "#09090b" : "#ffffff";
    const plotBg  = isDark ? "#18181b" : "#ffffff";
    const text    = isDark ? "#fafafa" : "#18181b";
    const muted   = isDark ? "#a1a1aa" : "#717182";
    const grid    = isDark ? "rgba(63,63,70,0.5)"  : "rgba(0,0,0,0.10)";
    const zero    = isDark ? "#3f3f46"              : "rgba(0,0,0,0.18)";

    const baseLayout = (payload.layout || {}) as Record<string, unknown>;
    const layout = {
      ...baseLayout,
      paper_bgcolor: frameBg,
      plot_bgcolor:  plotBg,
      font:   { ...(baseLayout.font   as object || {}), color: text,  family: "ui-sans-serif, system-ui, sans-serif" },
      legend: { ...(baseLayout.legend as object || {}), bgcolor: "transparent", font: { color: muted } },
      xaxis:  { ...(baseLayout.xaxis  as object || {}), gridcolor: grid, zerolinecolor: zero, tickfont: { color: muted } },
      yaxis:  { ...(baseLayout.yaxis  as object || {}), gridcolor: grid, zerolinecolor: zero, tickfont: { color: muted } },
      autosize: true,
      margin: { l: 44, r: 28, t: 44, b: 44 },
    };

    let cancelled = false;
    import("plotly.js-dist-min").then((Plotly) => {
      if (cancelled || !containerRef.current) return;
      Plotly.newPlot(
        container,
        (payload.data || []) as Plotly.Data[],
        layout as Partial<Plotly.Layout>,
        { responsive: true, displaylogo: false, scrollZoom: true, ...(payload.config || {}) },
      );
    });

    return () => {
      cancelled = true;
      import("plotly.js-dist-min").then((Plotly) => Plotly.purge(container));
    };
  }, [artifact.id, isDark]);

  return (
    <div
      ref={containerRef}
      className="h-[400px] w-full rounded-2xl border border-border/40"
    />
  );
}

export function ArtifactSurface({
  artifact,
  showCode = false,
}: {
  artifact: ArtifactPayload;
  showCode?: boolean;
}) {
  const code = typeof artifact.meta?.code === "string" ? artifact.meta.code : "";

  return (
    <article className="rounded-3xl border border-border/50 bg-card p-5 shadow-sm">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h4 className="text-base font-bold tracking-tight">{artifact.text || artifact.type}</h4>
          <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">{artifact.type}</p>
        </div>
      </div>

      {artifact.type === "plot" && artifact.data.format === "plotly-json" ? <PlotArtifact artifact={artifact} /> : null}
      {artifact.type === "table" && artifact.data.format === "split" ? <TableArtifact artifact={artifact} /> : null}
      {artifact.type === "value" && artifact.data.format === "value" ? <ValueArtifact artifact={artifact} /> : null}
      {artifact.type === "json" && artifact.data.format === "json" ? <JsonArtifact artifact={artifact} /> : null}
      {!(
        (artifact.type === "plot" && artifact.data.format === "plotly-json") ||
        (artifact.type === "table" && artifact.data.format === "split") ||
        (artifact.type === "value" && artifact.data.format === "value") ||
        (artifact.type === "json" && artifact.data.format === "json")
      ) ? (
        <pre className="overflow-auto rounded-2xl border border-border/40 bg-background/40 p-4 text-xs">
          {JSON.stringify(artifact.data.data, null, 2)}
        </pre>
      ) : null}

      {showCode && code ? (
        <details className="mt-4 rounded-2xl border border-border/40 bg-background/30 p-4">
          <summary className="cursor-pointer text-sm font-semibold">Код артефакта</summary>
          <pre className="mt-3 overflow-auto text-xs">{code}</pre>
        </details>
      ) : null}
    </article>
  );
}
