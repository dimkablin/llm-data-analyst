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

function PlotArtifact({ artifact }: { artifact: ArtifactPayload }) {
  const { resolvedTheme } = useTheme();

  const payload = artifact.data.data as {
    data?: unknown[];
    layout?: Record<string, unknown>;
    config?: Record<string, unknown>;
  };

  const isDark = resolvedTheme === "dark";

  const frameBg = isDark ? "#09090b" : "#ffffff";
  const plotBg = isDark ? "#18181b" : "#ffffff";
  const text = isDark ? "#fafafa" : "#18181b";
  const muted = isDark ? "#a1a1aa" : "#717182";
  const grid = isDark ? "rgba(63,63,70,0.5)" : "rgba(0,0,0,0.10)";
  const zero = isDark ? "#3f3f46" : "rgba(0,0,0,0.18)";

  const srcDoc = `
    <html>
      <head>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>
          html,body,#plot{
            margin:0;
            padding:0;
            width:100%;
            height:100%;
            background:${frameBg};
            color:${text};
            font-family:ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;
          }
        </style>
      </head>
      <body>
        <div id="plot"></div>
        <script>
          const payload = ${JSON.stringify(payload).replace(/<\//g, "<\\/")};
          const baseLayout = payload.layout || {};
          const themeLayout = {
            paper_bgcolor: "${frameBg}",
            plot_bgcolor: "${plotBg}",
            font: {
              ...(baseLayout.font || {}),
              color: "${text}",
              family: "ui-sans-serif, system-ui, sans-serif"
            },
            title: {
              ...(baseLayout.title || {}),
              font: {
                ...(((baseLayout.title || {}).font) || {}),
                color: "${text}"
              }
            },
            legend: {
              ...(baseLayout.legend || {}),
              bgcolor: "transparent",
              font: { ...(((baseLayout.legend || {}).font) || {}), color: "${muted}" }
            },
            xaxis: {
              ...(baseLayout.xaxis || {}),
              gridcolor: "${grid}",
              zerolinecolor: "${zero}",
              tickfont: { ...(((baseLayout.xaxis || {}).tickfont) || {}), color: "${muted}" }
            },
            yaxis: {
              ...(baseLayout.yaxis || {}),
              gridcolor: "${grid}",
              zerolinecolor: "${zero}",
              tickfont: { ...(((baseLayout.yaxis || {}).tickfont) || {}), color: "${muted}" }
            },
            autosize: true,
            margin: { l: 44, r: 28, t: 44, b: 44 },
          };
          Plotly.newPlot(
            "plot",
            payload.data || [],
            Object.assign(themeLayout, baseLayout, {
              paper_bgcolor: themeLayout.paper_bgcolor,
              plot_bgcolor: themeLayout.plot_bgcolor,
              font: themeLayout.font,
              legend: themeLayout.legend,
              xaxis: themeLayout.xaxis,
              yaxis: themeLayout.yaxis,
              autosize: true,
              margin: themeLayout.margin,
            }),
            Object.assign(
              { responsive: true, displaylogo: false, scrollZoom: true },
              payload.config || {},
            )
          );
        </script>
      </body>
    </html>
  `;

  return (
    <iframe
      title={artifact.text || artifact.id}
      className="h-[400px] w-full rounded-2xl border border-border/40 shadow-inner"
      style={{ background: frameBg }}
      srcDoc={srcDoc}
      sandbox="allow-scripts"
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
      {artifact.type !== "plot" || artifact.data.format !== "plotly-json" ? null : null}
      {!(
        (artifact.type === "plot" && artifact.data.format === "plotly-json") ||
        (artifact.type === "table" && artifact.data.format === "split") ||
        (artifact.type === "value" && artifact.data.format === "value")
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
