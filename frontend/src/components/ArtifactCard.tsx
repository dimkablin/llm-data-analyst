import type { ArtifactPayload } from "../types";

type ArtifactCardProps = {
  artifact: ArtifactPayload;
  actionLabel?: string;
  onAction?: (artifact: ArtifactPayload) => void;
  actionDisabled?: boolean;
  showCode?: boolean;
  themeMode?: "light" | "dark";
};

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return String(value);
  }

  const abs = Math.abs(value);
  if (abs >= 1_000_000_000 || (abs > 0 && abs < 0.0001)) {
    return value.toExponential(2);
  }

  if (Number.isInteger(value)) {
    return value.toLocaleString("ru-RU");
  }

  const maximumFractionDigits = abs >= 1000 ? 2 : 4;
  return value.toLocaleString("ru-RU", {
    minimumFractionDigits: 0,
    maximumFractionDigits
  });
}

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

function normalizeLabel(label: string, maxLen = 56): string {
  const clean = label.replace(/_/g, " ").replace(/\s+/g, " ").trim();
  if (!clean) {
    return "—";
  }
  if (clean.length <= maxLen) {
    return clean;
  }
  return `${clean.slice(0, maxLen - 1)}…`;
}

function renderTable(artifact: ArtifactPayload): JSX.Element {
  const raw = artifact.data.data as { columns?: unknown[]; index?: unknown[]; data?: unknown[][] };
  const columns = Array.isArray(raw.columns) ? raw.columns.map(String) : [];
  const index = Array.isArray(raw.index) ? raw.index : [];
  const rows = Array.isArray(raw.data) ? raw.data : [];
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            {columns.map((column) => (
              <th key={column} title={column}>
                {normalizeLabel(column)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIdx) => (
            <tr key={`${artifact.id}-${rowIdx}`}>
              <td title={String(index[rowIdx] ?? rowIdx)}>{formatCellValue(index[rowIdx] ?? rowIdx)}</td>
              {row.map((cell, cellIdx) => (
                <td
                  key={`${artifact.id}-${rowIdx}-${cellIdx}`}
                  className={typeof cell === "number" ? "table-cell-num" : undefined}
                  title={formatCellValue(cell)}
                >
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

function renderBody(artifact: ArtifactPayload, themeMode: "light" | "dark"): JSX.Element {
  if (artifact.type === "plot" && artifact.data.format === "plotly-json") {
    const payload = artifact.data.data as { data?: unknown[]; layout?: Record<string, unknown>; config?: Record<string, unknown> };
    const isDark = themeMode === "dark";
    const plotBg = isDark ? "#121827" : "#ffffff";
    const plotFont = isDark ? "#d9e5ff" : "#1f2a44";
    const paperGrid = isDark ? "#1d2640" : "#e8edf8";
    const srcDoc = `
      <html>
      <head>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>html,body,#plot{margin:0;padding:0;width:100%;height:100%;background:${plotBg};color:${plotFont};font-family:Manrope,Segoe UI,sans-serif}</style>
      </head>
      <body>
        <div id="plot"></div>
        <script>
          const payload = ${JSON.stringify(payload).replace(/<\//g, "<\\/")};
          Plotly.newPlot(
            "plot",
            payload.data || [],
            Object.assign(
              {
                autosize: true,
                margin: {l: 40, r: 20, t: 40, b: 40},
                paper_bgcolor: "${plotBg}",
                plot_bgcolor: "${plotBg}",
                font:{color:"${plotFont}"},
                xaxis: {gridcolor: "${paperGrid}"},
                yaxis: {gridcolor: "${paperGrid}"}
              },
              payload.layout || {}
            ),
            Object.assign({responsive: true, displaylogo: false}, payload.config || {})
          );
        </script>
      </body>
      </html>
    `;
    return <iframe className="plot-frame" srcDoc={srcDoc} title={artifact.text || artifact.id} sandbox="allow-scripts" />;
  }
  if (artifact.type === "table" && artifact.data.format === "split") {
    return renderTable(artifact);
  }
  if (artifact.type === "value" && artifact.data.format === "value") {
    const values = artifact.data.data as Record<string, unknown>;
    return (
      <dl className="value-grid">
        {Object.entries(values).map(([key, value]) => (
          <div key={`${artifact.id}-${key}`}>
            <dt title={key}>{normalizeLabel(key, 44)}</dt>
            <dd title={formatCellValue(value)}>{formatCellValue(value)}</dd>
          </div>
        ))}
      </dl>
    );
  }
  return <pre className="plain-artifact">{JSON.stringify(artifact.data.data, null, 2)}</pre>;
}

export function ArtifactCard({
  artifact,
  actionLabel,
  onAction,
  actionDisabled = false,
  showCode = false,
  themeMode = "light"
}: ArtifactCardProps): JSX.Element {
  const code = typeof artifact.meta?.code === "string" ? artifact.meta.code : "";
  const artifactTitle = String(artifact.text || artifact.type);

  return (
    <article className={`artifact artifact-${artifact.type}`}>
      <header>
        <strong title={artifactTitle}>{normalizeLabel(artifactTitle, 60)}</strong>
        <div className="artifact-actions">
          <span>{artifact.type}</span>
          {onAction ? (
            <button
              type="button"
              className="btn-ghost btn-xs"
              onClick={() => onAction(artifact)}
              disabled={actionDisabled}
            >
              {actionLabel ?? "Добавить"}
            </button>
          ) : null}
        </div>
      </header>
      {renderBody(artifact, themeMode)}
      {showCode && code ? (
        <details className="artifact-code">
          <summary>Код артефакта</summary>
          <pre>{code}</pre>
        </details>
      ) : null}
    </article>
  );
}
