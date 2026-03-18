import type { ArtifactPayload, ChatMessage } from "./backend-types";

function saveFile(filename: string, payload: string, mimeType: string): void {
  const blob = new Blob([payload], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function buildTableHtml(artifact: ArtifactPayload): string {
  const raw = artifact.data.data as { columns?: unknown[]; index?: unknown[]; data?: unknown[][] };
  const columns = Array.isArray(raw.columns) ? raw.columns : [];
  const index = Array.isArray(raw.index) ? raw.index : [];
  const rows = Array.isArray(raw.data) ? raw.data : [];

  const head = columns.map((column) => `<th>${escapeHtml(String(column))}</th>`).join("");
  const body = rows
    .map((row, rowIdx) => {
      const cells = row.map((cell) => `<td>${escapeHtml(String(cell))}</td>`).join("");
      return `<tr><td>${escapeHtml(String(index[rowIdx] ?? rowIdx))}</td>${cells}</tr>`;
    })
    .join("");

  return `
    <div class="table-wrap">
      <table>
        <thead><tr><th>#</th>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
  `;
}

function buildValueHtml(artifact: ArtifactPayload): string {
  const values = artifact.data.data as Record<string, unknown>;
  const rows = Object.entries(values)
    .map(
      ([key, value]) =>
        `<div class="value-row"><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd></div>`,
    )
    .join("");
  return `<dl class="value-grid">${rows}</dl>`;
}

function buildChatHistoryHtml(
  sessionId: string,
  sessionTitle: string,
  datasetName: string,
  messages: ChatMessage[],
): string {
  const plotScripts: string[] = [];
  const renderedMessages = messages
    .map((message, messageIndex) => {
      const metrics = message.metrics
        ? `<div class="chips"><span>${message.metrics.duration_ms} ms</span><span>${message.metrics.artifact_count} artifacts</span><span>${escapeHtml(message.metrics.model)}</span></div>`
        : "";
      const reasoning = message.reasoning
        ? `<details><summary>Рассуждение</summary><pre>${escapeHtml(message.reasoning)}</pre></details>`
        : "";
      const artifacts = (message.artifacts || [])
        .map((artifact, artifactIndex) => {
          const artifactTitle = artifact.text || artifact.type;
          const code = typeof artifact.meta?.code === "string" ? artifact.meta.code : "";
          const codeBlock = code
            ? `<details><summary>Код артефакта</summary><pre>${escapeHtml(code)}</pre></details>`
            : "";

          if (artifact.type === "plot" && artifact.data.format === "plotly-json") {
            const payload = artifact.data.data as {
              data?: unknown[];
              layout?: Record<string, unknown>;
              config?: Record<string, unknown>;
            };
            const plotId = `chat_plot_${messageIndex}_${artifactIndex}`;
            plotScripts.push(`
              Plotly.newPlot(
                "${plotId}",
                ${JSON.stringify(payload.data || [])},
                Object.assign(
                  { autosize: true, margin: { l: 42, r: 18, t: 38, b: 38 }, paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff", font: { color: "#1f2a44" } },
                  ${JSON.stringify(payload.layout || {})}
                ),
                Object.assign({ responsive: true, displaylogo: false }, ${JSON.stringify(payload.config || {})})
              );
            `);
            return `
              <div class="artifact-item">
                <div class="artifact-head"><strong>${escapeHtml(artifactTitle)}</strong><span>${escapeHtml(artifact.type)}</span></div>
                <div class="plot-wrap" id="${plotId}"></div>
                ${codeBlock}
              </div>
            `;
          }

          if (artifact.type === "table" && artifact.data.format === "split") {
            return `
              <div class="artifact-item">
                <div class="artifact-head"><strong>${escapeHtml(artifactTitle)}</strong><span>${escapeHtml(artifact.type)}</span></div>
                ${buildTableHtml(artifact)}
                ${codeBlock}
              </div>
            `;
          }

          if (artifact.type === "value" && artifact.data.format === "value") {
            return `
              <div class="artifact-item">
                <div class="artifact-head"><strong>${escapeHtml(artifactTitle)}</strong><span>${escapeHtml(artifact.type)}</span></div>
                ${buildValueHtml(artifact)}
                ${codeBlock}
              </div>
            `;
          }

          return `
            <div class="artifact-item">
              <div class="artifact-head"><strong>${escapeHtml(artifactTitle)}</strong><span>${escapeHtml(artifact.type)}</span></div>
              <pre>${escapeHtml(JSON.stringify(artifact.data.data, null, 2))}</pre>
              ${codeBlock}
            </div>
          `;
        })
        .join("\n");

      return `
        <article class="msg msg-${message.role}">
          <div class="msg-role">${message.role === "user" ? "Пользователь" : "Агент"}</div>
          <pre>${escapeHtml(message.content)}</pre>
          ${reasoning}
          ${metrics}
          ${artifacts ? `<section class="artifacts">${artifacts}</section>` : ""}
        </article>
      `;
    })
    .join("\n");

  const generatedAt = new Date().toLocaleString("ru-RU");
  return `<!doctype html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Chat ${escapeHtml(sessionTitle)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root { --line:#d8def2; --ink:#17243b; --muted:#5e6f90; --panel:#fff; --bg:#f6f8ff; --user:#eef4ff; }
    *{ box-sizing:border-box; }
    body{ margin:0; background:var(--bg); color:var(--ink); font-family:"Manrope", "Segoe UI", sans-serif; }
    .wrap{ max-width:1100px; margin:0 auto; padding:24px; display:grid; gap:14px; }
    .head{ border:1px solid var(--line); border-radius:16px; background:var(--panel); padding:16px; }
    .head h1{ margin:0 0 4px; font-size:24px; }
    .head p{ margin:0; color:var(--muted); }
    .msg{ border:1px solid var(--line); border-radius:14px; background:var(--panel); padding:12px; display:grid; gap:10px; }
    .msg-user{ background:var(--user); }
    .msg-role{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
    pre{ margin:0; white-space:pre-wrap; overflow:auto; background:#f4f7ff; border:1px solid var(--line); border-radius:10px; padding:10px; }
    .chips{ display:flex; flex-wrap:wrap; gap:8px; }
    .chips span{ padding:3px 8px; border:1px solid var(--line); border-radius:999px; font-size:12px; color:var(--muted); }
    .artifacts{ display:grid; gap:8px; }
    .artifact-item{ border:1px solid var(--line); border-radius:10px; padding:8px; background:#fbfcff; }
    .artifact-head{ display:flex; justify-content:space-between; gap:8px; color:var(--muted); font-size:12px; }
    .plot-wrap { width: 100%; min-height: 280px; border: 1px solid var(--line); border-radius: 10px; background: #fff; }
    .table-wrap { overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border: 1px solid var(--line); padding: 6px 8px; text-align: left; }
    th { background: #eef2ff; }
    .value-grid { margin: 0; display: grid; gap: 8px; }
    .value-row { display: grid; grid-template-columns: 1fr auto; gap: 10px; border: 1px solid var(--line); border-radius: 10px; padding: 8px; }
    .value-row dt { color: var(--muted); }
    .value-row dd { margin: 0; font-weight: 700; }
    @media (max-width:860px){ .wrap{ padding:12px; } }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="head">
      <h1>${escapeHtml(sessionTitle)}</h1>
      <p>session: ${escapeHtml(sessionId)} | dataset: ${escapeHtml(datasetName || "dataset.csv")} | exported: ${escapeHtml(generatedAt)}</p>
    </section>
    ${renderedMessages || "<p>История пуста.</p>"}
  </main>
  <script>
    ${plotScripts.join("\n")}
  </script>
</body>
</html>`;
}

export function exportChatHistory(
  sessionId: string,
  sessionTitle: string,
  datasetName: string,
  messages: ChatMessage[],
): void {
  if (!sessionId) {
    return;
  }
  const html = buildChatHistoryHtml(sessionId, sessionTitle, datasetName, messages);
  saveFile(`chat_${sessionId.slice(0, 8)}.html`, html, "text/html;charset=utf-8");
}
