import type { ArtifactPayload, ChatMessage } from "./backend-types";
import { formatDurationMs } from "./format";

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

function renderInlineMarkdown(text: string): string {
  const placeholders: string[] = [];
  let escaped = escapeHtml(text);
  escaped = escaped.replace(/`([^`]+)`/g, (_, code: string) => {
    const token = `@@CODE_${placeholders.length}@@`;
    placeholders.push(`<code>${code}</code>`);
    return token;
  });
  escaped = escaped.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  escaped = escaped.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  escaped = escaped.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  escaped = escaped.replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  escaped = escaped.replace(/(^|[\s(])_([^_\n]+)_/g, "$1<em>$2</em>");
  placeholders.forEach((html, index) => {
    escaped = escaped.replace(`@@CODE_${index}@@`, html);
  });
  return escaped;
}

function isMarkdownTableSeparator(line: string): boolean {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function renderMarkdownTable(lines: string[]): string {
  const rows = lines.map((line) =>
    line
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim()),
  );
  const header = rows[0] ?? [];
  const body = rows.slice(2);
  return `
    <div class="md-table-wrap">
      <table class="md-table">
        <thead><tr>${header.map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join("")}</tr></thead>
        <tbody>
          ${body
            .map((row) => `<tr>${row.map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>`)
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderMarkdown(text: string): string {
  const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
  const blocks: string[] = [];
  let paragraph: string[] = [];
  let listItems: string[] = [];
  let orderedItems: string[] = [];
  let codeLines: string[] | null = null;

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };
  const flushLists = () => {
    if (listItems.length) {
      blocks.push(`<ul>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
      listItems = [];
    }
    if (orderedItems.length) {
      blocks.push(`<ol>${orderedItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ol>`);
      orderedItems = [];
    }
  };

  for (let i = 0; i < lines.length; i += 1) {
    const raw = lines[i] ?? "";
    const line = raw.trimEnd();

    if (codeLines) {
      if (/^\s*```/.test(line)) {
        blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = null;
      } else {
        codeLines.push(raw);
      }
      continue;
    }

    if (/^\s*```/.test(line)) {
      flushParagraph();
      flushLists();
      codeLines = [];
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushLists();
      continue;
    }

    if (line.includes("|") && i + 1 < lines.length && isMarkdownTableSeparator(lines[i + 1] ?? "")) {
      flushParagraph();
      flushLists();
      const tableLines = [line, lines[i + 1] ?? ""];
      i += 2;
      while (i < lines.length && (lines[i] ?? "").includes("|") && (lines[i] ?? "").trim()) {
        tableLines.push(lines[i] ?? "");
        i += 1;
      }
      i -= 1;
      blocks.push(renderMarkdownTable(tableLines));
      continue;
    }

    const heading = /^(#{1,4})\s+(.+)$/.exec(line.trim());
    if (heading) {
      flushParagraph();
      flushLists();
      const level = Math.min(4, heading[1].length);
      blocks.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = /^\s*[-*+]\s+(.+)$/.exec(line);
    if (bullet) {
      flushParagraph();
      orderedItems = [];
      listItems.push(bullet[1]);
      continue;
    }

    const ordered = /^\s*\d+[.)]\s+(.+)$/.exec(line);
    if (ordered) {
      flushParagraph();
      listItems = [];
      orderedItems.push(ordered[1]);
      continue;
    }

    const quote = /^\s*>\s+(.+)$/.exec(line);
    if (quote) {
      flushParagraph();
      flushLists();
      blocks.push(`<blockquote>${renderInlineMarkdown(quote[1])}</blockquote>`);
      continue;
    }

    paragraph.push(line.trim());
  }

  if (codeLines) {
    blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  }
  flushParagraph();
  flushLists();
  return blocks.join("\n");
}

function buildTableHtml(artifact: ArtifactPayload): string {
  const raw = artifact.data.data as { columns?: unknown[]; index?: unknown[]; data?: unknown[][] };
  const columns = Array.isArray(raw.columns) ? raw.columns : [];
  const index = Array.isArray(raw.index) ? raw.index : [];
  const rows = Array.isArray(raw.data) ? raw.data : [];
  const hasMeaningfulIndex =
    index.length === rows.length &&
    index.some((value, idx) => String(value) !== String(idx));

  const head = columns.map((column) => `<th>${escapeHtml(String(column))}</th>`).join("");
  const body = rows
    .map((row, rowIdx) => {
      const cells = row.map((cell) => `<td>${escapeHtml(String(cell))}</td>`).join("");
      const indexCell = hasMeaningfulIndex ? `<td>${escapeHtml(String(index[rowIdx]))}</td>` : "";
      return `<tr>${indexCell}${cells}</tr>`;
    })
    .join("");
  const indexHead = hasMeaningfulIndex ? "<th>Индекс</th>" : "";

  return `
    <div class="table-wrap">
      <table>
        <thead><tr>${indexHead}${head}</tr></thead>
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
        ? `<div class="chips"><span>${formatDurationMs(message.metrics.duration_ms)}</span><span>${message.metrics.artifact_count} artifacts</span></div>`
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

          if (artifact.type === "note" && artifact.data.format === "markdown") {
            const content = String((artifact.data.data as { content?: unknown })?.content ?? "");
            return `
              <div class="artifact-item">
                <div class="artifact-head"><strong>${escapeHtml(artifactTitle)}</strong><span>${escapeHtml(artifact.type)}</span></div>
                <div class="markdown-body">${renderMarkdown(content)}</div>
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
          <div class="markdown-body">${renderMarkdown(message.content)}</div>
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
    .wrap{ width:100%; max-width:none; margin:0; padding:16px; display:grid; gap:14px; }
    .head{ border:1px solid var(--line); border-radius:16px; background:var(--panel); padding:16px; }
    .head h1{ margin:0 0 4px; font-size:24px; }
    .head p{ margin:0; color:var(--muted); }
    .msg{ border:1px solid var(--line); border-radius:14px; background:var(--panel); padding:14px 16px; display:grid; gap:10px; width:100%; }
    .msg-user{ background:var(--user); }
    .msg-role{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
    pre{ margin:0; white-space:pre-wrap; overflow:auto; background:#f4f7ff; border:1px solid var(--line); border-radius:10px; padding:10px; }
    .markdown-body{ line-height:1.55; display:block; max-width:100%; }
    .markdown-body > *{ margin:0; }
    .markdown-body > * + *{ margin-top:8px; }
    .markdown-body h1{ font-size:22px; line-height:1.25; margin-top:4px; }
    .markdown-body h2{ font-size:19px; line-height:1.3; margin-top:4px; }
    .markdown-body h3{ font-size:16px; line-height:1.35; margin-top:3px; }
    .markdown-body h4{ font-size:14px; line-height:1.35; margin-top:2px; }
    .markdown-body ul,.markdown-body ol{ padding-left:20px; }
    .markdown-body li + li{ margin-top:4px; }
    .markdown-body blockquote{ margin:0; border-left:3px solid #9db3ff; padding:6px 10px; background:#f4f7ff; color:var(--muted); border-radius:8px; }
    .markdown-body code{ border:1px solid var(--line); border-radius:6px; background:#edf2ff; padding:1px 4px; font-family:"JetBrains Mono","Consolas",monospace; font-size:.92em; }
    .markdown-body pre code{ border:0; background:transparent; padding:0; }
    .markdown-body a{ color:#2563eb; font-weight:700; text-decoration:none; }
    .md-table-wrap{ overflow:auto; margin-top:8px; }
    .md-table{ width:100%; border-collapse:collapse; font-size:13px; }
    .md-table th,.md-table td{ border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
    .md-table th{ background:#eef2ff; }
    .chips{ display:flex; flex-wrap:wrap; gap:8px; }
    .chips span{ padding:3px 8px; border:1px solid var(--line); border-radius:999px; font-size:12px; color:var(--muted); }
    .artifacts{ display:grid; gap:8px; }
    .artifact-item{ border:1px solid var(--line); border-radius:10px; padding:8px; background:#fbfcff; }
    .artifact-head{ display:flex; justify-content:space-between; gap:8px; color:var(--muted); font-size:12px; }
    .plot-wrap { width: 100%; min-height: 280px; border: 1px solid var(--line); border-radius: 10px; background: #fff; }
    .table-wrap { overflow: auto; width:100%; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; table-layout:auto; }
    th, td { border: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align:top; }
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
