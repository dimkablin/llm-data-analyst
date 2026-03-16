import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  CSSProperties,
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent
} from "react";

import {
  createSession,
  deleteSession,
  generateSessionTitle,
  getRuntimeModelProfile,
  getUserSettings,
  getSession,
  listSessions,
  updateUserSettings,
  updateSessionTitle,
  uploadCsv
} from "../api";
import { ArtifactsPanel, type DashboardLayoutItem } from "../components/ArtifactsPanel";
import { ChatPanel } from "../components/ChatPanel";
import { SettingsSidebar } from "../components/SettingsSidebar";
import { SiteNav } from "../components/SiteNav";
import { useChatAgent } from "../hooks/useChatAgent";
import type { AnalysisDepth, ArtifactPayload, AuthUser, ChatMessage, SessionSummary, UserSettings } from "../types";

const SESSION_POINTER_KEY_PREFIX = "llm_data_analyst_session_id_";
const DATASET_NAME_KEY_PREFIX = "llm_data_analyst_dataset_name_";
const LEGACY_PINNED_KEY_PREFIX = "llm_data_analyst_pinned_";
const DASHBOARD_ORDER_KEY_PREFIX = "llm_data_analyst_dashboard_order_";
const DASHBOARD_LAYOUT_KEY_PREFIX = "llm_data_analyst_dashboard_layout_";
const SETTING_USE_HISTORY_KEY = "llm_data_analyst_setting_use_history_";
const SETTING_REASONING_KEY = "llm_data_analyst_setting_reasoning_";
const SETTING_SHOW_CODE_KEY = "llm_data_analyst_setting_show_code_";
const SETTING_SORT_MODE_KEY = "llm_data_analyst_setting_sort_mode_";
const SETTING_DASHBOARD_COLUMNS_KEY = "llm_data_analyst_setting_dashboard_columns_";
const SETTING_DASHBOARD_LOCK_KEY = "llm_data_analyst_setting_dashboard_lock_";
const SETTING_THEME_KEY = "llm_data_analyst_setting_theme_";
const SETTING_CHAT_WIDTH_KEY = "llm_data_analyst_setting_chat_width_";
const CHAT_COLUMN_DEFAULT_WIDTH = 390;
const CHAT_COLUMN_MIN_WIDTH = 320;
const DASHBOARD_MIN_WIDTH = 420;
const CHAT_SPLITTER_WIDTH = 10;

type SortMode = "newest" | "oldest";
type DashboardColumns = 2 | 3 | 4;
type ThemeMode = "light" | "dark";

type WorkspacePageProps = {
  user: AuthUser;
  onLogout: () => Promise<void>;
  onNavigate: (path: "/" | "/user" | "/technical" | "/app" | "/phoenix") => void;
};

function readBoolSetting(key: string, defaultValue: boolean): boolean {
  const raw = window.localStorage.getItem(key);
  if (raw === null) {
    return defaultValue;
  }
  return raw === "1";
}

function readSortModeSetting(key: string): SortMode {
  const raw = window.localStorage.getItem(key);
  return raw === "oldest" ? "oldest" : "newest";
}

function readDashboardColumnsSetting(key: string): DashboardColumns {
  const raw = window.localStorage.getItem(key);
  if (raw === "2") {
    return 2;
  }
  if (raw === "4") {
    return 4;
  }
  return 3;
}

function readThemeSetting(key: string): ThemeMode {
  const raw = window.localStorage.getItem(key);
  return raw === "light" ? "light" : "dark";
}

function readChatColumnWidthSetting(key: string): number {
  const raw = Number(window.localStorage.getItem(key));
  if (!Number.isFinite(raw) || raw < CHAT_COLUMN_MIN_WIDTH) {
    return CHAT_COLUMN_DEFAULT_WIDTH;
  }
  return Math.round(raw);
}

function sessionPointerKey(userId: number): string {
  return `${SESSION_POINTER_KEY_PREFIX}${userId}`;
}

function datasetNameKey(userId: number, sessionId: string): string {
  return `${DATASET_NAME_KEY_PREFIX}${userId}_${sessionId}`;
}

function dashboardOrderKey(userId: number, sessionId: string): string {
  return `${DASHBOARD_ORDER_KEY_PREFIX}${userId}_${sessionId}`;
}

function dashboardLayoutKey(userId: number, sessionId: string): string {
  return `${DASHBOARD_LAYOUT_KEY_PREFIX}${userId}_${sessionId}`;
}

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
        `<div class="value-row"><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd></div>`
    )
    .join("");
  return `<dl class="value-grid">${rows}</dl>`;
}

type ExportGridLayoutItem = {
  x: number;
  y: number;
  w: number;
  h: number;
};

function defaultExportTileSize(artifact: ArtifactPayload, gridColumns: DashboardColumns): ExportGridLayoutItem {
  const cols = Math.max(1, Number(gridColumns));
  if (artifact.type === "plot") {
    return { x: 0, y: 0, w: Math.min(cols, cols >= 3 ? 2 : cols), h: 9 };
  }
  if (artifact.type === "table") {
    return { x: 0, y: 0, w: Math.min(cols, 2), h: 8 };
  }
  if (artifact.type === "value") {
    return { x: 0, y: 0, w: 1, h: 6 };
  }
  return { x: 0, y: 0, w: 1, h: 6 };
}

function resolveExportLayout(
  artifacts: ArtifactPayload[],
  layout: DashboardLayoutItem[],
  gridColumns: DashboardColumns
): Map<string, ExportGridLayoutItem> {
  const cols = Math.max(1, Number(gridColumns));
  const byId = new Map<string, ExportGridLayoutItem>();
  let maxBottom = 0;

  layout.forEach((item) => {
    if (!item || typeof item.i !== "string") {
      return;
    }
    const xRaw = Number(item.x);
    const yRaw = Number(item.y);
    const wRaw = Number(item.w);
    const hRaw = Number(item.h);
    if (!Number.isFinite(xRaw) || !Number.isFinite(yRaw) || !Number.isFinite(wRaw) || !Number.isFinite(hRaw)) {
      return;
    }
    const w = Math.max(1, Math.min(cols, Math.round(wRaw)));
    const x = Math.max(0, Math.min(Math.round(xRaw), Math.max(0, cols - w)));
    const h = Math.max(2, Math.round(hRaw));
    const y = Math.max(0, Math.round(yRaw));
    byId.set(item.i, { x, y, w, h });
    maxBottom = Math.max(maxBottom, y + h);
  });

  let appendY = maxBottom;
  artifacts.forEach((artifact) => {
    if (byId.has(artifact.id)) {
      return;
    }
    const seed = defaultExportTileSize(artifact, gridColumns);
    byId.set(artifact.id, {
      x: 0,
      y: appendY,
      w: seed.w,
      h: seed.h
    });
    appendY += seed.h;
  });

  return byId;
}

function buildDashboardHtml(
  sessionId: string,
  datasetName: string,
  artifacts: ArtifactPayload[],
  gridColumns: DashboardColumns,
  dashboardLayout: DashboardLayoutItem[],
  title: string
): string {
  const layoutById = resolveExportLayout(artifacts, dashboardLayout, gridColumns);
  const rendered: string[] = [];
  const plotScripts: string[] = [];

  artifacts.forEach((artifact, index) => {
    const artifactTitle = artifact.text || artifact.type;
    const code = typeof artifact.meta?.code === "string" ? artifact.meta.code : "";
    const codeBlock = code
      ? `<details><summary>Код артефакта</summary><pre>${escapeHtml(code)}</pre></details>`
      : "";
    const currentLayout = layoutById.get(artifact.id) ?? defaultExportTileSize(artifact, gridColumns);
    const placementStyle = `grid-column:${currentLayout.x + 1} / span ${currentLayout.w}; grid-row:${currentLayout.y + 1} / span ${currentLayout.h};`;

    if (artifact.type === "plot" && artifact.data.format === "plotly-json") {
      const payload = artifact.data.data as {
        data?: unknown[];
        layout?: Record<string, unknown>;
        config?: Record<string, unknown>;
      };
      const plotId = `plot_${index}`;
      rendered.push(`
        <article class="artifact-card" style="${placementStyle}">
          <header><h3>${escapeHtml(artifactTitle)}</h3><span>${escapeHtml(artifact.type)}</span></header>
          <div class="plot-wrap" id="${plotId}"></div>
          ${codeBlock}
        </article>
      `);
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
      return;
    }

    if (artifact.type === "table" && artifact.data.format === "split") {
      rendered.push(`
        <article class="artifact-card" style="${placementStyle}">
          <header><h3>${escapeHtml(artifactTitle)}</h3><span>${escapeHtml(artifact.type)}</span></header>
          ${buildTableHtml(artifact)}
          ${codeBlock}
        </article>
      `);
      return;
    }

    if (artifact.type === "value" && artifact.data.format === "value") {
      rendered.push(`
        <article class="artifact-card" style="${placementStyle}">
          <header><h3>${escapeHtml(artifactTitle)}</h3><span>${escapeHtml(artifact.type)}</span></header>
          ${buildValueHtml(artifact)}
          ${codeBlock}
        </article>
      `);
      return;
    }

    rendered.push(`
      <article class="artifact-card" style="${placementStyle}">
        <header><h3>${escapeHtml(artifactTitle)}</h3><span>${escapeHtml(artifact.type)}</span></header>
        <pre>${escapeHtml(JSON.stringify(artifact.data.data, null, 2))}</pre>
        ${codeBlock}
      </article>
    `);
  });

  const generatedAt = new Date().toLocaleString("ru-RU");
  return `<!doctype html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>${escapeHtml(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root { --line: #d8def2; --ink: #182338; --muted: #5e6f90; --accent: #4d72ff; --panel: #fff; --bg: #f6f8ff; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Manrope", "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    .wrap { max-width: 1600px; margin: 0 auto; padding: 24px; display: grid; gap: 16px; }
    .head { border: 1px solid var(--line); border-radius: 16px; background: var(--panel); padding: 16px 20px; }
    .head h1 { margin: 0 0 4px; font-size: 24px; }
    .head p { margin: 0; color: var(--muted); }
    .grid { display: grid; grid-template-columns: repeat(${gridColumns}, minmax(0, 1fr)); grid-auto-rows: 46px; gap: 10px; align-items: stretch; }
    .artifact-card { border: 1px solid var(--line); border-radius: 14px; background: var(--panel); padding: 12px; display: grid; gap: 10px; min-height: 0; height: 100%; grid-template-rows: auto minmax(0, 1fr) auto; }
    .artifact-card header { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
    .artifact-card h3 { margin: 0; font-size: 16px; }
    .artifact-card span { color: var(--muted); font-size: 12px; font-family: monospace; }
    .plot-wrap { width: 100%; min-height: 220px; height: 100%; border: 1px solid var(--line); border-radius: 10px; }
    .table-wrap { overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border: 1px solid var(--line); padding: 6px 8px; text-align: left; }
    th { background: #eef2ff; }
    .value-grid { margin: 0; display: grid; gap: 8px; }
    .value-row { display: grid; grid-template-columns: 1fr auto; gap: 10px; border: 1px solid var(--line); border-radius: 10px; padding: 8px; }
    .value-row dt { color: var(--muted); }
    .value-row dd { margin: 0; font-weight: 700; }
    pre { margin: 0; overflow: auto; min-height: 0; background: #f4f6ff; border: 1px solid var(--line); border-radius: 10px; padding: 10px; font-size: 12px; }
    details summary { cursor: pointer; color: var(--muted); font-size: 13px; }
    @media (max-width: 1320px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } .wrap { padding: 12px; } }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="head">
      <h1>${escapeHtml(title)}</h1>
      <p>session: ${escapeHtml(sessionId)} | dataset: ${escapeHtml(datasetName || "dataset.csv")} | exported: ${escapeHtml(generatedAt)}</p>
    </section>
    <section class="grid">
      ${rendered.join("\n")}
    </section>
  </div>
  <script>
    ${plotScripts.join("\n")}
    const resizePlots = () => {
      if (!window.Plotly || !window.Plotly.Plots) return;
      document.querySelectorAll(".plot-wrap").forEach((el) => {
        try { window.Plotly.Plots.resize(el); } catch {}
      });
    };
    window.addEventListener("resize", resizePlots);
    setTimeout(resizePlots, 80);
  </script>
</body>
</html>`;
}

function buildChatHistoryHtml(
  sessionId: string,
  sessionTitle: string,
  datasetName: string,
  messages: ChatMessage[]
): string {
  const plotScripts: string[] = [];
  const renderedMessages = messages
    .map((message, messageIndex) => {
      const metrics = message.metrics
        ? `<div class="chips"><span>${message.metrics.duration_ms} ms</span><span>${message.metrics.artifact_count} артефактов</span><span>${escapeHtml(message.metrics.model)}</span></div>`
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
    ${renderedMessages || '<p>История пуста.</p>'}
  </main>
  <script>
    ${plotScripts.join("\n")}
  </script>
</body>
</html>`;
}

export function WorkspacePage({ user, onLogout, onNavigate }: WorkspacePageProps): JSX.Element {
  const mainStageRef = useRef<HTMLElement | null>(null);
  const [sessionId, setSessionId] = useState<string>("");
  const [sessionTitle, setSessionTitle] = useState<string>("Новый чат");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [includeReasoning, setIncludeReasoning] = useState<boolean>(() =>
    readBoolSetting(`${SETTING_REASONING_KEY}${user.id}`, false)
  );
  const [useHistory, setUseHistory] = useState<boolean>(() =>
    readBoolSetting(`${SETTING_USE_HISTORY_KEY}${user.id}`, false)
  );
  const [showCode, setShowCode] = useState<boolean>(() =>
    readBoolSetting(`${SETTING_SHOW_CODE_KEY}${user.id}`, false)
  );
  const [sortMode, setSortMode] = useState<SortMode>(() =>
    readSortModeSetting(`${SETTING_SORT_MODE_KEY}${user.id}`)
  );
  const [dashboardColumns, setDashboardColumns] = useState<DashboardColumns>(() =>
    readDashboardColumnsSetting(`${SETTING_DASHBOARD_COLUMNS_KEY}${user.id}`)
  );
  const [dashboardLayoutLocked, setDashboardLayoutLocked] = useState<boolean>(() =>
    readBoolSetting(`${SETTING_DASHBOARD_LOCK_KEY}${user.id}`, false)
  );
  const [themeMode, setThemeMode] = useState<ThemeMode>(() =>
    readThemeSetting(`${SETTING_THEME_KEY}${user.id}`)
  );
  const [chatColumnWidth, setChatColumnWidth] = useState<number>(() =>
    readChatColumnWidthSetting(`${SETTING_CHAT_WIDTH_KEY}${user.id}`)
  );
  const [isChatColumnResizing, setIsChatColumnResizing] = useState(false);
  const [sidebarPinned, setSidebarPinned] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [dashboardArtifactIds, setDashboardArtifactIds] = useState<string[]>([]);
  const [dashboardLayout, setDashboardLayout] = useState<DashboardLayoutItem[]>([]);
  const [isReady, setIsReady] = useState(false);
  const [datasetName, setDatasetName] = useState<string>("");
  const [isUploading, setIsUploading] = useState(false);
  const [hasDataset, setHasDataset] = useState(false);
  const [runtimeModelName, setRuntimeModelName] = useState<string>("");
  const [runtimeSettings, setRuntimeSettings] = useState<Pick<
    UserSettings,
    | "llm_temperature_chat"
    | "llm_temperature_tool"
    | "llm_max_tokens_default"
    | "llm_max_tokens_reasoning"
    | "backend_query_timeout_sec"
    | "agent_max_steps"
    | "agent_step_timeout_sec"
    | "agent_inner_recursion_limit"
  >>({
    llm_temperature_chat: 0.5,
    llm_temperature_tool: 0.15,
    llm_max_tokens_default: 1200,
    llm_max_tokens_reasoning: 2200,
    backend_query_timeout_sec: 180,
    agent_max_steps: 5,
    agent_step_timeout_sec: 45,
    agent_inner_recursion_limit: 6
  });
  const [runtimeSettingsSaving, setRuntimeSettingsSaving] = useState(false);
  const [analysisDepth, setAnalysisDepth] = useState<AnalysisDepth>("light");

  const chatAgent = useChatAgent({
    sessionId,
    includeReasoning,
    useHistory,
    analysisDepth,
  });
  const {
    messages,
    artifacts,
    isStreaming,
    streamDraft,
    streamReasoning,
    streamPhases,
    error,
    lastQuery,
    hydrate,
    sendQuery,
    retryLast,
    stopStreaming,
    reset,
    clearError,
    setErrorMessage
  } = chatAgent;

  const activeSessionSummary = useMemo(
    () => sessions.find((item) => item.session_id === sessionId) ?? null,
    [sessionId, sessions]
  );

  const pinnedArtifactIds = useMemo(() => new Set(dashboardArtifactIds), [dashboardArtifactIds]);

  const pinnedArtifacts = useMemo(() => {
    const byId = new Map(artifacts.map((artifact) => [artifact.id, artifact]));
    return dashboardArtifactIds
      .map((artifactId) => byId.get(artifactId))
      .filter((artifact): artifact is ArtifactPayload => Boolean(artifact));
  }, [artifacts, dashboardArtifactIds]);

  const modelName = useMemo(() => {
    if (runtimeModelName.trim()) {
      return runtimeModelName.trim();
    }
    for (let idx = messages.length - 1; idx >= 0; idx -= 1) {
      const message = messages[idx];
      if (message.role === "assistant" && message.metrics?.model) {
        return message.metrics.model;
      }
    }
    return "текущая модель";
  }, [messages, runtimeModelName]);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const profile = await getRuntimeModelProfile();
        if (!active) {
          return;
        }
        setRuntimeModelName(String(profile.model || "").trim());
      } catch {
        if (active) {
          setRuntimeModelName("");
        }
      }
    })();
    return () => {
      active = false;
    };
  }, [user.id]);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const settings = await getUserSettings();
        if (!active) {
          return;
        }
        setIncludeReasoning(Boolean(settings.default_include_reasoning));
        setThemeMode(settings.theme === "light" ? "light" : "dark");
        if (settings.analysis_depth) {
          setAnalysisDepth(settings.analysis_depth);
        }
        setRuntimeSettings({
          llm_temperature_chat: Number(settings.llm_temperature_chat),
          llm_temperature_tool: Number(settings.llm_temperature_tool),
          llm_max_tokens_default: Number(settings.llm_max_tokens_default),
          llm_max_tokens_reasoning: Number(settings.llm_max_tokens_reasoning),
          backend_query_timeout_sec: Number(settings.backend_query_timeout_sec),
          agent_max_steps: Number(settings.agent_max_steps),
          agent_step_timeout_sec: Number(settings.agent_step_timeout_sec),
          agent_inner_recursion_limit: Number(settings.agent_inner_recursion_limit)
        });
      } catch {
        // Keep local defaults if profile fetch fails.
      }
    })();
    return () => {
      active = false;
    };
  }, [user.id]);

  const clampChatColumnWidth = useCallback((requestedWidth: number): number => {
    const stage = mainStageRef.current;
    if (!stage) {
      return Math.max(CHAT_COLUMN_MIN_WIDTH, Math.round(requestedWidth));
    }
    const stageWidth = stage.getBoundingClientRect().width;
    const maxWidth = Math.max(CHAT_COLUMN_MIN_WIDTH + 40, stageWidth - DASHBOARD_MIN_WIDTH);
    return Math.min(maxWidth, Math.max(CHAT_COLUMN_MIN_WIDTH, Math.round(requestedWidth)));
  }, []);

  const refreshSessionsList = useCallback(async (): Promise<SessionSummary[]> => {
    const rows = await listSessions();
    setSessions(rows);
    return rows;
  }, []);

  const loadSessionState = useCallback(
    async (nextSessionId: string): Promise<void> => {
      const sessionState = await getSession(nextSessionId);
      hydrate(sessionState);
      setSessionId(nextSessionId);
      setSessionTitle(sessionState.title || "Новый чат");
      setHasDataset(sessionState.has_dataset);
      window.localStorage.setItem(sessionPointerKey(user.id), nextSessionId);

      if (sessionState.has_dataset) {
        const storedDatasetName = window.localStorage.getItem(datasetNameKey(user.id, nextSessionId));
        setDatasetName(storedDatasetName || "dataset.csv (из сессии)");
      } else {
        setDatasetName("");
      }
    },
    [hydrate, user.id]
  );

  const cleanupDeletedSessionStorage = useCallback(
    (deletedSessionId: string): void => {
      if (!deletedSessionId) {
        return;
      }
      window.localStorage.removeItem(datasetNameKey(user.id, deletedSessionId));
      window.localStorage.removeItem(dashboardOrderKey(user.id, deletedSessionId));
      window.localStorage.removeItem(dashboardLayoutKey(user.id, deletedSessionId));
      window.localStorage.removeItem(`${LEGACY_PINNED_KEY_PREFIX}${deletedSessionId}`);
      const currentPointer = window.localStorage.getItem(sessionPointerKey(user.id));
      if (currentPointer === deletedSessionId) {
        window.localStorage.removeItem(sessionPointerKey(user.id));
      }
    },
    [user.id]
  );

  useEffect(() => {
    let mounted = true;

    async function init(): Promise<void> {
      setIsReady(false);
      reset();
      try {
        let rows = await refreshSessionsList();
        if (!mounted) {
          return;
        }

        let resolvedSessionId = window.localStorage.getItem(sessionPointerKey(user.id)) ?? "";
        if (!resolvedSessionId || !rows.some((item) => item.session_id === resolvedSessionId)) {
          resolvedSessionId = rows[0]?.session_id ?? "";
        }

        if (!resolvedSessionId) {
          resolvedSessionId = await createSession(false);
          rows = await refreshSessionsList();
          if (!mounted) {
            return;
          }
        }

        await loadSessionState(resolvedSessionId);
        if (!mounted) {
          return;
        }

        const resolvedMeta = rows.find((item) => item.session_id === resolvedSessionId);
        if (resolvedMeta) {
          setSessionTitle(resolvedMeta.title || "Новый чат");
        }
      } catch (err) {
        if (mounted) {
          setErrorMessage(String(err));
        }
      } finally {
        if (mounted) {
          setIsReady(true);
        }
      }
    }

    void init();
    return () => {
      mounted = false;
    };
  }, [loadSessionState, refreshSessionsList, reset, setErrorMessage, user.id]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    const rawOrder = window.localStorage.getItem(dashboardOrderKey(user.id, sessionId));
    const rawLegacy = window.localStorage.getItem(`${LEGACY_PINNED_KEY_PREFIX}${sessionId}`);
    const source = rawOrder || rawLegacy;
    if (!source) {
      setDashboardArtifactIds([]);
      return;
    }
    try {
      const parsed = JSON.parse(source) as string[];
      setDashboardArtifactIds(
        Array.isArray(parsed) ? parsed.filter((item) => typeof item === "string") : []
      );
    } catch {
      setDashboardArtifactIds([]);
    }
  }, [sessionId, user.id]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    const rawLayout = window.localStorage.getItem(dashboardLayoutKey(user.id, sessionId));
    if (!rawLayout) {
      setDashboardLayout([]);
      return;
    }
    try {
      const parsed = JSON.parse(rawLayout) as DashboardLayoutItem[];
      if (!Array.isArray(parsed)) {
        setDashboardLayout([]);
        return;
      }
      const normalized = parsed
        .filter((item) => item && typeof item === "object" && typeof item.i === "string")
        .map((item) => ({
          i: String(item.i),
          x: Number.isFinite(item.x) ? Number(item.x) : 0,
          y: Number.isFinite(item.y) ? Number(item.y) : 0,
          w: Number.isFinite(item.w) ? Number(item.w) : 1,
          h: Number.isFinite(item.h) ? Number(item.h) : 4,
          minW: Number.isFinite(item.minW) ? Number(item.minW) : undefined,
          minH: Number.isFinite(item.minH) ? Number(item.minH) : undefined,
          maxW: Number.isFinite(item.maxW) ? Number(item.maxW) : undefined
        }));
      setDashboardLayout(normalized);
    } catch {
      setDashboardLayout([]);
    }
  }, [sessionId, user.id]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    window.localStorage.setItem(
      dashboardOrderKey(user.id, sessionId),
      JSON.stringify(dashboardArtifactIds)
    );
    window.localStorage.setItem(`${LEGACY_PINNED_KEY_PREFIX}${sessionId}`, JSON.stringify(dashboardArtifactIds));
  }, [dashboardArtifactIds, sessionId, user.id]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    window.localStorage.setItem(
      dashboardLayoutKey(user.id, sessionId),
      JSON.stringify(dashboardLayout)
    );
  }, [dashboardLayout, sessionId, user.id]);

  useEffect(() => {
    window.localStorage.setItem(`${SETTING_USE_HISTORY_KEY}${user.id}`, useHistory ? "1" : "0");
  }, [useHistory, user.id]);

  useEffect(() => {
    window.localStorage.setItem(`${SETTING_REASONING_KEY}${user.id}`, includeReasoning ? "1" : "0");
  }, [includeReasoning, user.id]);

  useEffect(() => {
    window.localStorage.setItem(`${SETTING_SHOW_CODE_KEY}${user.id}`, showCode ? "1" : "0");
  }, [showCode, user.id]);

  useEffect(() => {
    window.localStorage.setItem(`${SETTING_SORT_MODE_KEY}${user.id}`, sortMode);
  }, [sortMode, user.id]);

  useEffect(() => {
    window.localStorage.setItem(`${SETTING_DASHBOARD_COLUMNS_KEY}${user.id}`, String(dashboardColumns));
  }, [dashboardColumns, user.id]);

  useEffect(() => {
    window.localStorage.setItem(`${SETTING_DASHBOARD_LOCK_KEY}${user.id}`, dashboardLayoutLocked ? "1" : "0");
  }, [dashboardLayoutLocked, user.id]);

  useEffect(() => {
    window.localStorage.setItem(`${SETTING_THEME_KEY}${user.id}`, themeMode);
    document.documentElement.setAttribute("data-theme", themeMode);
  }, [themeMode, user.id]);

  useEffect(() => {
    window.localStorage.setItem(`${SETTING_CHAT_WIDTH_KEY}${user.id}`, String(chatColumnWidth));
  }, [chatColumnWidth, user.id]);

  useEffect(() => {
    function syncChatWidthToViewport(): void {
      setChatColumnWidth((prev) => clampChatColumnWidth(prev));
    }

    syncChatWidthToViewport();
    window.addEventListener("resize", syncChatWidthToViewport);
    return () => {
      window.removeEventListener("resize", syncChatWidthToViewport);
    };
  }, [clampChatColumnWidth]);

  useEffect(() => {
    if (!isChatColumnResizing) {
      return;
    }

    function handleMouseMove(event: MouseEvent): void {
      const stage = mainStageRef.current;
      if (!stage) {
        return;
      }
      const rect = stage.getBoundingClientRect();
      const nextWidth = rect.right - event.clientX - CHAT_SPLITTER_WIDTH / 2;
      setChatColumnWidth(clampChatColumnWidth(nextWidth));
    }

    function handleMouseUp(): void {
      setIsChatColumnResizing(false);
    }

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    const previousCursor = document.body.style.cursor;
    const previousUserSelect = document.body.style.userSelect;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = previousCursor;
      document.body.style.userSelect = previousUserSelect;
    };
  }, [clampChatColumnWidth, isChatColumnResizing]);

  useEffect(() => {
    const existing = new Set(artifacts.map((item) => item.id));
    setDashboardArtifactIds((prev) => {
      let changed = false;
      const next: string[] = [];
      prev.forEach((id) => {
        if (existing.has(id)) {
          next.push(id);
        } else {
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }, [artifacts]);

  useEffect(() => {
    const pinnedSet = new Set(dashboardArtifactIds);
    setDashboardLayout((prev) => prev.filter((item) => pinnedSet.has(item.i)));
  }, [dashboardArtifactIds]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    void refreshSessionsList();
  }, [hasDataset, refreshSessionsList, sessionId]);

  useEffect(() => {
    if (activeSessionSummary) {
      setSessionTitle(activeSessionSummary.title || "Новый чат");
    }
  }, [activeSessionSummary]);

  async function handleUpload(file: File): Promise<void> {
    if (!sessionId) {
      return;
    }
    setIsUploading(true);
    clearError();
    try {
      await uploadCsv(sessionId, file);
      setDatasetName(file.name);
      setHasDataset(true);
      window.localStorage.setItem(datasetNameKey(user.id, sessionId), file.name);
      await refreshSessionsList();
    } catch (err) {
      setErrorMessage(String(err));
    } finally {
      setIsUploading(false);
    }
  }

  async function handleAnalysisDepthChange(next: AnalysisDepth): Promise<void> {
    setAnalysisDepth(next);
    try {
      await updateUserSettings({ analysis_depth: next });
    } catch {
      // Keep local value even if server save fails.
    }
  }

  async function handleSaveRuntimeSettings(
    next: typeof runtimeSettings
  ): Promise<void> {
    setRuntimeSettings(next);
    setRuntimeSettingsSaving(true);
    try {
      const updated = await updateUserSettings(next);
      setRuntimeSettings({
        llm_temperature_chat: Number(updated.llm_temperature_chat),
        llm_temperature_tool: Number(updated.llm_temperature_tool),
        llm_max_tokens_default: Number(updated.llm_max_tokens_default),
        llm_max_tokens_reasoning: Number(updated.llm_max_tokens_reasoning),
        backend_query_timeout_sec: Number(updated.backend_query_timeout_sec),
        agent_max_steps: Number(updated.agent_max_steps),
        agent_step_timeout_sec: Number(updated.agent_step_timeout_sec),
        agent_inner_recursion_limit: Number(updated.agent_inner_recursion_limit)
      });
      setErrorMessage(null);
    } catch (err) {
      setErrorMessage(`Не удалось сохранить runtime настройки: ${String(err)}`);
    } finally {
      setRuntimeSettingsSaving(false);
    }
  }

  const handlePinArtifact = useCallback(
    (artifactId: string): void => {
      setDashboardArtifactIds((prev) => {
        if (prev.includes(artifactId)) {
          return prev;
        }
        if (sortMode === "newest") {
          return [artifactId, ...prev];
        }
        return [...prev, artifactId];
      });
    },
    [sortMode]
  );

  const handleUnpinArtifact = useCallback((artifactId: string): void => {
    setDashboardArtifactIds((prev) => prev.filter((id) => id !== artifactId));
  }, []);

  const handleDownloadChat = useCallback((): void => {
    if (!sessionId) {
      return;
    }
    const html = buildChatHistoryHtml(sessionId, sessionTitle, datasetName, messages);
    saveFile(`chat_${sessionId.slice(0, 8)}.html`, html, "text/html;charset=utf-8");
  }, [datasetName, messages, sessionId, sessionTitle]);

  const handleDownloadDashboard = useCallback((): void => {
    if (!sessionId) {
      return;
    }
    const html = buildDashboardHtml(
      sessionId,
      datasetName,
      pinnedArtifacts,
      dashboardColumns,
      dashboardLayout,
      `Dashboard: ${sessionTitle}`
    );
    saveFile(`dashboard_${sessionId.slice(0, 8)}.html`, html, "text/html;charset=utf-8");
  }, [dashboardColumns, dashboardLayout, datasetName, pinnedArtifacts, sessionId, sessionTitle]);

  const handleSidebarHotzoneEnter = useCallback((): void => {
    setSidebarOpen(true);
  }, []);

  const handleSidebarMouseLeave = useCallback((): void => {
    if (!sidebarPinned) {
      setSidebarOpen(false);
    }
  }, [sidebarPinned]);

  const handleSidebarMouseEnter = useCallback((): void => {
    setSidebarOpen(true);
  }, []);

  const handleSidebarPinToggle = useCallback((): void => {
    setSidebarPinned((prev) => {
      const next = !prev;
      if (!next) {
        setSidebarOpen(false);
      } else {
        setSidebarOpen(true);
      }
      return next;
    });
  }, []);

  const handleResetSession = useCallback(async (): Promise<void> => {
    stopStreaming();
    setIsReady(false);
    try {
      if (sessionId) {
        const updated = await generateSessionTitle(sessionId);
        setSessionTitle(updated.title || "Новый чат");
        setSessions((prev) =>
          prev.map((item) => (item.session_id === updated.session_id ? updated : item))
        );
      }

      const nextSessionId = await createSession(false);
      reset();
      setDashboardArtifactIds([]);
      setDashboardLayout([]);
      await loadSessionState(nextSessionId);
      await refreshSessionsList();
    } catch (err) {
      setErrorMessage(String(err));
    } finally {
      setIsReady(true);
    }
  }, [
    loadSessionState,
    refreshSessionsList,
    reset,
    sessionId,
    setErrorMessage,
    stopStreaming
  ]);

  const handleSelectSession = useCallback(
    async (nextSessionId: string): Promise<void> => {
      if (!nextSessionId || nextSessionId === sessionId) {
        return;
      }
      stopStreaming();
      setIsReady(false);
      try {
        await loadSessionState(nextSessionId);
      } catch (err) {
        setErrorMessage(String(err));
      } finally {
        setIsReady(true);
      }
    },
    [loadSessionState, sessionId, setErrorMessage, stopStreaming]
  );

  const handleDeleteSession = useCallback(
    async (targetSessionId: string): Promise<void> => {
      if (!targetSessionId) {
        return;
      }
      const targetMeta = sessions.find((item) => item.session_id === targetSessionId);
      const targetName = targetMeta?.title || "этот чат";
      const approved = window.confirm(`Удалить "${targetName}" без возможности восстановления?`);
      if (!approved) {
        return;
      }

      stopStreaming();
      setIsReady(false);
      try {
        await deleteSession(targetSessionId);
        cleanupDeletedSessionStorage(targetSessionId);
        const nextRows = await refreshSessionsList();

        if (targetSessionId !== sessionId) {
          return;
        }

        let fallbackSessionId = nextRows[0]?.session_id ?? "";
        if (!fallbackSessionId) {
          fallbackSessionId = await createSession(false);
          await refreshSessionsList();
        }

        reset();
        setDashboardArtifactIds([]);
        setDashboardLayout([]);
        await loadSessionState(fallbackSessionId);
      } catch (err) {
        setErrorMessage(String(err));
      } finally {
        setIsReady(true);
      }
    },
    [
      cleanupDeletedSessionStorage,
      loadSessionState,
      refreshSessionsList,
      reset,
      sessionId,
      sessions,
      setErrorMessage,
      stopStreaming
    ]
  );

  const handleLogout = useCallback(async (): Promise<void> => {
    stopStreaming();
    await onLogout();
    onNavigate("/user");
  }, [onLogout, onNavigate, stopStreaming]);

  const handleRenameSession = useCallback(
    async (title: string): Promise<void> => {
      if (!sessionId) {
        return;
      }
      try {
        const updated = await updateSessionTitle(sessionId, title);
        setSessionTitle(updated.title);
        setSessions((prev) =>
          prev.map((item) => (item.session_id === updated.session_id ? updated : item))
        );
      } catch (err) {
        setErrorMessage(String(err));
      }
    },
    [sessionId, setErrorMessage]
  );

  const handleChatResizerMouseDown = useCallback((event: ReactMouseEvent<HTMLDivElement>): void => {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    setIsChatColumnResizing(true);
  }, []);

  const handleChatResizerKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>): void => {
      const step = event.shiftKey ? 40 : 20;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        setChatColumnWidth((prev) => clampChatColumnWidth(prev + step));
        return;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        setChatColumnWidth((prev) => clampChatColumnWidth(prev - step));
      }
    },
    [clampChatColumnWidth]
  );

  const handleChatResizerDoubleClick = useCallback((): void => {
    setChatColumnWidth(clampChatColumnWidth(CHAT_COLUMN_DEFAULT_WIDTH));
  }, [clampChatColumnWidth]);

  const mainStageStyle = useMemo(
    () =>
      ({
        "--chat-col-width": `${chatColumnWidth}px`
      }) as CSSProperties,
    [chatColumnWidth]
  );

  return (
    <div className="app-shell app-shell-refresh">
      <div className="sidebar-hotzone" onMouseEnter={handleSidebarHotzoneEnter} />
      <SiteNav
        currentUser={user}
        onNavigate={onNavigate}
        className="workspace-top-nav"
      />

      <SettingsSidebar
        open={sidebarOpen}
        pinned={sidebarPinned}
        onTogglePinned={handleSidebarPinToggle}
        onMouseEnter={handleSidebarMouseEnter}
        onMouseLeave={handleSidebarMouseLeave}
        onRequestClose={() => {
          if (!sidebarPinned) {
            setSidebarOpen(false);
          }
        }}
        currentUser={user}
        onLogout={handleLogout}
        sessions={sessions}
        activeSessionId={sessionId}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        sessionTitle={sessionTitle}
        onRenameSession={handleRenameSession}
        includeReasoning={includeReasoning}
        onToggleReasoning={setIncludeReasoning}
        useHistory={useHistory}
        onToggleHistory={setUseHistory}
        datasetName={datasetName}
        hasDataset={hasDataset}
        isUploading={isUploading}
        onUpload={handleUpload}
        showCode={showCode}
        onToggleShowCode={setShowCode}
        sortMode={sortMode}
        onSortModeChange={setSortMode}
        dashboardColumns={dashboardColumns}
        onDashboardColumnsChange={setDashboardColumns}
        dashboardLayoutLocked={dashboardLayoutLocked}
        onToggleDashboardLayoutLocked={setDashboardLayoutLocked}
        onResetDashboardLayout={() => setDashboardLayout([])}
        themeMode={themeMode}
        onThemeModeChange={setThemeMode}
        analysisDepth={analysisDepth}
        onAnalysisDepthChange={handleAnalysisDepthChange}
        modelName={modelName}
        runtimeSettings={runtimeSettings}
        onRuntimeSettingsChange={setRuntimeSettings}
        onSaveRuntimeSettings={handleSaveRuntimeSettings}
        runtimeSettingsSaving={runtimeSettingsSaving}
        pinnedCount={pinnedArtifacts.length}
        onDownloadChat={handleDownloadChat}
        onDownloadDashboard={handleDownloadDashboard}
        onResetSession={handleResetSession}
      />

      <div className="workspace workspace-refresh">
        <main
          ref={mainStageRef}
          className={`main-stage main-stage-resizable${isChatColumnResizing ? " is-resizing" : ""}`}
          style={mainStageStyle}
        >
          <section className="dashboard-column">
            <ArtifactsPanel
              artifacts={pinnedArtifacts}
              showCode={showCode}
              gridColumns={dashboardColumns}
              themeMode={themeMode}
              layoutLocked={dashboardLayoutLocked}
              layout={dashboardLayout}
              onLayoutChange={setDashboardLayout}
              onResetLayout={() => setDashboardLayout([])}
              onUnpinArtifact={handleUnpinArtifact}
            />
          </section>

          <div
            className={`column-resizer${isChatColumnResizing ? " is-active" : ""}`}
            role="separator"
            aria-orientation="vertical"
            aria-label="Изменить ширину колонки чата"
            tabIndex={0}
            onMouseDown={handleChatResizerMouseDown}
            onKeyDown={handleChatResizerKeyDown}
            onDoubleClick={handleChatResizerDoubleClick}
          />

          <section className="chat-column">
            <ChatPanel
              sessionId={sessionId}
              includeReasoning={includeReasoning}
              showCode={showCode}
              themeMode={themeMode}
              messages={messages}
              pinnedArtifactIds={pinnedArtifactIds}
              onPinArtifact={(artifact) => handlePinArtifact(artifact.id)}
              streamDraft={streamDraft}
              streamReasoning={streamReasoning}
              streamPhases={streamPhases}
              isStreaming={isStreaming}
              isReady={isReady}
              error={error}
              onClearError={clearError}
              onSubmit={sendQuery}
              onStop={stopStreaming}
              onRetry={retryLast}
              canRetry={Boolean(lastQuery)}
            />
          </section>
        </main>
      </div>
    </div>
  );
}
