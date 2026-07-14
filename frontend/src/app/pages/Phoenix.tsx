import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { useTheme } from "next-themes";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  ExternalLink,
  Layers,
  RefreshCw,
  Search,
  Server,
  Zap,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Navigation } from "../components/Navigation";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../components/ui/dialog";
import { getPhoenixOverview, getPhoenixTraceDetail, getPhoenixTraces, getPhoenixTracesBySession } from "../lib/backend-api";
import type { PhoenixOverview, PhoenixSpanSnapshotItem, PhoenixTokenUsageRow, PhoenixTraceRow } from "../lib/backend-types";
import {
  formatPhoenixTraceHistorySummary,
  getPhoenixTraceEmptyMessage,
  type PhoenixTraceHistoryStatus,
} from "../lib/phoenix-trace-history";
import { buildPhoenixProjectTraceUrl, resolvePhoenixUiBaseUrl } from "../lib/phoenix-url";

function formatNumber(value: number): string {
  return new Intl.NumberFormat("ru-RU").format(value);
}

function formatTokens(value: number | null | undefined): string {
  if (value == null) {
    return "—";
  }
  return formatNumber(value);
}

function formatDuration(durationMs: number): string {
  if (durationMs >= 1000) {
    return `${(durationMs / 1000).toFixed(durationMs >= 10000 ? 0 : 1)}с`;
  }
  return `${durationMs}мс`;
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function traceMatches(trace: PhoenixTraceRow, search: string): boolean {
  const query = search.trim().toLowerCase();
  if (!query) {
    return true;
  }
  return [
    trace.query_preview,
    trace.model ?? "",
    trace.user ?? "",
    trace.session_id ?? "",
    trace.trace_id,
  ].some((value) => value.toLowerCase().includes(query));
}

function formatRequestKind(kind: string | null | undefined): string {
  const normalized = String(kind || "").trim().toLowerCase();
  if (!normalized) {
    return "тип не указан";
  }
  if (normalized === "chat") {
    return "чат";
  }
  if (normalized === "analysis") {
    return "анализ";
  }
  if (normalized === "query") {
    return "запрос";
  }
  if (normalized === "stream") {
    return "поток";
  }
  return kind || "тип не указан";
}

function tokenRowHasTelemetry(row: PhoenixTokenUsageRow): boolean {
  return row.input_tokens != null || row.output_tokens != null || row.total_tokens != null;
}

function buildTokenUsageByDay(rows: PhoenixTokenUsageRow[]) {
  const buckets = new Map<string, { label: string; input: number; output: number }>();

  for (const row of rows) {
    if (!tokenRowHasTelemetry(row)) {
      continue;
    }
    const date = new Date(row.started_at);
    if (Number.isNaN(date.getTime())) {
      continue;
    }
    const key = date.toISOString().slice(0, 10);
    const label = new Intl.DateTimeFormat("ru-RU", {
      day: "2-digit",
      month: "short",
    }).format(date);
    const current = buckets.get(key) ?? { label, input: 0, output: 0 };
    current.input += row.input_tokens ?? 0;
    current.output += row.output_tokens ?? 0;
    buckets.set(key, current);
  }

  const result: { label: string; input: number; output: number }[] = [];
  const now = new Date();
  for (let offset = 6; offset >= 0; offset--) {
    const day = new Date(now);
    day.setDate(day.getDate() - offset);
    const key = day.toISOString().slice(0, 10);
    const label = new Intl.DateTimeFormat("ru-RU", {
      day: "2-digit",
      month: "short",
    }).format(day);
    result.push(buckets.get(key) ?? { label, input: 0, output: 0 });
  }
  return result;
}

function spanKindIcon(kind: string): string {
  switch (kind) {
    case "LLM": return "🧠";
    case "TOOL": return "🔧";
    case "CHAIN": return "⛓";
    case "RETRIEVER": return "📄";
    case "EMBEDDING": return "🔢";
    default: return "◻";
  }
}

function spanKindColor(kind: string): string {
  switch (kind) {
    case "LLM": return "text-sky-400";
    case "TOOL": return "text-amber-400";
    case "CHAIN": return "text-violet-400";
    case "RETRIEVER": return "text-emerald-400";
    default: return "text-muted-foreground";
  }
}

function buildSpanTree(spans: PhoenixSpanSnapshotItem[]): PhoenixSpanSnapshotItem[][] {
  const spanIds = new Set(spans.map((s) => s.span_id));
  const childrenOf = new Map<string | null, PhoenixSpanSnapshotItem[]>();
  for (const span of spans) {
    const parent = span.parent_id && spanIds.has(span.parent_id) ? span.parent_id : null;
    if (!childrenOf.has(parent)) childrenOf.set(parent, []);
    childrenOf.get(parent)!.push(span);
  }
  const levels: PhoenixSpanSnapshotItem[][] = [];
  const rootIds = spans
    .filter((s) => !s.parent_id || !spanIds.has(s.parent_id))
    .map((s) => s.span_id);
  const seen = new Set<string>();
  const queue = [...rootIds];
  while (queue.length > 0) {
    const level = queue
      .map((id) => spans.find((s) => s.span_id === id))
      .filter(Boolean) as PhoenixSpanSnapshotItem[];
    if (level.length === 0) break;
    levels.push(level);
    const next: string[] = [];
    for (const span of level) {
      const children = childrenOf.get(span.span_id) ?? [];
      for (const child of children) {
        if (!seen.has(child.span_id)) {
          seen.add(child.span_id);
          next.push(child.span_id);
        }
      }
    }
    queue.length = 0;
    queue.push(...next);
  }
  return levels;
}

function SpanTreeRow({ span, depth }: { span: PhoenixSpanSnapshotItem; depth: number }) {
  const [expanded, setExpanded] = useState(depth < 2);
  const hasIo = Boolean(span.input_value || span.output_value);
  return (
    <div>
      <div
        className="group cursor-pointer border-b border-border/10 py-2 text-sm transition-colors hover:bg-white/5"
        style={{ paddingLeft: `${depth * 24 + 16}px` }}
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          {depth > 0 && (
            expanded
              ? <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
              : <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
          )}
          <span className={`text-xs ${spanKindColor(span.span_kind)}`}>{spanKindIcon(span.span_kind)}</span>
          <span className="font-medium">{span.name}</span>
          <span className="text-xs text-muted-foreground">({span.span_kind})</span>
          <span className={`ml-auto text-xs ${span.status_code === "OK" ? "text-emerald-400" : "text-rose-400"}`}>
            {span.status_code === "OK" ? "OK" : span.status_code}
          </span>
          <span className="ml-4 w-16 text-right text-xs text-muted-foreground">
            {formatDuration(span.duration_ms)}
          </span>
          {span.total_tokens != null && (
            <span className="ml-2 text-xs text-muted-foreground">
              {formatTokens(span.total_tokens)} ток.
            </span>
          )}
          {hasIo && (
            <span className="ml-2 text-[10px] text-muted-foreground/50">
              IO
            </span>
          )}
        </div>
      </div>
      {expanded && hasIo && (
        <div className="border-b border-border/10 bg-black/10" style={{ paddingLeft: `${depth * 24 + 16}px` }}>
          {span.input_value && (
            <details className="group/io">
              <summary className="cursor-pointer px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest text-muted-foreground hover:text-foreground">
                Вход
              </summary>
              <pre className="overflow-x-auto px-4 pb-2 text-[11px] text-muted-foreground">
                {span.input_value.length > 500 ? span.input_value.slice(0, 500) + "\n..." : span.input_value}
              </pre>
            </details>
          )}
          {span.output_value && (
            <details className="group/io">
              <summary className="cursor-pointer px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest text-muted-foreground hover:text-foreground">
                Выход
              </summary>
              <pre className="overflow-x-auto px-4 pb-2 text-[11px] text-muted-foreground">
                {span.output_value.length > 500 ? span.output_value.slice(0, 500) + "\n..." : span.output_value}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Таймлайн трассы ──────────────────────────────────────

interface TimelineStep {
  type: string;
  icon: string;
  label: string;
  spans: PhoenixSpanSnapshotItem[];
  startMs: number;
  durationMs: number;
  status: "ok" | "error" | "mixed";
  skillName?: string | null;
}

const TOOL_ICONS: Record<string, string> = {
  sql_tool: "🛢",
  plotly_tool: "📊",
  pandas_tool: "🔄",
  value_tool: "📈",
  database_tool: "🗄",
  planner_tool: "📋",
  get_tool_instructions: "📖",
};

function toolIcon(name: string): string {
  return TOOL_ICONS[name] ?? "🔧";
}

function fmtDur(ms: number): string {
  if (ms < 1000) return `${ms}мс`;
  return `${(ms / 1000).toFixed(1)}с`;
}

function pickSkillName(spans: PhoenixSpanSnapshotItem[]): string | null {
  const idx = spans.findIndex((s) => s.name === "planner_tool");
  const next = spans
    .slice(idx + 1)
    .find((s) => s.name === "get_tool_instructions");
  if (!next) return null;
  try {
    const input = JSON.parse(next.input_value ?? "{}");
    return input.skill_id ?? null;
  } catch {
    return null;
  }
}

function buildTimelineSteps(
  spans: PhoenixSpanSnapshotItem[],
  meta: PhoenixTraceRow | null,
): { steps: TimelineStep[]; totalMs: number } {
  const sorted = [...spans].sort(
    (a, b) =>
      new Date(a.start_time).getTime() - new Date(b.start_time).getTime(),
  );
  if (sorted.length === 0) return { steps: [], totalMs: 0 };

  const refTime = new Date(sorted[0].start_time).getTime();
  const totalMs = Math.max(
    0,
    new Date(sorted[sorted.length - 1].end_time).getTime() - refTime,
  );
  const skipNames = new Set([
    "dispatch",
    "agent",
    "finalize",
    "RunnableCallable",
    "LangGraph",
  ]);

  const steps: TimelineStep[] = [];

  // 1. Вопрос
  if (meta) {
    steps.push({
      type: "question",
      icon: "💬",
      label: "Вопрос пользователя",
      spans: [],
      startMs: 0,
      durationMs: 0,
      status: "ok",
    });
  }

  // 2. Планирование
  const plannerSpans = sorted.filter((s) => s.name === "planner_tool");
  if (plannerSpans.length > 0) {
    const start =
      new Date(plannerSpans[0].start_time).getTime() - refTime;
    const end =
      new Date(plannerSpans[plannerSpans.length - 1].end_time).getTime() -
      refTime;
    const selSkill = pickSkillName(sorted);
    steps.push({
      type: "planning",
      icon: "📋",
      label: `Планирование${plannerSpans.length > 1 ? ` (${plannerSpans.length} попытки)` : ""}`,
      spans: plannerSpans,
      startMs: Math.max(0, start),
      durationMs: Math.max(0, end - start),
      status: plannerSpans.some((s) => s.status_code !== "OK")
        ? "error"
        : "ok",
      skillName: selSkill,
    });
  }

  // 3. get_tool_instructions
  for (const s of sorted.filter((s) => s.name === "get_tool_instructions")) {
    const start = new Date(s.start_time).getTime() - refTime;
    steps.push({
      type: "instructions",
      icon: "📖",
      label: "Инструкции",
      spans: [s],
      startMs: Math.max(0, start),
      durationMs: s.duration_ms,
      status: s.status_code === "OK" ? "ok" : "error",
    });
  }

  // 4. Вызовы инструментов
  for (const s of sorted) {
    if (s.span_kind !== "TOOL") continue;
    if (skipNames.has(s.name)) continue;
    if (s.name === "planner_tool" || s.name === "get_tool_instructions")
      continue;
    const start = new Date(s.start_time).getTime() - refTime;
    steps.push({
      type: "tool",
      icon: toolIcon(s.name),
      label: s.name,
      spans: [s],
      startMs: Math.max(0, start),
      durationMs: s.duration_ms,
      status: s.status_code === "OK" ? "ok" : "error",
    });
  }

  // 5. Генерация LLM
  const llmSpans = sorted.filter((s) => s.span_kind === "LLM");
  if (llmSpans.length > 0) {
    const start = new Date(llmSpans[0].start_time).getTime() - refTime;
    const end =
      new Date(llmSpans[llmSpans.length - 1].end_time).getTime() - refTime;
    steps.push({
      type: "llm",
      icon: "🧠",
      label: `Генерация ответа (${llmSpans.length} вызовов)`,
      spans: llmSpans,
      startMs: Math.max(0, start),
      durationMs: Math.max(0, end - start),
      status: llmSpans.some((s) => s.status_code !== "OK")
        ? "mixed"
        : "ok",
    });
  }

  return { steps, totalMs };
}

function TraceTimeline({
  spans,
  meta,
}: {
  spans: PhoenixSpanSnapshotItem[];
  meta: PhoenixTraceRow | null;
}) {
  const { steps, totalMs } = buildTimelineSteps(spans, meta);
  const [expandedStep, setExpandedStep] = useState<string | null>(null);
  const nonEmpty = steps.filter((s) => s.durationMs > 0 || s.type === "question");

  const segmentColors: Record<string, string> = {
    question: "bg-blue-400",
    planning: "bg-amber-400",
    instructions: "bg-sky-400",
    tool: "bg-violet-400",
    llm: "bg-emerald-400",
  };

  return (
    <div className="space-y-5">
      {/* Progress bar */}
      {totalMs > 0 && (
        <div className="space-y-1">
          <div className="relative h-2 w-full overflow-hidden rounded-full bg-border/20">
            {nonEmpty
              .filter((s) => s.durationMs > 0)
              .map((s, i) => (
                <div
                  key={i}
                  className={`absolute h-full rounded-full opacity-80 ${segmentColors[s.type] ?? "bg-blue-400"}`}
                  style={{
                    left: `${(s.startMs / totalMs) * 100}%`,
                    width: `${(s.durationMs / totalMs) * 100}%`,
                  }}
                />
              ))}
          </div>
          <div className="flex justify-between text-[10px] text-muted-foreground">
            <span>0с</span>
            <span>{fmtDur(totalMs)}</span>
          </div>
        </div>
      )}

      {/* Timeline steps */}
      <div className="space-y-0">
        {steps.map((step, i) => {
          const isExpanded = expandedStep === `${i}`;
          const toggle = () =>
            setExpandedStep(isExpanded ? null : `${i}`);
          const statusSym =
            step.status === "ok"
              ? "✓"
              : step.status === "error"
                ? "✗"
                : "~";
          const statusColor =
            step.status === "ok"
              ? "text-emerald-400"
              : step.status === "error"
                ? "text-rose-400"
                : "text-amber-400";
          const hasDetails = step.spans.length > 0;

          return (
            <div
              key={i}
              className={`group relative border-l-2 pb-6 pl-6 last:pb-0 ${step.type === "question" ? "border-transparent" : "border-border/30"}`}
            >
              {/* Dot */}
              {step.type !== "question" && (
                <div className="absolute -left-[9px] top-0 h-4 w-4 rounded-full border-2 border-border/30 bg-background" />
              )}

              <div className="flex flex-col gap-1">
                {/* Header row */}
                <div className="flex items-start gap-2">
                  <span className="mt-0.5 text-base leading-5">
                    {step.icon}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                      <span className="text-sm font-semibold">
                        {step.label}
                      </span>
                      {step.durationMs > 0 && (
                        <span className="text-xs text-muted-foreground">
                          {fmtDur(step.durationMs)}
                        </span>
                      )}
                      <span className={`text-xs font-mono ${statusColor}`}>
                        {statusSym}
                      </span>
                    </div>

                    {/* Step content */}
                    {step.type === "question" && meta && (
                      <p className="mt-0.5 text-xs text-muted-foreground">
                        {meta.query_preview}
                      </p>
                    )}

                    {step.type === "planning" && (
                      <div className="mt-1 space-y-0.5">
                        {step.spans.map((s, j) => {
                          const out = s.output_value ?? "";
                          const ok =
                            out.includes("## Цель") ||
                            out.includes("## План") ||
                            out.includes("## Используемые");
                          return (
                            <div
                              key={j}
                              className="text-xs text-muted-foreground"
                            >
                              <span
                                className={
                                  ok
                                    ? "text-emerald-400"
                                    : "text-rose-400"
                                }
                              >
                                {ok ? "✓" : "✗"}
                              </span>{" "}
                              {ok
                                ? "План сгенерирован"
                                : out.length > 100
                                  ? out.slice(0, 100) + "..."
                                  : out || "План не сгенерирован"}
                            </div>
                          );
                        })}
                        {step.skillName && (
                          <div className="text-xs text-muted-foreground">
                            → выбран:{" "}
                            <span className="font-medium">
                              {step.skillName}
                            </span>
                          </div>
                        )}
                      </div>
                    )}

                    {step.type === "instructions" && (
                      <div className="mt-1 space-y-0.5">
                        {step.spans.map((s, j) => {
                          const input = s.input_value ?? "";
                          const out = s.output_value ?? "";
                          let skillId = "—";
                          let details = false;
                          try {
                            const parsed = JSON.parse(input);
                            skillId = parsed.skill_id ?? "—";
                            details = parsed.details === true;
                          } catch {
                            /* skip */
                          }
                          const noExtended = out.includes("Extended instructions not available");
                          const hasContent = out.length > 20 && !noExtended;
                          let remark: string;
                          if (noExtended) {
                            remark = "основные инструкции (расширенные недоступны)";
                          } else if (details && hasContent) {
                            remark = "расширенные инструкции получены";
                          } else {
                            remark = "основные инструкции получены";
                          }
                          return (
                            <div
                              key={j}
                              className="text-xs text-muted-foreground"
                            >
                              <span className="font-medium">
                                {skillId}
                              </span>
                              {" → "}
                              {remark}
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {step.type === "tool" && (
                      <div className="mt-1 space-y-0.5">
                        {step.spans.map((s, j) => {
                          const input = s.input_value ?? "";
                          const out = s.output_value ?? "";
                          if (!input && !out) return null;
                          const txt = input || out;
                          const short =
                            txt.length > 300
                              ? txt.slice(0, 300) + "..."
                              : txt;
                          return (
                            <pre
                              key={j}
                              className="max-h-24 overflow-y-auto whitespace-pre-wrap break-all rounded bg-black/10 px-2 py-1 text-[10px] text-muted-foreground"
                            >
                              {short}
                            </pre>
                          );
                        })}
                      </div>
                    )}

                    {step.type === "llm" && (
                      <div className="mt-1 space-y-0.5">
                        {step.spans.map((s, j) => {
                          const errMsg = s.status_message ?? "";
                          const shortErr = errMsg.split("\n")[0] || errMsg;
                          return (
                            <div
                              key={j}
                              className="flex items-center gap-2 text-xs text-muted-foreground"
                            >
                              <span
                                className={
                                  s.status_code === "OK"
                                    ? "text-emerald-400"
                                    : "text-rose-400"
                                }
                              >
                                {s.status_code === "OK" ? "✓" : "✗"}
                              </span>
                              <span>
                                {s.total_tokens != null
                                  ? `${s.total_tokens} ток.`
                                  : "—"}
                              </span>
                              <span>· {fmtDur(s.duration_ms)}</span>
                              {s.status_code !== "OK" && shortErr && (
                                <span className="max-w-xs truncate text-rose-400" title={errMsg}>
                                  {shortErr}
                                </span>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  {/* Mini timing dot */}
                  {totalMs > 0 && step.durationMs > 0 && (
                    <span className="mt-1 shrink-0 text-[10px] text-muted-foreground">
                      ●
                    </span>
                  )}
                </div>

                {/* Click to expand details */}
                {hasDetails && (
                  <button
                    onClick={toggle}
                    className="self-start text-[10px] uppercase tracking-wider text-muted-foreground/50 hover:text-foreground"
                  >
                    {isExpanded ? "▲ Скрыть" : "▼ Подробнее"}
                  </button>
                )}

                {isExpanded && (
                  <div className="mt-1 max-h-80 space-y-2 overflow-y-auto rounded-lg bg-black/10 p-3">
                    {step.spans.map((s, j) => (
                      <div key={j}>
                        {s.input_value && (
                          <details className="group/io">
                            <summary className="cursor-pointer py-1 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                              Вход
                            </summary>
                            <pre className="whitespace-pre-wrap break-all pb-1 text-[11px] text-muted-foreground">
                              {s.input_value}
                            </pre>
                          </details>
                        )}
                        {s.output_value && (
                          <details className="group/io">
                            <summary className="cursor-pointer py-1 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                              Выход
                            </summary>
                            <pre className="whitespace-pre-wrap break-all pb-1 text-[11px] text-muted-foreground">
                              {s.output_value}
                            </pre>
                          </details>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function Phoenix() {
  const { resolvedTheme } = useTheme();
  const [overview, setOverview] = useState<PhoenixOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [iframeLoaded, setIframeLoaded] = useState(false);
  const iframeRef = useRef<HTMLIFrameElement | null>(null);

  const [traces, setTraces] = useState<PhoenixTraceRow[]>([]);
  const [traceTotal, setTraceTotal] = useState(0);
  const [tracePage, setTracePage] = useState(0);
  const [traceLoading, setTraceLoading] = useState(false);
  const [traceHistoryStatus, setTraceHistoryStatus] = useState<PhoenixTraceHistoryStatus>("idle");
  const traceLimit = 15;

  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);
  const [selectedSpans, setSelectedSpans] = useState<PhoenixSpanSnapshotItem[]>([]);
  const [selectedTraceMeta, setSelectedTraceMeta] = useState<PhoenixTraceRow | null>(null);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [dialogSize, setDialogSize] = useState<{ w?: number; h?: number }>({});
  const [copyLabel, setCopyLabel] = useState("📋 Копировать");

  const makeResize = useCallback((dim: "w" | "h" | "both") => (e: React.MouseEvent) => {
    e.preventDefault();
    const el = (e.currentTarget.parentElement as HTMLElement);
    if (!el) return;
    const startW = el.offsetWidth;
    const startH = el.offsetHeight;
    const startX = e.clientX;
    const startY = e.clientY;
    const onMove = (me: MouseEvent) => {
      const newW = dim !== "h" ? Math.max(640, startW + (me.clientX - startX)) : startW;
      const newH = dim !== "w" ? Math.max(400, startH + (me.clientY - startY)) : startH;
      setDialogSize({ w: newW, h: newH });
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, []);

  const buildExportText = useCallback((): string => {
    const meta = selectedTraceMeta;
    const spans = selectedSpans;
    const lines: string[] = [];

    lines.push("=== Детали запроса ===");
    if (meta) {
      lines.push(`Вопрос: ${meta.query_preview}`);
      lines.push(`Статус: ${meta.status === "success" ? "OK" : "Ошибка"}`);
      lines.push(`Модель: ${meta.model ?? "—"}`);
      lines.push(`Длительность: ${fmtDur(meta.duration_ms)}`);
      lines.push(`Сессия: ${meta.session_id ?? "—"}`);
    }
    lines.push("");

    if (spans.length === 0) {
      lines.push("Нет данных по спанам.");
      lines.push("");
      lines.push(`ID трассы: ${selectedTraceId ?? ""}`);
      return lines.join("\n");
    }

    const { steps } = buildTimelineSteps(spans, meta);
    lines.push("=== Ход выполнения ===");
    lines.push("");

    for (const step of steps) {
      if (step.type === "question") continue;
      lines.push(`${step.icon} ${step.label}`);
      if (step.durationMs > 0) {
        lines.push(`Длительность: ${fmtDur(step.durationMs)}`);
      }
      const sym =
        step.status === "ok" ? "✓" : step.status === "error" ? "✗" : "~";
      lines.push(`Статус: ${sym}`);
      lines.push("");

      for (const s of step.spans) {
        if (s.input_value) {
          lines.push("--- Вход ---");
          lines.push(s.input_value);
          lines.push("");
        }
        if (s.output_value) {
          lines.push("--- Выход ---");
          lines.push(s.output_value);
          lines.push("");
        }
      }
      lines.push("");
    }

    lines.push(`ID трассы: ${selectedTraceId ?? ""}`);
    return lines.join("\n");
  }, [selectedTraceMeta, selectedSpans, selectedTraceId]);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(buildExportText());
      setCopyLabel("✅ Скопировано!");
      setTimeout(() => setCopyLabel("📋 Копировать"), 2000);
    } catch {
      setCopyLabel("❌ Ошибка");
      setTimeout(() => setCopyLabel("📋 Копировать"), 2000);
    }
  }, [buildExportText]);

  const handleDownload = useCallback(() => {
    const text = buildExportText();
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trassa_${(selectedTraceId ?? "неизвестно").slice(0, 12)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [buildExportText, selectedTraceId]);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await getPhoenixOverview();
      setOverview(next);
      setIframeLoaded(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось загрузить данные наблюдаемости.");
      setTraceHistoryStatus("unavailable");
    } finally {
      setLoading(false);
    }
  };

  const loadTraces = useCallback(async (page: number) => {
    setTraceLoading(true);
    setTraceHistoryStatus("loading");
    try {
      const offset = page * traceLimit;
      const result = await getPhoenixTraces(traceLimit, offset);
      setTraces(result.traces);
      setTraceTotal(result.total);
      setTracePage(page);
      setTraceHistoryStatus("loaded");
    } catch {
      setTraces([]);
      setTraceTotal(0);
      setTraceHistoryStatus("error");
    } finally {
      setTraceLoading(false);
    }
  }, []);

  const openTraceDetail = useCallback(async (trace: PhoenixTraceRow) => {
    setSelectedTraceId(trace.trace_id);
    setSelectedTraceMeta(trace);
    setDetailLoading(true);
    setSelectedSpans([]);
    setSelectedProjectId(null);
    try {
      const detail = trace.session_id
        ? await getPhoenixTracesBySession(trace.session_id)
        : await getPhoenixTraceDetail(trace.trace_id);
      setSelectedSpans(detail.spans);
      if (detail.project_id) setSelectedProjectId(detail.project_id);
    } catch {
      setSelectedSpans([]);
    } finally {
      setDetailLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, []);

  useEffect(() => {
    if (overview?.available) {
      void loadTraces(0);
    } else if (overview) {
      setTraceHistoryStatus("unavailable");
    }
  }, [overview?.available, loadTraces]);

  useEffect(() => {
    const nextTheme = resolvedTheme === "light" ? "light" : "dark";
    const storageKey = "arize-phoenix-theme";
    const previousTheme = window.localStorage.getItem(storageKey);
    if (previousTheme === nextTheme) {
      return;
    }

    window.localStorage.setItem(storageKey, nextTheme);

    if (iframeLoaded) {
      iframeRef.current?.contentWindow?.location.reload();
      setIframeLoaded(false);
    }
  }, [resolvedTheme, iframeLoaded]);

  const filteredTraces = traces.filter((trace) => traceMatches(trace, search));
  const tokenTelemetryAvailable = (overview?.token_usage ?? []).some(tokenRowHasTelemetry);
  const tokenUsageByDay = buildTokenUsageByDay(overview?.token_usage ?? []);
  const traceHistorySummary = formatPhoenixTraceHistorySummary({
    status: traceHistoryStatus,
    total: traceTotal,
    page: tracePage,
    limit: traceLimit,
  });
  const traceEmptyMessage = getPhoenixTraceEmptyMessage({
    status: traceHistoryStatus,
    hasSearch: Boolean(search.trim()),
  });
  const phoenixUiUrl = resolvePhoenixUiBaseUrl(import.meta.env);
  const phoenixTraceUrl = selectedProjectId
    ? buildPhoenixProjectTraceUrl(import.meta.env, selectedProjectId, selectedTraceId ?? "")
    : phoenixUiUrl;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navigation />

      <main className="mx-auto max-w-[1460px] px-4 py-5 sm:px-6 lg:px-8 lg:py-6 xl:py-16">
        <div className="mb-4 flex flex-col gap-4 lg:mb-5 lg:flex-row lg:items-end lg:justify-between xl:mb-10">
          <div>
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-3 inline-flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-emerald-400 lg:mb-2 xl:mb-5"
            >
              <Activity className="h-3.5 w-3.5" />
              <span className="text-[11px] font-bold uppercase tracking-[0.24em]">
                Phoenix Live
              </span>
            </motion.div>
            <h1 className="mb-1 text-2xl font-bold tracking-tight lg:mb-2 lg:text-3xl xl:mb-3 xl:text-4xl">Наблюдаемость и трассировка</h1>
            <p className="max-w-3xl text-sm text-muted-foreground lg:text-base xl:text-lg">
              Сквозная трассировка вызовов, мониторинг инструментов и анализ стоимости токенов в реальном времени.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {overview ? (
              <a
                href={phoenixUiUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex h-10 items-center gap-2 rounded-2xl border border-border/50 bg-secondary px-4 text-sm font-bold transition-all hover:bg-muted xl:h-12 xl:px-5"
              >
                <ExternalLink className="h-4 w-4" />
                Открыть Phoenix
              </a>
            ) : null}
            <button
              type="button"
              onClick={() => void load()}
              className="inline-flex h-10 items-center gap-2 rounded-2xl border border-border/50 bg-primary px-4 text-sm font-bold text-primary-foreground transition-all hover:opacity-95 active:scale-[0.99] xl:h-12 xl:px-5"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
              Обновить данные
            </button>
          </div>
        </div>

        {error ? (
          <div className="rounded-[28px] border border-rose-500/20 bg-rose-500/10 p-8 text-rose-200 shadow-sm">
            <div className="mb-2 flex items-center gap-3 text-lg font-bold">
              <AlertTriangle className="h-5 w-5" />
              Не удалось загрузить Phoenix overview
            </div>
            <p className="text-sm text-rose-100/80">{error}</p>
          </div>
        ) : null}

        {!error && !overview?.available && !loading ? (
          <div className="rounded-[28px] border border-amber-500/20 bg-amber-500/10 p-8 text-amber-100 shadow-sm">
            <div className="mb-2 flex items-center gap-3 text-lg font-bold">
              <AlertTriangle className="h-5 w-5 text-amber-400" />
              Phoenix доступен не полностью
            </div>
            <p className="text-sm text-amber-100/80">
              {overview?.warnings[0] ?? "Backend пока не смог получить данные из Phoenix API."}
            </p>
          </div>
        ) : null}

        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2 lg:mt-5 lg:gap-4 xl:mt-10 xl:gap-6 xl:grid-cols-4">
          {[
            {
              label: "Трассы запросов",
              value: overview ? formatNumber(overview.stats.total_traces) : "—",
              hint: "реальные корневые трассы Phoenix",
              icon: Layers,
              accent: "text-sky-400",
            },
            {
              label: "Успешность",
              value: overview ? `${overview.stats.success_rate.toFixed(1)}%` : "—",
              hint: "доля успешных трасс запросов",
              icon: CheckCircle2,
              accent: "text-emerald-400",
            },
            {
              label: "P50 задержки",
              value: overview ? formatDuration(overview.stats.p50_latency_ms) : "—",
              hint: "медианная длительность запроса",
              icon: Clock,
              accent: "text-amber-400",
            },
            {
              label: "Сессии",
              value: overview ? formatNumber(overview.stats.unique_sessions) : "—",
              hint: "охваченные рабочие сессии",
              icon: Server,
              accent: "text-violet-400",
            },
          ].map((card, index) => (
            <motion.div
              key={card.label}
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              className="rounded-2xl border border-border/50 bg-card p-3.5 shadow-sm lg:rounded-3xl lg:p-4 xl:rounded-[28px] xl:p-6"
            >
              <div className="mb-3 flex items-center justify-between xl:mb-5">
                <div className={`rounded-xl bg-white/5 p-2.5 xl:rounded-2xl xl:p-3 ${card.accent}`}>
                  <card.icon className="h-4 w-4 xl:h-5 xl:w-5" />
                </div>
                {loading ? <div className="h-2 w-20 rounded-full bg-muted/70" /> : null}
              </div>
              <div className="mb-1 text-2xl font-bold tracking-tight xl:text-3xl">{card.value}</div>
              <div className="text-sm font-medium text-foreground">{card.label}</div>
              <div className="mt-1 text-xs text-muted-foreground">{card.hint}</div>
            </motion.div>
          ))}
        </div>

        <div className="mt-4 grid grid-cols-1 gap-3 lg:mt-5 lg:gap-4 xl:mt-10 xl:gap-8 xl:grid-cols-[1.25fr_0.95fr]">
          <section className="rounded-2xl border border-border/50 bg-card p-4 shadow-sm lg:rounded-3xl lg:p-5 xl:rounded-[32px] xl:p-8">
            <div className="mb-3 flex items-start justify-between gap-4 xl:mb-6">
              <div>
                <h2 className="text-lg font-bold xl:text-xl">Задержка по запросам</h2>
              </div>
              <div className="flex items-center gap-2">
                <div className="rounded-full border border-border/50 bg-secondary px-3 py-1 text-[11px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                  7 дней
                </div>
                <div className="rounded-full border border-border/50 bg-secondary px-3 py-1 text-[11px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                  p50 / p95 / p99
                </div>
              </div>
            </div>

            <div className="h-[200px] w-full lg:h-[220px] xl:h-[320px]">
              {overview && overview.latency.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={overview.latency}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.06)" />
                    <XAxis
                      dataKey="label"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#71717a", fontSize: 11, fontWeight: 600 }}
                      dy={10}
                    />
                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#71717a", fontSize: 11, fontWeight: 600 }}
                      tickFormatter={(value) => `${Math.round(Number(value) / 1000)}с`}
                    />
                    <Tooltip
                      formatter={(value: number) => formatDuration(Number(value))}
                      contentStyle={{
                        backgroundColor: "var(--card)",
                        border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: "16px",
                      }}
                    />
                    <Line type="monotone" dataKey="p50_ms" stroke="#60a5fa" strokeWidth={3} dot={false} />
                    <Line type="monotone" dataKey="p95_ms" stroke="#f59e0b" strokeWidth={2.4} dot={false} />
                    <Line type="monotone" dataKey="p99_ms" stroke="#c084fc" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center rounded-[28px] border border-dashed border-border/40 bg-secondary/20 text-center">
                  <div>
                    <Clock className="mx-auto mb-3 h-8 w-8 text-muted-foreground/60" />
                    <div className="font-semibold">Пока нет данных по задержке</div>
                    <div className="mt-1 text-sm text-muted-foreground">
                      Phoenix не вернул трассы запросов за выбранный период.
                    </div>
                  </div>
                </div>
              )}
            </div>
          </section>

          <section className="rounded-2xl border border-border/50 bg-card p-4 shadow-sm lg:rounded-3xl lg:p-5 xl:rounded-[32px] xl:p-8">
            <div className="mb-3 flex items-start justify-between gap-4 xl:mb-6">
              <div>
                <h2 className="text-lg font-bold xl:text-xl">Использование токенов</h2>
              </div>
              <div className="flex items-center gap-2">
                <div className="rounded-full border border-border/50 bg-secondary px-3 py-1 text-[11px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                  7 дней
                </div>
                <Zap className="h-5 w-5 text-primary" />
              </div>
            </div>

            {overview && overview.warnings.length > 0 ? (
              <div className="mb-4 rounded-2xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-900 dark:border-amber-500/20 dark:text-amber-100">
                {overview.warnings[0]}
              </div>
            ) : null}

            <div className="h-[200px] w-full lg:h-[220px] xl:h-[320px]">
              {tokenTelemetryAvailable && tokenUsageByDay.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={tokenUsageByDay} barGap={12}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.06)" />
                    <XAxis
                      dataKey="label"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#71717a", fontSize: 11, fontWeight: 600 }}
                      dy={10}
                    />
                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#71717a", fontSize: 11, fontWeight: 600 }}
                      tickFormatter={(value) => formatNumber(Number(value))}
                    />
                    <Tooltip
                      formatter={(value: number, name: string) => [
                        formatNumber(Number(value)),
                        name === "input" ? "Вход" : "Выход",
                      ]}
                      contentStyle={{
                        backgroundColor: "var(--card)",
                        border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: "16px",
                      }}
                    />
                    <Bar dataKey="input" name="input" fill="#3b82f6" radius={[8, 8, 0, 0]} maxBarSize={42} />
                    <Bar dataKey="output" name="output" fill="#7baaf7" radius={[8, 8, 0, 0]} maxBarSize={42} opacity={0.72} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center rounded-[28px] border border-dashed border-border/40 bg-secondary/20 text-center">
                  <div>
                    <Zap className="mx-auto mb-3 h-8 w-8 text-muted-foreground/60" />
                    <div className="font-semibold">Нет информации</div>
                    <div className="mt-1 text-sm text-muted-foreground">
                      Трассы Phoenix доступны, но использование токенов не передается текущим провайдером.
                    </div>
                  </div>
                </div>
              )}
            </div>
          </section>
        </div>

        <section className="mt-6 overflow-hidden rounded-2xl border border-border/50 bg-card shadow-sm lg:rounded-3xl xl:mt-10 xl:rounded-[32px]">
          <div className="flex flex-col gap-4 border-b border-border/40 px-5 py-4 lg:flex-row lg:items-center lg:justify-between lg:px-6 lg:py-5 xl:px-8 xl:py-6">
            <div className="flex items-center gap-3">
              <Activity className="h-4 w-4 text-amber-400" />
              <h2 className="text-xl font-bold">Phoenix UI · Встроенная доска</h2>
            </div>
            {overview ? (
              <a
                href={phoenixUiUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 self-start rounded-full border border-border/50 bg-secondary px-4 py-2 text-sm font-semibold transition-all hover:bg-muted"
              >
                <ExternalLink className="h-4 w-4" />
                Открыть в новой вкладке
              </a>
            ) : null}
          </div>

          {overview ? (
            <div className="relative h-[480px] w-full lg:h-[600px] xl:h-[720px]">
              {!iframeLoaded ? (
                <div
                  className="absolute inset-0 flex items-center justify-center text-center"
                  style={{
                    backgroundColor: "var(--phoenix-embed-bg)",
                    backgroundImage:
                      "repeating-linear-gradient(45deg, var(--phoenix-embed-stripe) 0 12px, rgba(255, 255, 255, 0) 12px 24px)",
                  }}
                >
                  <div>
                    <Server className="mx-auto mb-4 h-9 w-9 text-muted-foreground/45" />
                    <div className="mb-2 text-[15px] font-bold text-foreground/80">
                      Встроенная доска Phoenix
                    </div>
                    <div className="text-[13px] text-muted-foreground">
                      встроенный фрейм → {phoenixUiUrl}
                    </div>
                  </div>
                </div>
              ) : null}
              <iframe
                ref={iframeRef}
                src={phoenixUiUrl}
                title="Доска Phoenix"
                className={`h-full w-full bg-background transition-opacity duration-300 ${
                  iframeLoaded ? "opacity-100" : "opacity-0"
                }`}
                onLoad={() => {
                  setIframeLoaded(true);
                }}
              />
            </div>
          ) : (
            <div className="flex h-[240px] items-center justify-center">
              <div className="text-center">
                <Server className="mx-auto mb-3 h-8 w-8 text-muted-foreground/60" />
                <div className="font-semibold">Нет URL для встроенной доски</div>
              </div>
            </div>
          )}
        </section>

        <section className="mt-6 overflow-hidden rounded-2xl border border-border/50 bg-card shadow-sm lg:rounded-3xl xl:mt-10 xl:rounded-[32px]">
          <div className="flex flex-col gap-4 border-b border-border/40 px-5 py-4 lg:flex-row lg:items-center lg:justify-between lg:px-6 lg:py-5 xl:px-8 xl:py-6">
            <div>
              <h2 className="text-xl font-bold">История запросов</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                {traceHistorySummary}
              </p>
            </div>

            <label className="relative block">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Поиск по запросам..."
                className="h-11 w-72 rounded-2xl border border-border/50 bg-secondary pl-10 pr-4 text-sm outline-none transition-all focus:border-primary/40"
              />
            </label>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead>
                <tr className="bg-secondary/35">
                  <th className="px-8 py-4 text-[11px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                    Запрос
                  </th>
                  <th className="px-6 py-4 text-[11px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                    Статус
                  </th>
                  <th className="px-6 py-4 text-[11px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                    Задержка
                  </th>
                  <th className="px-6 py-4 text-[11px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                    Инструменты
                  </th>
                  <th className="px-6 py-4 text-[11px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                    Спаны
                  </th>
                  <th className="px-6 py-4 text-[11px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                    Модель
                  </th>
                  <th className="px-8 py-4 text-right text-[11px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                    Время
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/20">
                {filteredTraces.length === 0 && !traceLoading && traceEmptyMessage && (
                  <tr>
                    <td colSpan={7} className="px-8 py-12 text-center text-sm text-muted-foreground">
                      {traceEmptyMessage}
                    </td>
                  </tr>
                )}
                {filteredTraces.map((trace) => (
                  <tr
                    key={trace.trace_id}
                    className="cursor-pointer transition-colors hover:bg-white/5"
                    onClick={() => void openTraceDetail(trace)}
                  >
                    <td className="px-8 py-5">
                      <div className="font-semibold">{trace.query_preview}</div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {formatRequestKind(trace.request_kind)} • {trace.user ?? "неизвестно"} • {trace.session_id ?? "нет сессии"}
                      </div>
                    </td>
                    <td className="px-6 py-5">
                      <span
                        className={`inline-flex rounded-full border px-3 py-1 text-[10px] font-bold uppercase tracking-[0.18em] ${
                          trace.status === "success"
                            ? "border-emerald-500/20 bg-emerald-500/10 text-emerald-400"
                            : "border-rose-500/20 bg-rose-500/10 text-rose-400"
                        }`}
                      >
                        {trace.status === "success" ? "OK" : "Ошибка"}
                      </span>
                    </td>
                    <td className="px-6 py-5 font-semibold">{formatDuration(trace.duration_ms)}</td>
                    <td className="px-6 py-5 font-semibold">{formatNumber(trace.tool_calls)}</td>
                    <td className="px-6 py-5 font-semibold">{formatNumber(trace.span_count)}</td>
                    <td className="px-6 py-5 font-semibold text-muted-foreground">{trace.model ?? "—"}</td>
                    <td className="px-8 py-5 text-right text-sm text-muted-foreground">
                      {formatDateTime(trace.started_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {traceTotal > traceLimit && (
            <div className="flex items-center justify-between border-t border-border/40 px-8 py-4">
              <button
                disabled={tracePage === 0}
                onClick={() => void loadTraces(tracePage - 1)}
                className="rounded-xl border border-border/50 bg-secondary px-4 py-2 text-sm font-semibold transition-all hover:bg-muted disabled:opacity-30 disabled:cursor-not-allowed"
              >
                ← Назад
              </button>
              <span className="text-sm text-muted-foreground">
                {tracePage + 1} / {Math.max(1, Math.ceil(traceTotal / traceLimit))}
              </span>
              <button
                disabled={(tracePage + 1) * traceLimit >= traceTotal}
                onClick={() => void loadTraces(tracePage + 1)}
                className="rounded-xl border border-border/50 bg-secondary px-4 py-2 text-sm font-semibold transition-all hover:bg-muted disabled:opacity-30 disabled:cursor-not-allowed"
              >
                Вперед →
              </button>
            </div>
          )}
        </section>
      </main>

      <Dialog open={selectedTraceId !== null} onOpenChange={(open) => { if (!open) setSelectedTraceId(null); }}>
        <DialogContent
          className="group sm:max-w-5xl max-h-[calc(100vh-2rem)] overflow-y-auto"
          style={
            dialogSize.w || dialogSize.h
              ? { width: dialogSize.w, maxWidth: dialogSize.w, height: dialogSize.h, maxHeight: dialogSize.h }
              : undefined
          }
        >
          {/* Resize: right edge (horizontal) */}
          <div className="absolute right-0 top-0 bottom-0 z-50 w-1.5 cursor-ew-resize opacity-0 transition-opacity hover:opacity-100 group-hover:opacity-50" onMouseDown={makeResize("w")} />
          {/* Resize: bottom edge (vertical) */}
          <div className="absolute bottom-0 left-0 right-0 z-50 h-1.5 cursor-ns-resize opacity-0 transition-opacity hover:opacity-100 group-hover:opacity-50" onMouseDown={makeResize("h")} />
          {/* Resize: corner (both) */}
          <div className="absolute bottom-0 right-0 z-50 cursor-se-resize p-1.5 text-muted-foreground/30 hover:text-muted-foreground/60 select-none" onMouseDown={makeResize("both")}>
            <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M12 0v12H0" />
            </svg>
          </div>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-amber-400" />
              Детали запроса
              <div className="ml-auto flex items-center gap-1.5">
                <button
                  onClick={handleCopy}
                  className="rounded-lg border border-border/50 bg-secondary px-3 py-1 text-[11px] font-semibold transition-all hover:bg-muted"
                >
                  {copyLabel}
                </button>
                <button
                  onClick={handleDownload}
                  className="rounded-lg border border-border/50 bg-secondary px-3 py-1 text-[11px] font-semibold transition-all hover:bg-muted"
                >
                  ⬇ Скачать .txt
                </button>
              </div>
            </DialogTitle>
          </DialogHeader>

          {selectedTraceMeta && (
            <div className="mb-4 grid grid-cols-2 gap-3 rounded-2xl bg-secondary/30 p-4 text-sm">
              <div>
                <span className="text-muted-foreground">Запрос: </span>
                <span className="font-medium">{selectedTraceMeta.query_preview}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Статус: </span>
                <span className={selectedTraceMeta.status === "success" ? "text-emerald-400" : "text-rose-400"}>
                  {selectedTraceMeta.status === "success" ? "OK" : "Ошибка"}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Длительность: </span>
                <span className="font-medium">{formatDuration(selectedTraceMeta.duration_ms)}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Модель: </span>
                <span className="font-medium">{selectedTraceMeta.model ?? "—"}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Сессия: </span>
                <span className="font-medium">{selectedTraceMeta.session_id ?? "—"}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Вызовов инструментов: </span>
                <span className="font-medium">{selectedTraceMeta.tool_calls}</span>
              </div>
              {selectedTraceMeta.skill_ids && (
                <div className="col-span-2">
                  <span className="text-muted-foreground">Навыки: </span>
                  <span className="font-medium">{selectedTraceMeta.skill_ids}</span>
                </div>
              )}
            </div>
          )}

          <div className="rounded-2xl border border-border/40">
            <div className="border-b border-border/40 px-4 py-3 text-xs font-bold uppercase tracking-widest text-muted-foreground">
              Таймлайн
            </div>
            {detailLoading && (
              <div className="flex items-center justify-center py-12 text-sm text-muted-foreground">
                Загрузка спанов...
              </div>
            )}
            {!detailLoading && selectedSpans.length === 0 && (
              <div className="flex items-center justify-center py-12 text-sm text-muted-foreground">
                Нет данных по спанам
              </div>
            )}
            {!detailLoading && selectedSpans.length > 0 && (
              <div className="px-4 py-4">
                <TraceTimeline spans={selectedSpans} meta={selectedTraceMeta} />
                {/* Collapsible raw span tree */}
                <details className="group mt-6">
                  <summary className="cursor-pointer text-xs font-bold uppercase tracking-widest text-muted-foreground hover:text-foreground">
                    Показать все спаны ({selectedSpans.length})
                  </summary>
                  <div className="mt-3 divide-y divide-border/10">
                    {buildSpanTree(selectedSpans).map((level, levelIdx) =>
                      level.map((span) => (
                        <SpanTreeRow key={span.span_id} span={span} depth={levelIdx} />
                      ))
                    )}
                  </div>
                </details>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between gap-3">
            <span className="text-xs text-muted-foreground select-all">
              ID трассы: {selectedTraceId ?? ""}
            </span>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">
                Не открывается? <a href={phoenixUiUrl} target="_blank" rel="noreferrer" className="underline">Откройте Phoenix</a> и вставьте ID трассы в поиск.
              </span>
              <a
                href={phoenixTraceUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-full border border-border/50 bg-secondary px-4 py-2 text-xs font-semibold transition-all hover:bg-muted"
              >
                <ExternalLink className="h-3 w-3" />
                Открыть в Phoenix
              </a>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
