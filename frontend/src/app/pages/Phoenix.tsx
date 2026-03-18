import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { useTheme } from "next-themes";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
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
import { getPhoenixOverview } from "../lib/backend-api";
import type { PhoenixOverview, PhoenixTokenUsageRow, PhoenixTraceRow } from "../lib/backend-types";

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

  return Array.from(buckets.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, value]) => value);
}

export function Phoenix() {
  const { resolvedTheme } = useTheme();
  const [overview, setOverview] = useState<PhoenixOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [iframeLoaded, setIframeLoaded] = useState(false);
  const iframeRef = useRef<HTMLIFrameElement | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await getPhoenixOverview();
      setOverview(next);
      setIframeLoaded(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось загрузить observability-данные.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

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

  const filteredTraces = (overview?.traces ?? []).filter((trace) => traceMatches(trace, search));
  const tokenTelemetryAvailable = (overview?.token_usage ?? []).some(tokenRowHasTelemetry);
  const tokenUsageByDay = buildTokenUsageByDay(overview?.token_usage ?? []);
  const phoenixUiUrl = "/phoenix/";

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navigation />

      <main className="mx-auto max-w-[1460px] px-8 py-16">
        <div className="mb-10 flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-5 inline-flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-emerald-400"
            >
              <Activity className="h-3.5 w-3.5" />
              <span className="text-[11px] font-bold uppercase tracking-[0.24em]">
                Phoenix Live
              </span>
            </motion.div>
            <h1 className="mb-3 text-4xl font-bold tracking-tight">Наблюдаемость и трассировка</h1>
            <p className="max-w-3xl text-lg text-muted-foreground">
              Сквозная трассировка вызовов, мониторинг инструментов и анализ стоимости токенов в реальном времени.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {overview ? (
              <a
                href={phoenixUiUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex h-12 items-center gap-2 rounded-2xl border border-border/50 bg-secondary px-5 text-sm font-bold transition-all hover:bg-muted"
              >
                <ExternalLink className="h-4 w-4" />
                Открыть Phoenix
              </a>
            ) : null}
            <button
              type="button"
              onClick={() => void load()}
              className="inline-flex h-12 items-center gap-2 rounded-2xl border border-border/50 bg-primary px-5 text-sm font-bold text-primary-foreground transition-all hover:opacity-95 active:scale-[0.99]"
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

        <div className="mt-10 grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-4">
          {[
            {
              label: "Трассы запросов",
              value: overview ? formatNumber(overview.stats.total_traces) : "—",
              hint: "реальные root traces Phoenix",
              icon: Layers,
              accent: "text-sky-400",
            },
            {
              label: "Успешность",
              value: overview ? `${overview.stats.success_rate.toFixed(1)}%` : "—",
              hint: "доля success по request traces",
              icon: CheckCircle2,
              accent: "text-emerald-400",
            },
            {
              label: "P50 latency",
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
              className="rounded-[28px] border border-border/50 bg-card p-6 shadow-sm"
            >
              <div className="mb-5 flex items-center justify-between">
                <div className={`rounded-2xl bg-white/5 p-3 ${card.accent}`}>
                  <card.icon className="h-5 w-5" />
                </div>
                {loading ? <div className="h-2 w-20 rounded-full bg-muted/70" /> : null}
              </div>
              <div className="mb-1 text-3xl font-bold tracking-tight">{card.value}</div>
              <div className="text-sm font-medium text-foreground">{card.label}</div>
              <div className="mt-1 text-xs text-muted-foreground">{card.hint}</div>
            </motion.div>
          ))}
        </div>

        <div className="mt-10 grid grid-cols-1 gap-8 xl:grid-cols-[1.25fr_0.95fr]">
          <section className="rounded-[32px] border border-border/50 bg-card p-8 shadow-sm">
            <div className="mb-6 flex items-start justify-between gap-4">
              <div>
                <h2 className="text-xl font-bold">Latency по запросам</h2>
              </div>
              <div className="rounded-full border border-border/50 bg-secondary px-3 py-1 text-[11px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                p50 / p95 / p99
              </div>
            </div>

            <div className="h-[320px] w-full">
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
                    <div className="font-semibold">Пока нет данных по latency</div>
                    <div className="mt-1 text-sm text-muted-foreground">
                      Phoenix не вернул request traces за выбранный период.
                    </div>
                  </div>
                </div>
              )}
            </div>
          </section>

          <section className="rounded-[32px] border border-border/50 bg-card p-8 shadow-sm">
            <div className="mb-6 flex items-start justify-between gap-4">
              <div>
                <h2 className="text-xl font-bold">Использование токенов</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Дневной профиль input/output токенов по реальным model runs из Phoenix.
                </p>
              </div>
              <Zap className="mt-1 h-5 w-5 text-primary" />
            </div>

            {overview && overview.warnings.length > 0 ? (
              <div className="mb-4 rounded-2xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-900 dark:border-amber-500/20 dark:text-amber-100">
                {overview.warnings[0]}
              </div>
            ) : null}

            <div className="h-[320px] w-full">
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
                        name === "input" ? "Input" : "Output",
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
                      Phoenix traces доступны, но token usage не передается текущим провайдером.
                    </div>
                  </div>
                </div>
              )}
            </div>
          </section>
        </div>

        <section className="mt-10 overflow-hidden rounded-[32px] border border-border/50 bg-card shadow-sm">
          <div className="flex flex-col gap-4 border-b border-border/40 px-8 py-6 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-3">
              <Activity className="h-4 w-4 text-amber-400" />
              <h2 className="text-xl font-bold">Phoenix UI · Встроенный дашборд</h2>
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
            <div className="relative h-[720px] w-full">
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
                      Phoenix Dashboard Embed
                    </div>
                    <div className="text-[13px] text-muted-foreground">
                      iframe → {phoenixUiUrl}
                    </div>
                  </div>
                </div>
              ) : null}
              <iframe
                ref={iframeRef}
                src={phoenixUiUrl}
                title="Phoenix Dashboard"
                className={`h-[720px] w-full bg-background transition-opacity duration-300 ${
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
                <div className="font-semibold">Нет URL для встроенного dashboard</div>
              </div>
            </div>
          )}
        </section>

        <section className="mt-10 overflow-hidden rounded-[32px] border border-border/50 bg-card shadow-sm">
          <div className="flex flex-col gap-4 border-b border-border/40 px-8 py-6 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h2 className="text-xl font-bold">Последние request traces</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                Список строится по реальным root traces и подтягивает модель, tool calls и span count.
              </p>
            </div>

            <label className="relative block">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Поиск по traces..."
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
                    Latency
                  </th>
                  <th className="px-6 py-4 text-[11px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                    Инструменты
                  </th>
                  <th className="px-6 py-4 text-[11px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                    Spans
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
                {filteredTraces.map((trace) => (
                  <tr key={trace.trace_id} className="transition-colors hover:bg-white/5">
                    <td className="px-8 py-5">
                      <div className="font-semibold">{trace.query_preview}</div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {trace.request_kind} • {trace.user ?? "unknown"} • {trace.session_id ?? "no-session"}
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
                        {trace.status === "success" ? "OK" : "Error"}
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
        </section>
      </main>
    </div>
  );
}
