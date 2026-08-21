import { useMemo } from "react";
import type { ArtifactPayload } from "../../lib/backend-types";

type PlanfactDashboardData = {
  period_label?: string;
  kpi?: Record<string, unknown>;
  control?: Record<string, unknown>;
  summary?: Record<string, unknown>;
  suggested_questions?: string[];
  metric_type?: string;
  plan_metric?: string;
};

function number(value: unknown): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

function money(value: unknown, signed = false): string {
  const numeric = number(value);
  const sign = signed && numeric > 0 ? "+" : "";
  const absolute = Math.abs(numeric);
  if (absolute >= 1_000_000) return `${sign}${(numeric / 1_000_000).toFixed(1).replace(".", ",")} млн ₽`;
  return `${sign}${(numeric / 1_000).toFixed(1).replace(".", ",")} тыс. ₽`;
}

function percent(value: unknown): string {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${numeric.toFixed(1).replace(".", ",")}%` : "н/д";
}

function toneClass(tone: string): string {
  if (tone === "risk") return "border-rose-500/25 bg-rose-500/10 text-rose-700 dark:text-rose-200";
  if (tone === "saving") return "border-emerald-500/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-200";
  if (tone === "quality") return "border-amber-500/25 bg-amber-500/10 text-amber-800 dark:text-amber-200";
  return "border-border/50 bg-secondary/35 text-foreground";
}

function varianceTone(payload: PlanfactDashboardData, variance: unknown): "risk" | "saving" | "neutral" {
  const metric = String(payload.metric_type ?? payload.kpi?.metric_type ?? payload.plan_metric ?? "").toLowerCase();
  const positiveIsGood = /revenue|sales|income|profit|margin|cash_in|net_cash|выруч|доход|прибыл/.test(metric);
  const value = number(variance);
  if (value === 0) return "neutral";
  return (value > 0) === positiveIsGood ? "saving" : "risk";
}

export function PlanfactFirstLook({
  artifact,
  onAsk,
}: {
  artifact: ArtifactPayload;
  onAsk: (question: string) => Promise<void>;
}) {
  const payload = (artifact.data?.data ?? {}) as PlanfactDashboardData;
  const kpi = payload.kpi ?? {};
  const control = payload.control ?? {};
  const summary = payload.summary ?? {};
  const suggestedQuestions = Array.isArray(payload.suggested_questions)
    ? payload.suggested_questions.filter((question) => question.trim())
    : [];
  const period = payload.period_label || "период не определён";
  const tone = varianceTone(payload, kpi.variance);
  const topDeviation = Array.isArray(summary.key_deviations)
    ? (summary.key_deviations[0] as Record<string, unknown> | undefined)
    : undefined;

  const conclusion = useMemo(() => {
    const variance = number(kpi.variance);
    const status = variance > 0
      ? `Факт превысил план на ${money(variance, true)}, исполнение составило ${percent(kpi.execution_pct)}.`
      : variance < 0
        ? `План исполнен с экономией ${money(variance, true)}, исполнение составило ${percent(kpi.execution_pct)}.`
        : `Факт соответствует плану, исполнение составило ${percent(kpi.execution_pct)}.`;
    const parts = [`За ${period}: ${status}`];
    if (summary.main_driver) {
      parts.push(`Основной вклад внесло ЦФО «${String(summary.main_driver)}» с отклонением ${money(summary.main_driver_variance, true)}.`);
    }
    if (topDeviation?.article) {
      parts.push(`Ключевая статья: «${String(topDeviation.article)}» (${money(topDeviation.variance_amount, true)}).`);
    }
    if (number(control.fact_without_plan_count) || number(control.plan_without_fact_count)) {
      parts.push("Есть операции без плана или плановые статьи без факта — их стоит проверить отдельно.");
    }
    return parts.join(" ");
  }, [control.fact_without_plan_count, control.plan_without_fact_count, kpi.execution_pct, kpi.variance, period, summary.main_driver, summary.main_driver_variance, topDeviation]);
  const priority = useMemo(() => {
    const actions: string[] = [];
    if (topDeviation?.article) actions.push(`разобрать причины отклонения по статье «${String(topDeviation.article)}»`);
    if (number(control.fact_without_plan_count)) actions.push(`проверить ${number(control.fact_without_plan_count)} операций без плана`);
    if (number(control.plan_without_fact_count)) actions.push(`уточнить исполнение по ${number(control.plan_without_fact_count)} плановым статьям без факта`);
    return actions.length
      ? `Приоритет действий: ${actions.join("; ")}.`
      : "Критичных разрывов между планом и фактом не обнаружено; рекомендуется контролировать крупнейшие отклонения по ЦФО и статьям.";
  }, [control.fact_without_plan_count, control.plan_without_fact_count, topDeviation]);

  const attention = [
    ["Наибольшее отклонение по статье", topDeviation ? money(topDeviation.variance_amount, true) : "—", String(topDeviation?.article ?? "Статья не определена"), topDeviation ? varianceTone(payload, topDeviation.variance_amount) : "neutral"],
    ["Наибольшее отклонение по ЦФО", String(summary.main_driver ?? "—"), "Главный драйвер общего отклонения", tone],
    ["Факт без плана", String(kpi.fact_without_plan_count ?? 0), "Факт без плана по статье", "quality"],
    ["План без факта", String(kpi.plan_without_fact_count ?? 0), "План без факта по статье", "quality"],
  ] as const;

  return (
    <section className="mb-4 rounded-[24px] border border-border/25 bg-card/70 p-4 shadow-sm lg:p-6">
      <div>
        <h2 className="text-xl font-bold tracking-tight">Первичный ИИ-анализ</h2>
        <p className="mt-1 text-sm text-muted-foreground">Фокусный период: {period}</p>
      </div>

      <div className="mt-4 rounded-2xl border border-primary/20 bg-primary/5 p-4">
        <div className="text-xs font-bold uppercase tracking-wider text-primary">Управленческий вывод</div>
        <p className="mt-2 text-base font-semibold leading-relaxed">{conclusion}</p>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{priority}</p>
      </div>

      <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
        {[
          ["План", money(kpi.plan), "neutral"],
          ["Факт", money(kpi.fact), "neutral"],
          ["Отклонение", money(kpi.variance, true), tone],
          ["Исполнение", percent(kpi.execution_pct), tone],
          ["ЦФО с превышением", String(kpi.cfo_overruns ?? 0), "risk"],
          ["ЦФО с экономией", String(kpi.cfo_savings ?? 0), "saving"],
        ].map(([label, value, cardTone]) => (
          <div key={String(label)} className={`min-h-[84px] rounded-2xl border px-4 py-3 ${toneClass(String(cardTone))}`}>
            <div className="text-[11px] font-bold uppercase tracking-wider opacity-75">{label}</div>
            <div className="mt-2 text-xl font-bold tracking-tight">{value}</div>
          </div>
        ))}
      </div>

      <div className="mt-5">
        <div className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Зоны внимания</div>
        <div className="mt-2 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {attention.map(([title, value, description, cardTone]) => (
            <div key={title} className={`rounded-2xl border px-4 py-3 ${toneClass(cardTone)}`}>
              <div className="text-[11px] font-bold uppercase tracking-wider opacity-75">{title}</div>
              <div className="mt-2 text-base font-bold leading-tight">{value}</div>
              <div className="mt-1 text-xs opacity-80">{description}</div>
            </div>
          ))}
        </div>
      </div>

      {suggestedQuestions.length ? (
        <div className="mt-5 rounded-2xl border border-border/40 bg-secondary/15 p-4">
          <div className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Что спросить дальше?</div>
          <div className="mt-3 flex flex-wrap gap-2">
            {suggestedQuestions.map((question) => (
              <button
                key={question}
                type="button"
                onClick={() => void onAsk(question)}
                className="rounded-xl border border-primary/20 bg-primary/10 px-3 py-2 text-left text-xs font-bold text-primary transition hover:bg-primary/15"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}
