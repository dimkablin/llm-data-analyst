import { useMemo, useState } from "react";
import { CheckCircle2, LoaderCircle } from "lucide-react";
import { confirmPlanfactSource, detectPlanfactSource } from "../../lib/backend-api";
import type { PlanfactDetectResponse } from "../../lib/backend-types";
import { summarizeError } from "../../lib/format";

type LooseRecord = Record<string, unknown>;

type PlanfactConfig = LooseRecord & {
  plan?: LooseRecord;
  fact?: LooseRecord;
  cfo_matching?: {
    rows?: LooseRecord[];
    plan_cfos?: LooseRecord[];
  };
  article_matching?: {
    stats?: LooseRecord;
    matches?: LooseRecord[];
    plan_articles?: LooseRecord[];
  };
  cfo_mapping?: LooseRecord[];
  article_mapping?: LooseRecord[];
};

type Props = {
  sessionId: string;
  disabled?: boolean;
  onConfirmed: () => Promise<void>;
};

type MatchFilter = "review" | "unmatched" | "suggested" | "confirmed" | "all";

function parseConfig(value: string): PlanfactConfig | null {
  try {
    const parsed: unknown = JSON.parse(value);
    return parsed && !Array.isArray(parsed) && typeof parsed === "object"
      ? (parsed as PlanfactConfig)
      : null;
  } catch {
    return null;
  }
}

function text(value: unknown, fallback = ""): string {
  const result = String(value ?? "").trim();
  return result || fallback;
}

function number(value: unknown): number {
  const result = Number(value ?? 0);
  return Number.isFinite(result) ? result : 0;
}

function formatMoney(value: unknown): string {
  const amount = number(value);
  const absolute = Math.abs(amount);
  const divisor = absolute >= 1_000_000 ? 1_000_000 : 1_000;
  const suffix = absolute >= 1_000_000 ? "млн ₽" : "тыс. ₽";
  return `${(amount / divisor).toLocaleString("ru-RU", { maximumFractionDigits: 1 })} ${suffix}`;
}

function matchLabel(value: string): string {
  return {
    exact: "Совпало автоматически",
    dictionary: "Найдено по словарю",
    fuzzy_auto: "Похоже, принято",
    fuzzy_suggested: "Требует подтверждения",
    manual: "Подтверждено вручную",
    unmatched: "Не найдена пара",
  }[value] ?? value;
}

function FieldSelect({
  label,
  value,
  columns,
  onChange,
  optional = false,
}: {
  label: string;
  value: unknown;
  columns: string[];
  onChange: (value: string) => void;
  optional?: boolean;
}) {
  return (
    <label className="grid gap-1 text-xs font-semibold text-muted-foreground">
      {label}
      <select
        value={text(value)}
        onChange={(event) => onChange(event.target.value)}
        className="rounded-xl border border-border/60 bg-background px-3 py-2 text-sm text-foreground"
      >
        <option value="">{optional ? "Не использовать" : "Выберите поле"}</option>
        {columns.map((column) => (
          <option key={column} value={column}>
            {column}
          </option>
        ))}
      </select>
    </label>
  );
}

export function PlanfactSourcePanel({ sessionId, disabled = false, onConfirmed }: Props) {
  const [planFile, setPlanFile] = useState<File | null>(null);
  const [factFile, setFactFile] = useState<File | null>(null);
  const [mappingFile, setMappingFile] = useState<File | null>(null);
  const [detection, setDetection] = useState<PlanfactDetectResponse | null>(null);
  const [configText, setConfigText] = useState("");
  const [matchFilter, setMatchFilter] = useState<MatchFilter>("review");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const config = useMemo(() => parseConfig(configText), [configText]);

  const cfoMatching = useMemo(() => {
    if (!config?.cfo_matching) return null;
    const manual = new Map(
      (config.cfo_mapping ?? []).map((item) => [
        text(item.fact_cfo_key || item.fact_cfo),
        text(item.plan_cfo_key || item.plan_cfo),
      ]),
    );
    const options = (config.cfo_matching.plan_cfos ?? [])
      .map((item) => ({ key: text(item.cfo_key), label: text(item.cfo) }))
      .filter((item) => item.key);
    const rows = (config.cfo_matching.rows ?? []).map((item) => {
      const key = text(item.cfo_key);
      const selectedKey = manual.get(key) ?? "";
      return {
        key,
        cfo: text(item.cfo, "не указано"),
        status: selectedKey ? "manual" : text(item.status, "matched"),
        selectedKey,
        plan: number(item.plan_amount),
        fact: number(item.fact_amount),
      };
    });
    return { options, rows, issues: rows.filter((item) => item.status !== "matched") };
  }, [config]);

  const articleMatching = useMemo(() => {
    if (!config?.article_matching) return null;
    const manual = new Map(
      (config.article_mapping ?? []).map((item) => [
        text(item.fact_article_key || item.fact_article),
        text(item.plan_article_key || item.plan_article),
      ]),
    );
    const options = (config.article_matching.plan_articles ?? [])
      .map((item) => ({
        key: text(item.article_key),
        label: text(item.article),
        cfo: text(item.cfo),
      }))
      .filter((item) => item.key);
    const rows = (config.article_matching.matches ?? []).map((item) => {
      const key = text(item.fact_article_key);
      const selectedKey = manual.get(key) ?? text(item.plan_article_key);
      const originalType = text(item.match_type, "unmatched");
      return {
        key,
        factArticle: text(item.fact_article, "не указано"),
        planArticle: text(item.plan_article),
        selectedKey,
        matchType: manual.has(key) ? "manual" : originalType,
        originalType,
        confidence: Math.round(number(item.confidence) * 100),
        amount: number(item.fact_amount),
        cfo: text(item.cfo),
      };
    });
    const filtered = rows.filter((item) => {
      if (matchFilter === "all") return true;
      if (matchFilter === "unmatched") return item.matchType === "unmatched";
      if (matchFilter === "suggested") return item.matchType === "fuzzy_suggested";
      if (matchFilter === "confirmed") return item.matchType === "manual";
      return ["fuzzy_suggested", "unmatched", "manual"].includes(item.matchType);
    });
    return {
      options,
      rows,
      filtered: filtered.sort((left, right) => Math.abs(right.amount) - Math.abs(left.amount)),
      stats: config.article_matching.stats ?? {},
    };
  }, [config, matchFilter]);

  function updateConfig(update: (next: PlanfactConfig) => void): void {
    const current = parseConfig(configText);
    if (!current) return;
    update(current);
    setConfigText(JSON.stringify(current, null, 2));
  }

  function updateField(section: "plan" | "fact", field: string, value: string): void {
    updateConfig((next) => {
      next[section] = { ...(next[section] ?? {}), [field]: value || null };
    });
  }

  function updateCfoMapping(factCfo: string, factKey: string, planKey: string): void {
    updateConfig((next) => {
      const retained = (next.cfo_mapping ?? []).filter(
        (item) => text(item.fact_cfo_key || item.fact_cfo) !== factKey,
      );
      const selected = cfoMatching?.options.find((item) => item.key === planKey);
      next.cfo_mapping = selected
        ? [
            ...retained,
            {
              fact_cfo: factCfo,
              fact_cfo_key: factKey,
              plan_cfo: selected.label,
              plan_cfo_key: selected.key,
            },
          ]
        : retained;
    });
  }

  function updateArticleMapping(
    factArticle: string,
    factKey: string,
    planKey: string,
  ): void {
    updateConfig((next) => {
      const retained = (next.article_mapping ?? []).filter(
        (item) => text(item.fact_article_key || item.fact_article) !== factKey,
      );
      const selected = articleMatching?.options.find((item) => item.key === planKey);
      next.article_mapping = selected
        ? [
            ...retained,
            {
              fact_article: factArticle,
              fact_article_key: factKey,
              plan_article: selected.label,
              plan_article_key: selected.key,
            },
          ]
        : retained;
    });
  }

  function acceptSuggestedArticles(): void {
    if (!articleMatching) return;
    updateConfig((next) => {
      const mappings = new Map(
        (next.article_mapping ?? []).map((item) => [
          text(item.fact_article_key || item.fact_article),
          item,
        ]),
      );
      for (const row of articleMatching.rows) {
        if (row.originalType !== "fuzzy_suggested" || !row.selectedKey) continue;
        const selected = articleMatching.options.find((item) => item.key === row.selectedKey);
        if (!selected) continue;
        mappings.set(row.key, {
          fact_article: row.factArticle,
          fact_article_key: row.key,
          plan_article: selected.label,
          plan_article_key: selected.key,
        });
      }
      next.article_mapping = [...mappings.values()];
    });
  }

  async function detect(): Promise<void> {
    if (!planFile || !factFile || disabled) return;
    setBusy(true);
    setError(null);
    try {
      const result = await detectPlanfactSource(sessionId, planFile, factFile, mappingFile);
      setDetection(result);
      setConfigText(JSON.stringify(result.suggested_config, null, 2));
    } catch (reason) {
      setError(summarizeError(reason));
    } finally {
      setBusy(false);
    }
  }

  async function confirm(): Promise<void> {
    if (disabled || !config) return;
    setBusy(true);
    setError(null);
    try {
      await confirmPlanfactSource(sessionId, config);
      await onConfirmed();
    } catch (reason) {
      setError(summarizeError(reason));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="rounded-[28px] border border-border/50 bg-card/45 p-6 shadow-sm">
      <div className="text-[12px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
        Plan-fact
      </div>
      <h3 className="mt-2 text-xl font-bold tracking-tight">План-факт</h3>
      <p className="mt-2 text-sm text-muted-foreground">
        Загрузите план и факт, проверьте найденные поля и подтвердите спорные сопоставления.
      </p>

      <div className="mt-4 grid gap-3 md:grid-cols-3">
        {[
          ["План", planFile, setPlanFile],
          ["Факт", factFile, setFactFile],
          ["Мэппинг кодов (необязательно)", mappingFile, setMappingFile],
        ].map(([label, file, setter]) => (
          <label
            key={String(label)}
            className="rounded-xl border border-dashed border-border/60 p-3 text-sm font-bold"
          >
            {String(label)}
            <input
              className="mt-2 block w-full text-xs"
              type="file"
              accept=".csv,.xlsx"
              disabled={busy || disabled}
              onChange={(event) =>
                (setter as (value: File | null) => void)(event.target.files?.[0] ?? null)
              }
            />
            {file instanceof File ? (
              <span className="mt-1 block truncate text-xs font-normal text-muted-foreground">
                {file.name}
              </span>
            ) : null}
          </label>
        ))}
      </div>
      <button
        type="button"
        onClick={() => void detect()}
        disabled={!planFile || !factFile || busy || disabled}
        className="mt-4 inline-flex items-center gap-2 rounded-xl bg-sky-500 px-4 py-2.5 text-sm font-bold text-sky-950 disabled:opacity-50"
      >
        {busy ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
        Распознать
      </button>

      {detection && config ? (
        <div className="mt-5 space-y-5">
          {detection.warnings.length ? (
            <p className="rounded-xl border border-amber-500/25 bg-amber-500/10 p-3 text-xs text-amber-700 dark:text-amber-300">
              {detection.warnings.join(" ")}
            </p>
          ) : (
            <p className="flex items-center gap-2 text-sm font-semibold text-emerald-700 dark:text-emerald-300">
              <CheckCircle2 className="h-4 w-4" /> Все обязательные поля распознаны
            </p>
          )}

          <div className="rounded-2xl border border-border/50 bg-secondary/20 p-4">
            <div className="text-sm font-bold">Проверка полей</div>
            <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <FieldSelect label="План · ЦФО" value={config.plan?.cfo_column} columns={detection.plan.columns} onChange={(value) => updateField("plan", "cfo_column", value)} />
              <FieldSelect label="План · статья" value={config.plan?.article_column} columns={detection.plan.columns} onChange={(value) => updateField("plan", "article_column", value)} />
              <FieldSelect label="План · доп. ключ" value={config.plan?.extra_key_column} columns={detection.plan.columns} onChange={(value) => updateField("plan", "extra_key_column", value)} optional />
              <FieldSelect label="Факт · дата" value={config.fact?.date_column} columns={detection.fact.columns} onChange={(value) => updateField("fact", "date_column", value)} />
              <FieldSelect label="Факт · ЦФО" value={config.fact?.cfo_column} columns={detection.fact.columns} onChange={(value) => updateField("fact", "cfo_column", value)} />
              <FieldSelect label="Факт · статья" value={config.fact?.article_column} columns={detection.fact.columns} onChange={(value) => updateField("fact", "article_column", value)} />
              <FieldSelect label="Факт · сумма" value={config.fact?.amount_column} columns={detection.fact.columns} onChange={(value) => updateField("fact", "amount_column", value)} />
              <FieldSelect label="Факт · содержание услуги" value={config.fact?.service_content_column} columns={detection.fact.columns} onChange={(value) => updateField("fact", "service_content_column", value)} optional />
            </div>
          </div>

          {cfoMatching ? (
            <div className="rounded-2xl border border-border/50 bg-secondary/20 p-4">
              <div className="text-sm font-bold">Сопоставление ЦФО</div>
              <p className="mt-1 text-xs text-muted-foreground">
                Для ЦФО только из факта выберите соответствующий ЦФО плана.
              </p>
              <div className="mt-3 max-h-72 space-y-2 overflow-y-auto">
                {cfoMatching.issues.length ? cfoMatching.issues.map((row) => (
                  <div key={row.key} className="grid gap-2 rounded-xl border border-border/50 bg-card/60 p-3 md:grid-cols-[1fr_1fr_100px_100px] md:items-center">
                    <div className="text-sm font-semibold">{row.cfo}</div>
                    {row.status === "fact_only" || row.status === "manual" ? (
                      <select value={row.selectedKey} onChange={(event) => updateCfoMapping(row.cfo, row.key, event.target.value)} className="rounded-lg border border-border/60 bg-background px-2 py-1.5 text-xs">
                        <option value="">Не сопоставлять</option>
                        {cfoMatching.options.map((option) => <option key={option.key} value={option.key}>{option.label}</option>)}
                      </select>
                    ) : <span className="text-xs text-muted-foreground">Только в плане</span>}
                    <div className="text-xs">План {formatMoney(row.plan)}</div>
                    <div className="text-xs">Факт {formatMoney(row.fact)}</div>
                  </div>
                )) : <p className="text-sm text-muted-foreground">Все ЦФО совпали.</p>}
              </div>
            </div>
          ) : null}

          {articleMatching ? (
            <div className="rounded-2xl border border-border/50 bg-secondary/20 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="text-sm font-bold">Сопоставление статей</div>
                  <p className="mt-1 text-xs text-muted-foreground">Сначала проверьте крупные суммы без пары и неуверенные совпадения.</p>
                </div>
                <button type="button" onClick={acceptSuggestedArticles} className="rounded-xl border border-emerald-500/25 bg-emerald-500/10 px-3 py-1.5 text-xs font-semibold text-emerald-700 dark:text-emerald-300">Принять предложенные</button>
              </div>
              <div className="mt-3 grid gap-2 sm:grid-cols-4">
                {[
                  ["Автоматически", articleMatching.stats.auto_matched],
                  ["На подтверждение", articleMatching.stats.needs_confirmation],
                  ["Не сопоставлено", articleMatching.stats.unmatched],
                  ["Всего", articleMatching.stats.total],
                ].map(([label, value]) => <div key={String(label)} className="rounded-xl border border-border/50 bg-card/60 p-3"><div className="text-[10px] font-bold uppercase text-muted-foreground">{String(label)}</div><div className="mt-1 text-lg font-bold">{number(value).toLocaleString("ru-RU")}</div></div>)}
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {(["review", "unmatched", "suggested", "confirmed", "all"] as MatchFilter[]).map((value) => (
                  <button key={value} type="button" onClick={() => setMatchFilter(value)} className={`rounded-lg border px-2.5 py-1 text-xs font-semibold ${matchFilter === value ? "border-primary/40 bg-primary/10 text-primary" : "border-border/60 bg-background"}`}>
                    {{ review: "Требуют проверки", unmatched: "Без пары", suggested: "Предложена пара", confirmed: "Подтвержденные", all: "Все" }[value]}
                  </button>
                ))}
              </div>
              <div className="mt-3 max-h-96 space-y-2 overflow-y-auto">
                {articleMatching.filtered.slice(0, 100).map((row) => (
                  <div key={`${row.key}-${row.cfo}`} className="grid gap-2 rounded-xl border border-border/50 bg-card/60 p-3 md:grid-cols-[1.2fr_1.2fr_90px_100px] md:items-center">
                    <div className="min-w-0"><div className="truncate text-sm font-semibold" title={row.factArticle}>{row.factArticle}</div><div className="truncate text-[11px] text-muted-foreground">{row.cfo}</div></div>
                    <select value={row.selectedKey} onChange={(event) => updateArticleMapping(row.factArticle, row.key, event.target.value)} className="min-w-0 rounded-lg border border-border/60 bg-background px-2 py-1.5 text-xs">
                      <option value="">Не сопоставлять</option>
                      {articleMatching.options.map((option) => <option key={`${option.key}-${option.cfo}`} value={option.key}>{option.label}{option.cfo ? ` · ${option.cfo}` : ""}</option>)}
                    </select>
                    <div className="text-xs font-semibold">{formatMoney(row.amount)}</div>
                    <div className="text-[11px] text-muted-foreground">{matchLabel(row.matchType)} · {row.confidence}%</div>
                  </div>
                ))}
                {!articleMatching.filtered.length ? <p className="text-sm text-muted-foreground">По выбранному фильтру строк нет.</p> : null}
              </div>
            </div>
          ) : null}

          <button type="button" onClick={() => void confirm()} disabled={!config || busy || disabled} className="inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-4 py-2.5 text-sm font-bold text-emerald-950 disabled:opacity-50">
            {busy ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
            Подтвердить и создать обзор
          </button>
        </div>
      ) : null}
      {error ? <p className="mt-3 text-sm text-destructive">{error}</p> : null}
    </section>
  );
}
