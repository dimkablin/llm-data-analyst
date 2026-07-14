import type { ContextUsageSnapshot, ContextUsageStatus } from "./backend-types";

export type ContextUsageTone =
  | "unavailable"
  | "normal"
  | "warning"
  | "critical"
  | "overflow";

export type ContextUsageTooltipDetails = {
  title: string;
  percentLine: string;
  usedLine: string;
};

const MIN_PERCENT = 0;
const MAX_PERCENT = 100;
const TONE_CLASS: Record<ContextUsageTone, string> = {
  unavailable: "text-muted-foreground",
  normal: "text-emerald-500",
  warning: "text-amber-500",
  critical: "text-orange-500",
  overflow: "text-rose-500",
};

export function getContextUsagePercent(usage: ContextUsageSnapshot | null): number {
  if (!usage || usage.usage_percent === null || usage.usage_percent === undefined) {
    return MIN_PERCENT;
  }
  if (!Number.isFinite(usage.usage_percent)) {
    return MIN_PERCENT;
  }
  return Math.min(MAX_PERCENT, Math.max(MIN_PERCENT, Math.round(usage.usage_percent)));
}

export function getContextUsageTone(usage: ContextUsageSnapshot | null): ContextUsageTone {
  if (!usage) {
    return "unavailable";
  }
  const status = usage.status as ContextUsageStatus;
  if (
    status === "normal" ||
    status === "warning" ||
    status === "critical" ||
    status === "overflow"
  ) {
    return status;
  }
  return "unavailable";
}

export function buildContextUsageTooltip(
  usage: ContextUsageSnapshot | null,
  isLoading: boolean,
): string {
  if (isLoading) {
    return "Считаю заполненность контекста.";
  }
  if (!usage) {
    return "Заполненность контекста пока неизвестна.";
  }
  const percent = getContextUsagePercent(usage);
  if (usage.max_context_tokens) {
    return `Контекст заполнен на ${percent}%. Использовано ${usage.used_tokens} из ${usage.max_context_tokens} токенов.`;
  }
  return `Контекст заполнен на ${percent}%. Максимум контекста пока неизвестен.`;
}

export function buildContextUsageTooltipDetails(
  usage: ContextUsageSnapshot | null,
  isLoading: boolean,
): ContextUsageTooltipDetails {
  if (isLoading) {
    return {
      title: "Контекст сессии",
      percentLine: "Считаю заполненность",
      usedLine: "Использовано пока неизвестно",
    };
  }
  if (!usage || usage.usage_percent === null || usage.usage_percent === undefined) {
    return {
      title: "Контекст сессии",
      percentLine: "Заполненность неизвестна",
      usedLine: "Использовано пока неизвестно",
    };
  }

  const percent = getContextUsagePercent(usage);
  const maxTokens = usage.max_context_tokens ?? 0;

  return {
    title: "Контекст сессии",
    percentLine: `${percent}% заполнено`,
    usedLine: maxTokens
      ? `Использовано ${formatContextUsageTokenCompact(usage.used_tokens)} / ${formatContextUsageTokenCompact(maxTokens)} tokens`
      : `Использовано ${formatContextUsageTokenCompact(usage.used_tokens)} tokens`,
  };
}

export function formatContextUsageLabel(usage: ContextUsageSnapshot | null): string {
  if (!usage || usage.usage_percent === null || usage.usage_percent === undefined || !usage.max_context_tokens) {
    return "Контекст: нет данных";
  }
  return `Контекст: ${getContextUsagePercent(usage)}% · ${formatTokens(usage.used_tokens)} из ${formatTokens(usage.max_context_tokens)} ток.`;
}

export function selectContextUsageLevelClass(tone: ContextUsageTone): string {
  return TONE_CLASS[tone] ?? TONE_CLASS.unavailable;
}

export function contextUsageStrokeColor(_tone: ContextUsageTone): string {
  return "currentColor";
}

export function formatContextUsageTokenCompact(value: number): string {
  const normalized = Math.max(0, Math.round(value || 0));
  if (normalized < 1000) {
    return String(normalized);
  }
  const thousands = Math.round((normalized / 1000) * 10) / 10;
  const text = Number.isInteger(thousands) ? thousands.toFixed(0) : thousands.toFixed(1);
  return `${text.replace(".", ",")}k`;
}

function formatTokens(value: number): string {
  const normalized = Math.max(0, Math.round(value || 0));
  return String(normalized).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
}
