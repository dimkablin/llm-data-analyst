export type PhoenixTraceHistoryStatus = "idle" | "loading" | "loaded" | "error" | "unavailable";

type PhoenixTraceHistorySummaryInput = {
  status: PhoenixTraceHistoryStatus;
  total: number;
  page: number;
  limit: number;
};

function formatCount(value: number): string {
  return new Intl.NumberFormat("ru-RU").format(value);
}

export function formatPhoenixTraceHistorySummary({
  status,
  total,
  page,
  limit,
}: PhoenixTraceHistorySummaryInput): string {
  if (status === "idle" || status === "loading") {
    return "Загрузка истории запросов...";
  }
  if (status === "unavailable") {
    return "Phoenix traces недоступны";
  }
  if (status === "error") {
    return "Не удалось загрузить историю запросов";
  }
  if (total <= 0) {
    return "Нет запросов за выбранный период";
  }

  const firstVisibleItem = page * limit + 1;
  const lastVisibleItem = Math.min((page + 1) * limit, total);
  return `Всего ${formatCount(total)} запросов • показаны ${formatCount(firstVisibleItem)}–${formatCount(lastVisibleItem)}`;
}

export function getPhoenixTraceEmptyMessage({
  status,
  hasSearch,
}: {
  status: PhoenixTraceHistoryStatus;
  hasSearch: boolean;
}): string | null {
  if (status === "idle" || status === "loading") {
    return null;
  }
  if (status === "unavailable") {
    return "Phoenix traces недоступны";
  }
  if (status === "error") {
    return "Не удалось загрузить историю запросов";
  }
  return hasSearch ? "Ничего не найдено" : "Нет данных";
}
