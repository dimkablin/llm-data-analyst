import assert from "node:assert/strict";
import test from "node:test";

import {
  formatPhoenixTraceHistorySummary,
  getPhoenixTraceEmptyMessage,
} from "../frontend/src/app/lib/phoenix-trace-history.ts";

test("keeps the history summary in loading state before traces are loaded", () => {
  assert.equal(
    formatPhoenixTraceHistorySummary({
      status: "loading",
      total: 0,
      page: 0,
      limit: 15,
    }),
    "Загрузка истории запросов...",
  );
});

test("does not show loading after Phoenix returns an empty trace page", () => {
  assert.equal(
    formatPhoenixTraceHistorySummary({
      status: "loaded",
      total: 0,
      page: 0,
      limit: 15,
    }),
    "Нет запросов за выбранный период",
  );
});

test("formats the visible trace range after traces are loaded", () => {
  assert.equal(
    formatPhoenixTraceHistorySummary({
      status: "loaded",
      total: 35,
      page: 1,
      limit: 15,
    }),
    "Всего 35 запросов • показаны 16–30",
  );
});

test("shows explicit trace history errors instead of an empty-table fallback", () => {
  assert.equal(
    formatPhoenixTraceHistorySummary({
      status: "error",
      total: 0,
      page: 0,
      limit: 15,
    }),
    "Не удалось загрузить историю запросов",
  );
  assert.equal(
    getPhoenixTraceEmptyMessage({ status: "error", hasSearch: false }),
    "Не удалось загрузить историю запросов",
  );
});
