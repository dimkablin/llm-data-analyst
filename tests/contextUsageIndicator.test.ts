import assert from "node:assert/strict";
import test from "node:test";

import type { ContextUsageSnapshot } from "../frontend/src/app/lib/backend-types.ts";
import {
  buildContextUsageTooltip,
  buildContextUsageTooltipDetails,
  formatContextUsageLabel,
  formatContextUsageTokenCompact,
  contextUsageStrokeColor,
  getContextUsagePercent,
  getContextUsageTone,
  selectContextUsageLevelClass,
} from "../frontend/src/app/lib/context-usage.ts";

const baseUsage: ContextUsageSnapshot = {
  input_tokens: 1200,
  reserved_response_tokens: 300,
  used_tokens: 1500,
  max_context_tokens: 2000,
  remaining_tokens: 500,
  usage_ratio: 0.75,
  usage_percent: 75,
  overflow: false,
  status: "warning",
  context_window_source: "settings",
};

test("clamps context usage percent for the ring indicator", () => {
  assert.equal(getContextUsagePercent(baseUsage), 75);
  assert.equal(getContextUsagePercent({ ...baseUsage, usage_percent: 3215 }), 100);
  assert.equal(getContextUsagePercent({ ...baseUsage, usage_percent: -10 }), 0);
  assert.equal(getContextUsagePercent(null), 0);
});

test("maps context usage statuses to stable visual tones", () => {
  assert.equal(getContextUsageTone({ ...baseUsage, status: "normal" }), "normal");
  assert.equal(getContextUsageTone({ ...baseUsage, status: "warning" }), "warning");
  assert.equal(getContextUsageTone({ ...baseUsage, status: "critical" }), "critical");
  assert.equal(getContextUsageTone({ ...baseUsage, status: "overflow" }), "overflow");
  assert.equal(getContextUsageTone(null), "unavailable");
});

test("builds Russian context usage tooltip from backend values", () => {
  assert.equal(
    buildContextUsageTooltip(baseUsage, false),
    "Контекст заполнен на 75%. Использовано 1500 из 2000 токенов.",
  );
  assert.equal(buildContextUsageTooltip(null, true), "Считаю заполненность контекста.");
  assert.equal(buildContextUsageTooltip(null, false), "Заполненность контекста пока неизвестна.");
});

test("builds compact tooltip details for the session context bubble", () => {
  assert.deepEqual(buildContextUsageTooltipDetails(baseUsage, false), {
    title: "Контекст сессии",
    percentLine: "75% заполнено",
    usedLine: "Использовано 1,5k / 2k tokens",
  });
  assert.equal(formatContextUsageTokenCompact(13100), "13,1k");
  assert.equal(formatContextUsageTokenCompact(32000), "32k");
});

test("formats compact Russian context usage label", () => {
  assert.equal(
    formatContextUsageLabel(baseUsage),
    "Контекст: 75% · 1 500 из 2 000 ток.",
  );
  assert.equal(formatContextUsageLabel(null), "Контекст: нет данных");
});

test("selects stable visual classes and stroke colors", () => {
  assert.match(selectContextUsageLevelClass("warning"), /amber/);
  assert.match(selectContextUsageLevelClass("critical"), /orange/);
  assert.match(selectContextUsageLevelClass("overflow"), /rose/);
  assert.equal(contextUsageStrokeColor("normal"), "currentColor");
});

test("chat panel contains compact live status labels", async () => {
  const fs = await import("node:fs/promises");
  const source = await fs.readFile(
    "frontend/src/app/components/workspace/ChatPanel.tsx",
    "utf8",
  );

  assert.match(source, /Сжимаю контекст/);
  assert.match(source, /Думаю/);
});
