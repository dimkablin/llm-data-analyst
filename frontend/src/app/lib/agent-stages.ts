import type { AssistantBlock, StreamToolCall, ToolUseBlock } from "./backend-types";

export type AgentStageId = 1 | 2 | 3 | 4 | 5;

/** User-facing statuses only — tool errors are folded into «done». */
export type AgentStageStatus = "pending" | "running" | "done";

export type AgentStageItem = {
  id: AgentStageId;
  title: string;
  subtitle: string;
  hint: string;
  status: AgentStageStatus;
  started_at?: number;
};

type StageDefinition = {
  id: AgentStageId;
  title: string;
  subtitle: string;
  hint: string;
  runningMessages: readonly string[];
};

export const AGENT_STAGE_DEFINITIONS: readonly StageDefinition[] = [
  {
    id: 1,
    title: "План",
    subtitle: "Выбираю инструменты",
    hint: "Сопоставляю ваш вопрос с данными и подходящими навыками",
    runningMessages: [
      "Разбираю формулировку вопроса…",
      "Выбираю лучший маршрут анализа…",
      "Сверяю доступные инструменты…",
    ],
  },
  {
    id: 2,
    title: "Данные",
    subtitle: "Получаю и готовлю таблицы",
    hint: "Загружаю строки, считаю агрегаты и проверяю качество цифр",
    runningMessages: [
      "Выполняю запросы к данным…",
      "Собираю нужные срезы и метрики…",
      "Сверяю значения между таблицами…",
    ],
  },
  {
    id: 3,
    title: "Графики",
    subtitle: "Строю визуализации",
    hint: "Подбираю тип диаграммы и оформление под ваш вопрос",
    runningMessages: [
      "Рисую распределения и сравнения…",
      "Выделяю ключевые сегменты на графике…",
      "Добавляю подписи и доли…",
    ],
  },
  {
    id: 4,
    title: "Анализ",
    subtitle: "Ищу закономерности",
    hint: "Связываю метрики, сравниваю периоды и проверяю гипотезы",
    runningMessages: [
      "Сопоставляю факты из таблиц и графиков…",
      "Ищу драйверы и отклонения…",
      "Проверяю, что выводы подтверждаются цифрами…",
    ],
  },
  {
    id: 5,
    title: "Итог",
    subtitle: "Суммаризирую ответ",
    hint: "Собираю понятный текст с главными цифрами и выводами",
    runningMessages: [
      "Формулирую ответ простым языком…",
      "Выделяю главное и убираю лишнее…",
      "Проверяю, что цифры сходятся с данными…",
    ],
  },
];

const STAGE_BY_ID = new Map(AGENT_STAGE_DEFINITIONS.map((def) => [def.id, def]));

/** Internal/meta tools folded into stage 1 — not shown as separate rows. */
export const HIDDEN_META_TOOLS = new Set(["get_tool_instructions"]);

const TOOL_STAGE_MAP: Record<string, AgentStageId> = {
  planner_tool: 1,
  get_tool_instructions: 1,
  sql_tool: 2,
  pandas_tool: 2,
  duckdb_tool: 2,
  python_tool: 2,
  file_tool: 2,
  csv_tool: 2,
  plotly_tool: 3,
  anomaly_tool: 4,
  forecast_tool: 4,
  search_tool: 4,
  rag_tool: 4,
  planfact_tool: 4,
};

export function mapToolNameToStage(toolName: string): AgentStageId {
  const normalized = String(toolName || "").trim().toLowerCase();
  if (TOOL_STAGE_MAP[normalized]) {
    return TOOL_STAGE_MAP[normalized];
  }
  if (/(sql|pandas|duckdb|data|table|query)/.test(normalized)) {
    return 2;
  }
  if (/(plot|chart|viz|visual)/.test(normalized)) {
    return 3;
  }
  if (/(plan|instruction|planner)/.test(normalized)) {
    return 1;
  }
  return 4;
}

export function pickStageRunningMessage(stage: AgentStageItem, nowMs = Date.now()): string {
  const def = STAGE_BY_ID.get(stage.id);
  const pool = def?.runningMessages ?? [];
  if (!pool.length) {
    return stage.hint;
  }
  const anchor = stage.started_at ?? nowMs;
  const index = Math.floor((nowMs - anchor) / 2800) % pool.length;
  return pool[index] ?? stage.hint;
}

function mergeStageStatus(current: AgentStageStatus, incoming: AgentStageStatus): AgentStageStatus {
  if (current === "running" || incoming === "running") {
    return "running";
  }
  if (current === "done" || incoming === "done") {
    return "done";
  }
  return incoming;
}

function toolStatusToStageStatus(toolStatus: StreamToolCall["status"]): AgentStageStatus {
  if (toolStatus === "running") {
    return "running";
  }
  return "done";
}

function applyToolToStages(
  stages: AgentStageItem[],
  tool: Pick<StreamToolCall, "tool_name" | "status" | "started_at">,
): void {
  const stageId = mapToolNameToStage(tool.tool_name);
  const index = stageId - 1;
  const item = stages[index];
  if (!item) {
    return;
  }

  const toolStatus = toolStatusToStageStatus(tool.status);

  item.status = mergeStageStatus(item.status, toolStatus);
  if (toolStatus === "running" && !item.started_at) {
    item.started_at = tool.started_at;
  }

  for (let idx = 0; idx < index; idx += 1) {
    if (stages[idx].status === "pending") {
      stages[idx].status = "done";
    }
  }
}

function createStageItems(): AgentStageItem[] {
  return AGENT_STAGE_DEFINITIONS.map((def) => ({
    id: def.id,
    title: def.title,
    subtitle: def.subtitle,
    hint: def.hint,
    status: "pending" as AgentStageStatus,
  }));
}

export function buildAgentStagesFromTools(
  tools: StreamToolCall[],
  options?: {
    isLive?: boolean;
    isSummarizing?: boolean;
  },
): AgentStageItem[] {
  const stages = createStageItems();
  const isLive = Boolean(options?.isLive);

  const visibleTools = tools.filter((tool) => !HIDDEN_META_TOOLS.has(tool.tool_name));
  for (const tool of visibleTools) {
    applyToolToStages(stages, tool);
  }

  const hasRunningTool = visibleTools.some((tool) => tool.status === "running");
  const hasAnyTool = visibleTools.length > 0;
  const summarizing = Boolean(options?.isSummarizing);

  if (isLive && !hasAnyTool && !summarizing) {
    stages[0].status = "running";
    stages[0].started_at = stages[0].started_at ?? Date.now();
  }

  if (summarizing) {
    stages[4].status = "running";
    stages[4].started_at = stages[4].started_at ?? Date.now();
    for (let idx = 0; idx < 4; idx += 1) {
      if (stages[idx].status === "pending") {
        stages[idx].status = "done";
      }
    }
  } else if (!isLive && hasAnyTool && !hasRunningTool) {
    stages[4].status = "done";
  }

  // If graphs are already built (stage 3 is done) but there are no explicit
  // “analysis tools” in this run, stage 4 would stay pending.
  // Make it feel “alive” by marking stage 4 as running until stage 5 starts.
  if (
    isLive &&
    stages[2].status === "done" &&
    stages[3].status === "pending" &&
    stages[4].status === "pending"
  ) {
    stages[3].status = "running";
    stages[3].started_at = stages[3].started_at ?? Date.now();
  }

  const showFullJourney = isLive || hasAnyTool || summarizing;
  if (showFullJourney) {
    return stages;
  }

  const highestActive = stages.reduce(
    (max, stage) => (stage.status !== "pending" ? Math.max(max, stage.id) : max),
    0,
  );
  return stages.filter((stage) => stage.id <= Math.max(highestActive, 1));
}

export function buildAgentStagesFromBlocks(
  blocks: AssistantBlock[],
  options?: { isLive?: boolean; isSummarizing?: boolean },
): AgentStageItem[] {
  const pseudoTools: StreamToolCall[] = [];

  for (const block of blocks) {
    if (block.type !== "tool_use") {
      continue;
    }
    const toolBlock = block as ToolUseBlock;
    if (HIDDEN_META_TOOLS.has(toolBlock.tool_name)) {
      continue;
    }
    pseudoTools.push({
      id: toolBlock.id,
      tool_name: toolBlock.tool_name,
      input_summary: toolBlock.input_summary,
      status: toolBlock.status,
      started_at: toolBlock.started_at,
    });
  }

  return buildAgentStagesFromTools(pseudoTools, options);
}
