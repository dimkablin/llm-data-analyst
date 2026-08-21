import type { AssistantBlock, ToolResultBlock } from "./backend-types";

export type AnalysisPlanItem = {
  step: string;
  status: "pending" | "in_progress" | "completed";
};

export type AnalysisPlanDisplayState = "running" | "completed" | "stopped";

function parseAnalysisPlan(block: ToolResultBlock): AnalysisPlanItem[] | null {
  if (block.tool_name !== "update_plan" || block.status !== "ok" || !block.output_preview) {
    return null;
  }
  try {
    const payload = JSON.parse(block.output_preview) as { plan?: unknown };
    if (!Array.isArray(payload.plan) || payload.plan.length === 0) return null;
    const plan = payload.plan.filter(
      (item): item is AnalysisPlanItem =>
        typeof item === "object" &&
        item !== null &&
        typeof (item as AnalysisPlanItem).step === "string" &&
        ["pending", "in_progress", "completed"].includes((item as AnalysisPlanItem).status),
    );
    return plan.length === payload.plan.length ? plan : null;
  } catch {
    return null;
  }
}

export function latestAnalysisPlan(blocks: AssistantBlock[]): {
  plan: AnalysisPlanItem[] | null;
  successfulToolUseIds: Set<string>;
} {
  const successfulToolUseIds = new Set<string>();
  let plan: AnalysisPlanItem[] | null = null;
  for (const block of blocks) {
    if (block.type !== "tool_result") continue;
    const parsed = parseAnalysisPlan(block);
    if (!parsed) continue;
    successfulToolUseIds.add(block.tool_use_id);
    plan = parsed;
  }
  return { plan, successfulToolUseIds };
}

export function analysisPlanDisplayState(
  plan: AnalysisPlanItem[],
  isLive: boolean,
): AnalysisPlanDisplayState {
  if (plan.every((item) => item.status === "completed")) return "completed";
  return isLive ? "running" : "stopped";
}
