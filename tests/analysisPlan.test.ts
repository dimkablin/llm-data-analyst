import assert from "node:assert/strict";
import test from "node:test";

import {
  analysisPlanDisplayState,
  latestAnalysisPlan,
} from "../frontend/src/app/lib/analysis-plan.ts";
import type { AssistantBlock } from "../frontend/src/app/lib/backend-types.ts";

function planResult(
  id: string,
  plan: unknown,
  status: "ok" | "error" = "ok",
): AssistantBlock {
  return {
    type: "tool_result",
    id: `result-${id}`,
    tool_use_id: id,
    tool_name: "update_plan",
    status,
    result_summary: "",
    output_preview: typeof plan === "string" ? plan : JSON.stringify({ plan }),
  };
}

test("analysis plan display distinguishes running, completed, and stopped", () => {
  const incomplete = [
    { step: "Inspect data", status: "completed" as const },
    { step: "Calculate result", status: "in_progress" as const },
  ];
  const completed = incomplete.map((item) => ({ ...item, status: "completed" as const }));

  assert.equal(analysisPlanDisplayState(incomplete, true), "running");
  assert.equal(analysisPlanDisplayState(completed, false), "completed");
  assert.equal(analysisPlanDisplayState(incomplete, false), "stopped");
});

test("latest analysis plan ignores failed and malformed updates", () => {
  const first = [{ step: "Inspect data", status: "in_progress" as const }];
  const latest = [
    { step: "Inspect data", status: "completed" as const },
    { step: "Calculate result", status: "in_progress" as const },
  ];
  const blocks: AssistantBlock[] = [
    planResult("plan-1", first),
    planResult("plan-error", latest, "error"),
    planResult("plan-malformed", "not json"),
    planResult("plan-2", latest),
  ];

  const state = latestAnalysisPlan(blocks);

  assert.deepEqual(state.plan, latest);
  assert.deepEqual([...state.successfulToolUseIds], ["plan-1", "plan-2"]);
});
