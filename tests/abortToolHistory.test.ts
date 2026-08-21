import assert from "node:assert/strict";
import test from "node:test";

import {
  INTERRUPTED_TOOL_OUTPUT,
  finalizeInterruptedAssistantState,
} from "../frontend/src/app/lib/stream-abort.ts";
import {
  analysisPlanDisplayState,
  latestAnalysisPlan,
} from "../frontend/src/app/lib/analysis-plan.ts";
import type { AssistantBlock, StreamToolCall } from "../frontend/src/app/lib/backend-types.ts";

test("finalizes tool-only aborted stream without dropping tool history", () => {
  const tools: StreamToolCall[] = [
    {
      id: "tool-running",
      tool_name: "sql_tool",
      input_summary: "SELECT 1",
      status: "running",
      started_at: 100,
    },
    {
      id: "tool-done",
      tool_name: "pandas_tool",
      input_summary: "df.head()",
      output_preview: "rows: 2",
      status: "done",
      started_at: 200,
    },
  ];
  const blocks: AssistantBlock[] = [
    {
      type: "tool_use",
      id: "block-running",
      tool_name: "sql_tool",
      input_summary: "SELECT 1",
      status: "running",
      started_at: 100,
    },
    {
      type: "tool_use",
      id: "block-done",
      tool_name: "pandas_tool",
      input_summary: "df.head()",
      status: "done",
      started_at: 200,
    },
    {
      type: "tool_result",
      id: "result-done",
      tool_use_id: "block-done",
      tool_name: "pandas_tool",
      status: "ok",
      result_summary: "2 rows",
      output_preview: "rows: 2",
    },
  ];

  const result = finalizeInterruptedAssistantState({
    text: "",
    reasoning: null,
    phases: [],
    tools,
    blocks,
  });

  assert.equal(result.shouldAppend, true);
  assert.equal(result.tools?.[0]?.status, "error");
  assert.equal(result.tools?.[0]?.output_preview, INTERRUPTED_TOOL_OUTPUT);
  assert.equal(result.tools?.[1]?.output_preview, "rows: 2");

  const interruptedUse = result.blocks?.find((block) => block.id === "block-running");
  assert.equal(interruptedUse?.type, "tool_use");
  assert.equal(interruptedUse?.status, "error");

  const interruptedResult = result.blocks?.find(
    (block) => block.type === "tool_result" && block.tool_use_id === "block-running",
  );
  assert.equal(interruptedResult?.type, "tool_result");
  assert.equal(interruptedResult?.status, "error");
  assert.equal(interruptedResult?.output_preview, INTERRUPTED_TOOL_OUTPUT);

  const completedResult = result.blocks?.find((block) => block.id === "result-done");
  assert.equal(completedResult?.type, "tool_result");
  assert.equal(completedResult?.output_preview, "rows: 2");
});

test("interrupted plan update keeps the last successful plan as stopped", () => {
  const plan = [{ step: "Inspect data", status: "in_progress" as const }];
  const output = JSON.stringify({ plan, completed: 0, total: 1 });
  const tools: StreamToolCall[] = [
    {
      id: "plan-done",
      tool_call_id: "plan-call-1",
      tool_name: "update_plan",
      input_summary: "update plan",
      output_preview: output,
      status: "done",
      started_at: 100,
    },
    {
      id: "plan-running",
      tool_call_id: "plan-call-2",
      tool_name: "update_plan",
      input_summary: "update plan",
      status: "running",
      started_at: 200,
    },
  ];
  const blocks: AssistantBlock[] = [
    {
      type: "tool_use",
      id: "plan-use-1",
      tool_name: "update_plan",
      input_summary: "update plan",
      status: "done",
      started_at: 100,
    },
    {
      type: "tool_result",
      id: "plan-result-1",
      tool_use_id: "plan-use-1",
      tool_name: "update_plan",
      status: "ok",
      result_summary: "",
      output_preview: output,
    },
    {
      type: "tool_use",
      id: "plan-use-2",
      tool_name: "update_plan",
      input_summary: "update plan",
      status: "running",
      started_at: 200,
    },
  ];

  const result = finalizeInterruptedAssistantState({
    text: "",
    reasoning: null,
    phases: [],
    tools,
    blocks,
  });
  const state = latestAnalysisPlan(result.blocks ?? []);

  assert.deepEqual(state.plan, plan);
  assert.equal(analysisPlanDisplayState(state.plan!, false), "stopped");
  assert.equal(
    result.blocks?.find(
      (block) => block.type === "tool_result" && block.tool_use_id === "plan-use-2",
    )?.status,
    "error",
  );
});
