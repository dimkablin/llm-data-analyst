import assert from "node:assert/strict";
import test from "node:test";

import {
  INTERRUPTED_TOOL_OUTPUT,
  finalizeInterruptedAssistantState,
} from "../frontend/src/app/lib/stream-abort.ts";
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
