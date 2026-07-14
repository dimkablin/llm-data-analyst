import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("live tool display matches parallel tool_end events by tool_call_id", () => {
  const hook = readFileSync("frontend/src/app/hooks/useChatAgent.ts", "utf8");
  const types = readFileSync("frontend/src/app/lib/backend-types.ts", "utf8");
  const api = readFileSync("frontend/src/app/lib/backend-api.ts", "utf8");

  assert.match(api, /type ToolEvent = \{[\s\S]*tool_call_id\?: string;/);
  assert.match(types, /export type StreamToolCall = \{[\s\S]*tool_call_id\?: string;/);
  assert.match(types, /export type ToolUseBlock = \{[\s\S]*tool_call_id\?: string;/);
  assert.match(hook, /tool_call_id: event\.tool_call_id/);
  assert.match(hook, /event\.tool_call_id[\s\S]*tool_call_id === event\.tool_call_id/);
  assert.doesNotMatch(
    readFileSync("frontend/src/app/components/workspace/AgentActivityFeed.tsx", "utf8"),
    /Параллельно:/,
  );
  assert.doesNotMatch(
    readFileSync("frontend/src/app/components/workspace/blocks/BlockTimeline.tsx", "utf8"),
    /Параллельно:/,
  );
});
