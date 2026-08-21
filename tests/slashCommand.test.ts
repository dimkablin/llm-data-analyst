import assert from "node:assert/strict";
import test from "node:test";

import {
  loadQueryExecutionOptions,
  saveQueryExecutionOptions,
} from '../frontend/src/app/lib/query-execution-options.ts';

import {
  isToolSlashAvailable,
  matchesSlashCommand,
  parseSlashInput,
} from "../frontend/src/app/lib/slash-command.ts";

test("slash parser removes the command token", () => {
  assert.deepEqual(parseSlashInput("/sql_tool покажи продажи"), {
    commandId: "sql_tool",
    query: "покажи продажи",
  });
  assert.deepEqual(parseSlashInput("обычный запрос"), {
    commandId: null,
    query: "обычный запрос",
  });
});

test("slash commands filter by id, label, and description", () => {
  const command = {
    type: "skill" as const,
    id: "planfact_variance_analysis",
    label: "План-факт",
    description: "Анализ отклонений",
  };
  assert.equal(matchesSlashCommand(command, "variance"), true);
  assert.equal(matchesSlashCommand(command, "план"), true);
  assert.equal(matchesSlashCommand(command, "отклон"), true);
  assert.equal(matchesSlashCommand(command, "sql"), false);
});

test("disabled and source-bound tools are unavailable", () => {
  const tool = {
    tool_key: "sql_tool",
    kind: "builtin" as const,
    tool_label: "SQL",
    display_name_ru: "SQL",
    description: "SQL",
    capabilities: [],
    requires_session_data: true,
    enabled_globally: true,
    available_globally: true,
    enabled_for_user: true,
    effective_enabled: true,
    status: "available",
  };
  assert.equal(isToolSlashAvailable(tool, false), false);
  assert.equal(isToolSlashAvailable(tool, true), true);
  assert.equal(isToolSlashAvailable({ ...tool, effective_enabled: false }, true), false);
});

test('retry execution options survive reload only for the matching query', () => {
  const values = new Map<string, string>();
  const storage = {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
  };

  saveQueryExecutionOptions(storage, 'session-1', 'show variance', {
    requestedToolKey: 'sql_tool',
  });

  assert.deepEqual(loadQueryExecutionOptions(storage, 'session-1', 'show variance'), {
    requestedToolKey: 'sql_tool',
  });
  assert.deepEqual(loadQueryExecutionOptions(storage, 'session-1', 'another query'), {});
});
