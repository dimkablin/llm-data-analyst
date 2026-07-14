import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { ANALYSIS_DEPTH_STEP_CEILING } from "../frontend/src/app/lib/backend-types.ts";
import {
  DEFAULT_USER_SETTINGS,
  normalizeUserSettings,
  toUserSettingsPatchPayload,
} from "../frontend/src/app/lib/default-settings.ts";

test("frontend fallback user settings match backend defaults", () => {
  assert.equal(DEFAULT_USER_SETTINGS.llm_temperature_chat, 0.7);
  assert.equal(DEFAULT_USER_SETTINGS.llm_temperature_tool, 0.5);
  assert.equal(DEFAULT_USER_SETTINGS.llm_max_tokens_default, 2048);
  assert.equal(DEFAULT_USER_SETTINGS.llm_max_tokens_reasoning, 4096);
  assert.equal(DEFAULT_USER_SETTINGS.backend_query_timeout_sec, 180);
  assert.equal(
    DEFAULT_USER_SETTINGS.agent_max_steps,
    ANALYSIS_DEPTH_STEP_CEILING[DEFAULT_USER_SETTINGS.analysis_depth],
  );
  assert.equal(DEFAULT_USER_SETTINGS.agent_inner_recursion_limit, 32);
});

test("frontend settings do not expose unsupported placeholders", () => {
  const sources = [
    "../frontend/src/app/components/workspace/SettingsPanel.tsx",
    "../frontend/src/app/pages/Account.tsx",
    "../frontend/src/app/lib/default-settings.ts",
    "../frontend/src/app/lib/backend-types.ts",
  ];
  const unsupportedSettings = [
    "agent_react_enabled",
    "default_answer_style",
  ];

  for (const source of sources) {
    const content = readFileSync(new URL(source, import.meta.url), "utf8");
    for (const identifier of unsupportedSettings) {
      assert.equal(
        content.includes(identifier),
        false,
        `${source} still exposes ${identifier}`,
      );
    }
  }

  const account = readFileSync(new URL("../frontend/src/app/pages/Account.tsx", import.meta.url), "utf8");
  for (const identifier of [
    "browserNotifications",
    "emailNotifications",
    "systemAlerts",
    "const [language",
    "setLanguage",
    "ToggleCard",
  ]) {
    assert.equal(
      account.includes(identifier),
      false,
      `Account.tsx still exposes ${identifier}`,
    );
  }
});

test("frontend settings normalize legacy backend fields out of state and payload", () => {
  const raw = {
    ...DEFAULT_USER_SETTINGS,
    theme: "light",
    llm_streaming: false,
    agent_react_enabled: true,
    default_answer_style: "concise",
  };

  const normalized = normalizeUserSettings(raw);
  assert.equal(normalized.theme, "light");
  assert.equal(normalized.llm_streaming, false);
  assert.equal(Object.hasOwn(normalized, "agent_react_enabled"), false);
  assert.equal(Object.hasOwn(normalized, "default_answer_style"), false);

  const payload = toUserSettingsPatchPayload(raw as Partial<typeof raw>);
  assert.equal(payload.llm_streaming, false);
  assert.equal(Object.hasOwn(payload, "agent_react_enabled"), false);
  assert.equal(Object.hasOwn(payload, "default_answer_style"), false);
});

test("workspace settings expose a single analysis depth control", () => {
  const panel = readFileSync(
    new URL("../frontend/src/app/components/workspace/SettingsPanel.tsx", import.meta.url),
    "utf8",
  );

  assert.equal(panel.includes("Режим анализа"), false);
  assert.equal(panel.includes("draft.analysis_mode"), false);
});
