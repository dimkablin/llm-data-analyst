import type { UserSettings } from "./backend-types";

export const DEFAULT_USER_SETTINGS: UserSettings = {
  theme: "dark",
  default_include_reasoning: true,
  analysis_mode: "fast",
  analysis_depth: "light",
  llm_temperature_chat: 0.7,
  llm_temperature_tool: 0.5,
  llm_max_tokens_default: 4096,
  llm_max_tokens_reasoning: 4096,
  backend_query_timeout_sec: 180,
  agent_max_steps: 32,
  agent_step_timeout_sec: 45,
  agent_inner_recursion_limit: 32,
  ui_scale: 100,
  llm_streaming: true,
  show_thinking: true,
  show_think_planning: true,
  show_think_tool: true,
  show_think_final: true,
  always_use_analysis_plan: false,
  show_detailed_tool_steps: false,
  show_rag_errors: true,
  anomaly_check_enabled: false,
};

export const USER_SETTING_KEYS = Object.keys(DEFAULT_USER_SETTINGS) as Array<keyof UserSettings>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function normalizeUserSettings(raw: unknown): UserSettings {
  const source = isRecord(raw) ? raw : {};
  const next: UserSettings = { ...DEFAULT_USER_SETTINGS };

  for (const key of USER_SETTING_KEYS) {
    if (Object.prototype.hasOwnProperty.call(source, key)) {
      (next as Record<string, unknown>)[key] = source[key];
    }
  }

  return next;
}

export function toUserSettingsPatchPayload(payload: Partial<UserSettings>): Partial<UserSettings> {
  const source = payload as Record<string, unknown>;
  const next: Partial<UserSettings> = {};

  for (const key of USER_SETTING_KEYS) {
    if (Object.prototype.hasOwnProperty.call(source, key)) {
      (next as Record<string, unknown>)[key] = source[key];
    }
  }

  return next;
}
