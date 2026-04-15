import type { PersistedReasoningStep, UserSettings } from "./backend-types";

type ThinkVisibility = Pick<
  UserSettings,
  "show_thinking" | "show_think_planning" | "show_think_tool" | "show_think_final"
>;

/**
 * Filter reasoning steps according to user visibility settings.
 * - If show_thinking is false, returns [].
 * - Otherwise filters by kind: planning / tool_synthesis / final_synthesis.
 * - "unknown" kind falls back to show_think_tool.
 */
export function filterReasoningSteps(
  steps: PersistedReasoningStep[] | null | undefined,
  settings: ThinkVisibility,
): PersistedReasoningStep[] {
  if (!steps || steps.length === 0) return [];
  if (!settings.show_thinking) return [];

  return steps.filter((step) => {
    switch (step.kind) {
      case "planning":
        return settings.show_think_planning;
      case "tool_synthesis":
        return settings.show_think_tool;
      case "final_synthesis":
        return settings.show_think_final;
      default:
        return settings.show_think_tool; // "unknown" → tool fallback
    }
  });
}
