import type { AssistantBlock, PersistedReasoningStep, UserSettings } from "./backend-types";

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

/**
 * Filter an AssistantBlock[] by thinking visibility settings.
 * Non-thinking blocks pass through unchanged.
 * Thinking blocks without a kind default to "tool_synthesis".
 */
export function filterBlocks(
  blocks: AssistantBlock[],
  settings: ThinkVisibility,
): AssistantBlock[] {
  if (!settings.show_thinking) {
    return blocks.filter((b) => b.type !== "thinking");
  }
  return blocks.filter((b) => {
    if (b.type !== "thinking") return true;
    const kind = b.kind ?? "tool_synthesis";
    switch (kind) {
      case "planning":
        return settings.show_think_planning;
      case "tool_synthesis":
        return settings.show_think_tool;
      case "final_synthesis":
        return settings.show_think_final;
      default:
        return settings.show_think_tool;
    }
  });
}
