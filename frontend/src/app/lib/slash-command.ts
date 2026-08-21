import type { ToolAvailability } from "./backend-types";

export type SlashCommand = {
  type: "skill" | "tool";
  id: string;
  label: string;
  description: string;
};

export function parseSlashInput(value: string): {
  commandId: string | null;
  query: string;
} {
  const match = value.match(/^\/([^\s]*)\s*(.*)$/s);
  if (!match) {
    return { commandId: null, query: value.trim() };
  }
  return {
    commandId: match[1]?.trim().toLowerCase() ?? "",
    query: match[2]?.trim() ?? "",
  };
}

export function matchesSlashCommand(command: SlashCommand, search: string): boolean {
  const needle = search.trim().toLowerCase();
  return !needle || [command.id, command.label, command.description].some(
    (value) => value.toLowerCase().includes(needle),
  );
}

export function isToolSlashAvailable(
  tool: ToolAvailability,
  hasSessionData: boolean,
): boolean {
  return tool.effective_enabled
    && tool.available_globally
    && (!tool.requires_session_data || hasSessionData);
}
