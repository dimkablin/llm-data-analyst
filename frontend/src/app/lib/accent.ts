export const ACCENT_STORAGE_KEY = "llm_v1_accent";
export const ACCENT_CHANGED_EVENT = "llm-accent-changed";

export type AccentId = "default" | "blue" | "green" | "yellow" | "pink" | "orange" | "violet";

export function getStoredAccent(): AccentId {
  if (typeof window === "undefined") return "default";
  const value = window.localStorage.getItem(ACCENT_STORAGE_KEY) as AccentId | null;
  return value ?? "default";
}

export function applyAccent(accent: AccentId): void {
  if (typeof document === "undefined") return;
  document.documentElement.setAttribute("data-accent", accent);
}

export function setStoredAccent(accent: AccentId): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(ACCENT_STORAGE_KEY, accent);
  applyAccent(accent);
  window.dispatchEvent(new Event(ACCENT_CHANGED_EVENT));
}
