export const SPINNER_STORAGE_KEY = "llm_v1_spinner";
export const SPINNER_CHANGED_EVENT = "llm-spinner-changed";

export type SpinnerId = "ring" | "dots" | "bars" | "pulse" | "scan";

export interface SpinnerOption {
  id: SpinnerId;
  label: string;
}

export const SPINNER_OPTIONS: SpinnerOption[] = [
  { id: "ring",  label: "Кольцо" },
  { id: "dots",  label: "Точки" },
  { id: "bars",  label: "Полосы" },
  { id: "pulse", label: "Пульс" },
  { id: "scan",  label: "Сканер" },
];

export const DEFAULT_SPINNER_ID: SpinnerId = "ring";

export function getStoredSpinner(): SpinnerId {
  if (typeof window === "undefined") return DEFAULT_SPINNER_ID;
  const value = window.localStorage.getItem(SPINNER_STORAGE_KEY) as SpinnerId | null;
  return (SPINNER_OPTIONS.some((o) => o.id === value) ? value : null) ?? DEFAULT_SPINNER_ID;
}

export function setStoredSpinner(id: SpinnerId): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(SPINNER_STORAGE_KEY, id);
  window.dispatchEvent(new Event(SPINNER_CHANGED_EVENT));
}
