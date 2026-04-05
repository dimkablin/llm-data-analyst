export const SPINNER_STORAGE_KEY = "llm_v1_spinner";
export const SPINNER_CHANGED_EVENT = "llm-spinner-changed";

export type SpinnerId =
  | "blocks"
  | "dots"
  | "arc"
  | "arrows"
  | "moon"
  | "stars"
  | "cats";

export interface SpinnerOption {
  id: SpinnerId;
  label: string;
  frames: string[];
}

export const SPINNER_OPTIONS: SpinnerOption[] = [
  { id: "blocks", label: "Блоки", frames: ["▱▱▱", "▰▱▱", "▰▰▱", "▰▰▰", "▱▰▰", "▱▱▰"] },
  { id: "dots",   label: "Точки", frames: ["·", "•", "●", "•"] },
  { id: "arc",    label: "Дуга",  frames: ["◜", "◠", "◝", "◞", "◡", "◟"] },
  { id: "arrows", label: "Стрелки", frames: ["▹▹▹", "▸▹▹", "▹▸▹", "▹▹▸"] },
  { id: "moon",   label: "Луна",  frames: ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"] },
  { id: "stars",  label: "Звёзды", frames: ["✨", "⭐", "🌟", "💫"] },
  { id: "cats",   label: "Котики", frames: ["😺", "😸", "😹", "😺"] },
];

export const DEFAULT_SPINNER_ID: SpinnerId = "arc";

export function getStoredSpinner(): SpinnerId {
  if (typeof window === "undefined") return DEFAULT_SPINNER_ID;
  const value = window.localStorage.getItem(SPINNER_STORAGE_KEY) as SpinnerId | null;
  return value ?? DEFAULT_SPINNER_ID;
}

export function setStoredSpinner(id: SpinnerId): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(SPINNER_STORAGE_KEY, id);
  window.dispatchEvent(new Event(SPINNER_CHANGED_EVENT));
}

export function getSpinnerFrames(id: SpinnerId): string[] {
  return SPINNER_OPTIONS.find((o) => o.id === id)?.frames ?? SPINNER_OPTIONS[0]!.frames;
}
