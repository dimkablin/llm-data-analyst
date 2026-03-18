import { AUTH_CHANGED_EVENT, TOKEN_STORAGE_KEY } from "./backend-api";

export const AUTH_STORAGE_KEY = TOKEN_STORAGE_KEY;

export function isSignedIn(): boolean {
  if (typeof window === "undefined") return false;
  return Boolean(window.localStorage.getItem(AUTH_STORAGE_KEY));
}

export function setSignedInState(value: boolean): void {
  if (typeof window === "undefined") return;

  if (value) {
    window.localStorage.setItem(AUTH_STORAGE_KEY, "placeholder");
  } else {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
  }

  window.dispatchEvent(new Event(AUTH_CHANGED_EVENT));
}
