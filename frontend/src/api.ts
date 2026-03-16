import type {
  AuthResult,
  AuthUser,
  PhaseEvent,
  QueryResponse,
  RuntimeModelProfile,
  SessionState,
  SessionSummary,
  UserSettings
} from "./types";
import { appBasePath } from "./basePath";

const API_BASE = appBasePath === "/" ? "" : appBasePath.slice(0, -1);
const TOKEN_STORAGE_KEY = "llm_data_analyst_access_token";

async function parseJsonSafe(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

async function assertOk(response: Response): Promise<void> {
  if (response.ok) {
    return;
  }
  const body = await parseJsonSafe(response);
  throw new Error(`HTTP ${response.status}: ${JSON.stringify(body)}`);
}

export function getStoredToken(): string | null {
  return window.localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setStoredToken(token: string): void {
  window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
}

export function clearStoredToken(): void {
  window.localStorage.removeItem(TOKEN_STORAGE_KEY);
}

async function authFetch(path: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers ?? {});
  const token = getStoredToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers
  });
  return response;
}

export async function registerUser(username: string, password: string): Promise<AuthResult> {
  const response = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password })
  });
  await assertOk(response);
  return (await response.json()) as AuthResult;
}

export async function loginUser(username: string, password: string): Promise<AuthResult> {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password })
  });
  await assertOk(response);
  return (await response.json()) as AuthResult;
}

export async function getCurrentUser(): Promise<AuthUser> {
  const response = await authFetch("/auth/me", { method: "GET" });
  await assertOk(response);
  return (await response.json()) as AuthUser;
}

export async function logoutUser(): Promise<void> {
  const response = await authFetch("/auth/logout", { method: "POST" });
  await assertOk(response);
}

export async function changePassword(
  currentPassword: string,
  newPassword: string
): Promise<void> {
  const response = await authFetch("/auth/change-password", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword
    })
  });
  await assertOk(response);
}

export async function getUserSettings(): Promise<UserSettings> {
  const response = await authFetch("/auth/settings", { method: "GET" });
  await assertOk(response);
  return (await response.json()) as UserSettings;
}

export async function updateUserSettings(payload: Partial<UserSettings>): Promise<UserSettings> {
  const response = await authFetch("/auth/settings", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  await assertOk(response);
  return (await response.json()) as UserSettings;
}

export async function listAdminUsers(): Promise<AuthUser[]> {
  const response = await authFetch("/admin/users", { method: "GET" });
  await assertOk(response);
  return (await response.json()) as AuthUser[];
}

export async function createAdminUser(
  username: string,
  password: string,
  isAdmin: boolean
): Promise<AuthUser> {
  const response = await authFetch("/admin/users", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, is_admin: isAdmin })
  });
  await assertOk(response);
  return (await response.json()) as AuthUser;
}

export async function updateAdminUser(
  userId: number,
  payload: { password?: string; is_admin?: boolean }
): Promise<AuthUser> {
  const response = await authFetch(`/admin/users/${userId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  await assertOk(response);
  return (await response.json()) as AuthUser;
}

export async function deleteAdminUser(userId: number): Promise<void> {
  const response = await authFetch(`/admin/users/${userId}`, { method: "DELETE" });
  await assertOk(response);
}

export async function listSessions(): Promise<SessionSummary[]> {
  const response = await authFetch("/sessions", { method: "GET" });
  await assertOk(response);
  return (await response.json()) as SessionSummary[];
}

export async function updateSessionTitle(sessionId: string, title: string): Promise<SessionSummary> {
  const response = await authFetch(`/sessions/${sessionId}/title`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title })
  });
  await assertOk(response);
  return (await response.json()) as SessionSummary;
}

export async function generateSessionTitle(sessionId: string): Promise<SessionSummary> {
  const response = await authFetch(`/sessions/${sessionId}/title/generate`, {
    method: "POST"
  });
  await assertOk(response);
  return (await response.json()) as SessionSummary;
}

export async function createSession(enableAutoTitle = false): Promise<string> {
  const query = enableAutoTitle ? "?enable_auto_title=1" : "";
  const response = await authFetch(`/sessions${query}`, { method: "POST" });
  await assertOk(response);
  const data = (await response.json()) as { session_id: string };
  return data.session_id;
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await authFetch(`/sessions/${sessionId}`, { method: "DELETE" });
  await assertOk(response);
}

export async function getSession(sessionId: string): Promise<SessionState> {
  const response = await authFetch(`/sessions/${sessionId}`);
  await assertOk(response);
  return (await response.json()) as SessionState;
}

export async function getRuntimeModelProfile(): Promise<RuntimeModelProfile> {
  const response = await authFetch("/runtime/model");
  await assertOk(response);
  return (await response.json()) as RuntimeModelProfile;
}

export async function uploadCsv(sessionId: string, file: File): Promise<void> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await authFetch(`/sessions/${sessionId}/data`, {
    method: "POST",
    body: formData
  });
  await assertOk(response);
}

type StreamHandlers = {
  onToken: (token: string) => void;
  onFinal: (payload: QueryResponse) => void;
  onReasoning: (payload: string, mode: "chunk" | "token") => void;
  onPhase?: (event: PhaseEvent) => void;
  onPhaseToken?: (token: string) => void;
  onError: (error: string) => void;
};

function consumeSseLine(
  line: string,
  state: { event: string | null },
  handlers: StreamHandlers
): void {
  if (line.startsWith("event: ")) {
    state.event = line.slice(7).trim();
    return;
  }
  if (!line.startsWith("data: ")) {
    return;
  }
  const payloadRaw = line.slice(6);
  let payload: unknown = payloadRaw;
  try {
    payload = JSON.parse(payloadRaw);
  } catch {
    payload = payloadRaw;
  }
  const currentEvent = state.event;
  state.event = null;

  if (currentEvent === "token" && typeof payload === "string") {
    handlers.onToken(payload);
    return;
  }
  if (currentEvent === "final" && typeof payload === "object" && payload !== null) {
    handlers.onFinal(payload as QueryResponse);
    return;
  }
  if (currentEvent === "reasoning" && typeof payload === "string") {
    handlers.onReasoning(payload, "chunk");
    return;
  }
  if (currentEvent === "reasoning_token" && typeof payload === "string") {
    handlers.onReasoning(payload, "token");
    return;
  }
  if (currentEvent === "phase" && typeof payload === "object" && payload !== null) {
    handlers.onPhase?.(payload as PhaseEvent);
    return;
  }
  if (currentEvent === "phase_token" && typeof payload === "string") {
    handlers.onPhaseToken?.(payload);
    return;
  }
  if (currentEvent === "error") {
    handlers.onError(typeof payload === "string" ? payload : JSON.stringify(payload));
  }
}

export async function streamQuery(
  sessionId: string,
  query: string,
  includeReasoning: boolean,
  useHistory: boolean,
  handlers: StreamHandlers,
  signal?: AbortSignal,
  analysisDepth?: string
): Promise<void> {
  const body: Record<string, unknown> = {
    query,
    use_history: useHistory,
    include_reasoning: includeReasoning,
  };
  if (analysisDepth) {
    body.analysis_depth = analysisDepth;
  }
  const response = await authFetch(`/sessions/${sessionId}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify(body)
  });
  await assertOk(response);

  if (!response.body) {
    throw new Error("SSE stream body is empty");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const state = { event: null as string | null };
  let carry = "";

  const processChunkLines = (chunk: string): void => {
    if (!chunk) {
      return;
    }
    const lines = chunk.split("\n");
    carry = lines.pop() ?? "";
    for (const rawLine of lines) {
      const line = rawLine.trimEnd();
      if (!line) {
        continue;
      }
      consumeSseLine(line, state, handlers);
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    carry += decoder.decode(value, { stream: true });
    processChunkLines(carry);
  }

  const tail = (carry + decoder.decode()).trim();
  if (tail) {
    for (const rawLine of tail.split("\n")) {
      const line = rawLine.trimEnd();
      if (!line) {
        continue;
      }
      consumeSseLine(line, state, handlers);
    }
  }
}
