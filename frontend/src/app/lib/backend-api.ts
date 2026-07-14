import type {
  AdminMCPServerConfig,
  AdminMCPServerPayload,
  AdminSkillDetail,
  AdminSkillUpdatePayload,
  ArtifactPayload,
  AuthResult,
  AuthUser,
  BatchUploadResponse,
  ContextUsageSnapshot,
  DBConnection,
  DBConnectionFormPayload,
  DBConnectionSchema,
  DBConnectionTestResult,
  ExecutionGraph,
  MCPServerAvailability,
  OpenProjectProjectsResponse,
  OpenProjectSyncRequest,
  OpenProjectSyncResponse,
  PhaseEvent,
  PhoenixOverview,
  PhoenixTraceDetail,
  PhoenixTracesResponse,
  QueryResponse,
  RagDocumentDeleteResponse,
  RagDocumentsResponse,
  RagTrackStatusResponse,
  RagUploadResponse,
  RuntimeModelProfile,
  SemanticCatalog,
  SemanticCatalogGenerationAcceptedResponse,
  SemanticCatalogGenerationRequest,
  SemanticCatalogGenerationResponse,
  SemanticCatalogStatusResponse,
  SemanticMetric,
  SemanticMetricPayload,
  SemanticRelationship,
  SemanticRelationshipPayload,
  SemanticTerm,
  SemanticTermPayload,
  SessionState,
  SessionSourceState,
  SessionSummary,
  Skill,
  TabularPreprocessingOptions,
  ToolAvailability,
  UserMemory,
  UserSettings,
} from "./backend-types";
import { normalizeUserSettings, toUserSettingsPatchPayload } from "./default-settings.ts";

export type BoardExportFormat = "docx" | "pdf" | "xlsx";

const VITE_ENV = (import.meta as unknown as { env?: Record<string, string | undefined> }).env ?? {};
const API_BASE =
  VITE_ENV.VITE_API_BASE?.replace(/\/$/, "") ??
  (VITE_ENV.BASE_URL ?? "/").replace(/\/$/, "");
export const TOKEN_STORAGE_KEY = "llm_data_analyst_access_token";
export const AUTH_CHANGED_EVENT = "llm-backend-auth-changed";

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

export function hasStoredToken(): boolean {
  return Boolean(getStoredToken());
}

export function setStoredToken(token: string): void {
  window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
  window.dispatchEvent(new Event(AUTH_CHANGED_EVENT));
}

export function clearStoredToken(): void {
  window.localStorage.removeItem(TOKEN_STORAGE_KEY);
  window.dispatchEvent(new Event(AUTH_CHANGED_EVENT));
}

async function authFetch(path: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers ?? {});
  const token = getStoredToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  return fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
  });
}

export async function registerUser(username: string, password: string): Promise<AuthResult> {
  const response = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  await assertOk(response);
  return (await response.json()) as AuthResult;
}

export async function loginUser(username: string, password: string): Promise<AuthResult> {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  await assertOk(response);
  return (await response.json()) as AuthResult;
}

export async function getCurrentUser(): Promise<AuthUser> {
  const response = await authFetch("/auth/me");
  await assertOk(response);
  return (await response.json()) as AuthUser;
}

export async function logoutUser(): Promise<void> {
  const response = await authFetch("/auth/logout", { method: "POST" });
  await assertOk(response);
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  const response = await authFetch("/auth/change-password", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword,
    }),
  });
  await assertOk(response);
}

export async function getUserSettings(): Promise<UserSettings> {
  const response = await authFetch("/auth/settings");
  await assertOk(response);
  return normalizeUserSettings(await response.json());
}

export async function updateUserSettings(payload: Partial<UserSettings>): Promise<UserSettings> {
  const response = await authFetch("/auth/settings", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(toUserSettingsPatchPayload(payload)),
  });
  await assertOk(response);
  return normalizeUserSettings(await response.json());
}

export async function getUserMemory(): Promise<UserMemory> {
  const response = await authFetch("/auth/memory");
  await assertOk(response);
  return (await response.json()) as UserMemory;
}

export async function updateUserMemory(payload: Partial<UserMemory>): Promise<UserMemory> {
  const response = await authFetch("/auth/memory", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await assertOk(response);
  return (await response.json()) as UserMemory;
}

export async function listSkills(): Promise<Skill[]> {
  const response = await authFetch("/skills");
  await assertOk(response);
  return (await response.json()) as Skill[];
}

export async function updateSkillEnabled(skillId: string, enabled: boolean): Promise<Skill> {
  const response = await authFetch(`/skills/${encodeURIComponent(skillId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  await assertOk(response);
  return (await response.json()) as Skill;
}

export async function getUserTools(): Promise<ToolAvailability[]> {
  const response = await authFetch("/auth/tools");
  await assertOk(response);
  return (await response.json()) as ToolAvailability[];
}

export async function updateUserToolEnabled(
  toolKey: string,
  enabled: boolean,
): Promise<ToolAvailability> {
  const response = await authFetch(`/auth/tools/${toolKey}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  await assertOk(response);
  return (await response.json()) as ToolAvailability;
}

export async function listMcpServers(): Promise<MCPServerAvailability[]> {
  const response = await authFetch("/mcp/servers");
  await assertOk(response);
  return (await response.json()) as MCPServerAvailability[];
}

export async function updateMcpServerEnabled(
  serverId: string,
  enabled: boolean,
): Promise<MCPServerAvailability> {
  const response = await authFetch(`/mcp/servers/${encodeURIComponent(serverId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  await assertOk(response);
  return (await response.json()) as MCPServerAvailability;
}

export async function listAdminUsers(): Promise<AuthUser[]> {
  const response = await authFetch("/admin/users");
  await assertOk(response);
  return (await response.json()) as AuthUser[];
}

export async function createAdminUser(
  username: string,
  password: string,
  isAdmin: boolean,
): Promise<AuthUser> {
  const response = await authFetch("/admin/users", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, is_admin: isAdmin }),
  });
  await assertOk(response);
  return (await response.json()) as AuthUser;
}

export async function updateAdminUser(
  userId: number,
  payload: { password?: string; is_admin?: boolean },
): Promise<AuthUser> {
  const response = await authFetch(`/admin/users/${userId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await assertOk(response);
  return (await response.json()) as AuthUser;
}

export async function deleteAdminUser(userId: number): Promise<void> {
  const response = await authFetch(`/admin/users/${userId}`, { method: "DELETE" });
  await assertOk(response);
}

// ── Admin skills ────────────────────────────────────────────────────────────

export async function listAdminMcpServers(): Promise<AdminMCPServerConfig[]> {
  const response = await authFetch("/admin/mcp/servers");
  await assertOk(response);
  return (await response.json()) as AdminMCPServerConfig[];
}

export async function upsertAdminMcpServer(
  payload: AdminMCPServerPayload,
): Promise<AdminMCPServerConfig> {
  const response = await authFetch("/admin/mcp/servers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await assertOk(response);
  return (await response.json()) as AdminMCPServerConfig;
}

export async function updateAdminMcpServer(
  serverId: string,
  payload: Partial<AdminMCPServerPayload>,
): Promise<AdminMCPServerConfig> {
  const { server_id: _serverId, ...body } = payload;
  const response = await authFetch(`/admin/mcp/servers/${encodeURIComponent(serverId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  await assertOk(response);
  return (await response.json()) as AdminMCPServerConfig;
}

export async function deleteAdminMcpServer(serverId: string): Promise<void> {
  const response = await authFetch(
    `/admin/mcp/servers/${encodeURIComponent(serverId)}`,
    { method: "DELETE" },
  );
  await assertOk(response);
}

export async function listAdminSkills(): Promise<AdminSkillDetail[]> {
  const response = await authFetch("/admin/skills");
  await assertOk(response);
  return (await response.json()) as AdminSkillDetail[];
}

export async function getAdminSkillDetail(skillId: string): Promise<AdminSkillDetail> {
  const response = await authFetch(`/admin/skills/${encodeURIComponent(skillId)}`);
  await assertOk(response);
  return (await response.json()) as AdminSkillDetail;
}

export async function updateAdminSkill(
  skillId: string,
  payload: AdminSkillUpdatePayload,
): Promise<AdminSkillDetail> {
  const response = await authFetch(`/admin/skills/${encodeURIComponent(skillId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await assertOk(response);
  return (await response.json()) as AdminSkillDetail;
}

export async function deleteAdminSkillOverride(skillId: string): Promise<AdminSkillDetail> {
  const response = await authFetch(
    `/admin/skills/${encodeURIComponent(skillId)}/override`,
    { method: "DELETE" },
  );
  await assertOk(response);
  return (await response.json()) as AdminSkillDetail;
}

export async function reloadAdminSkills(): Promise<void> {
  const response = await authFetch("/admin/skills/reload", { method: "POST" });
  await assertOk(response);
}

export async function exportSkillsArchive(): Promise<Blob> {
  const response = await authFetch("/admin/skills/export/zip");
  await assertOk(response);
  return await response.blob();
}

export async function listSessions(): Promise<SessionSummary[]> {
  const response = await authFetch("/sessions");
  await assertOk(response);
  return (await response.json()) as SessionSummary[];
}

export async function updateSessionTitle(sessionId: string, title: string): Promise<SessionSummary> {
  const response = await authFetch(`/sessions/${sessionId}/title`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  await assertOk(response);
  return (await response.json()) as SessionSummary;
}

export async function generateSessionTitle(sessionId: string): Promise<SessionSummary> {
  const response = await authFetch(`/sessions/${sessionId}/title/generate`, {
    method: "POST",
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

export async function deleteAllSessions(): Promise<void> {
  const response = await authFetch("/sessions", { method: "DELETE" });
  await assertOk(response);
}

export async function deleteLastMessages(sessionId: string, messageId: string): Promise<void> {
  const params = new URLSearchParams({ message_id: messageId });
  const response = await authFetch(`/sessions/${sessionId}/messages/last?${params.toString()}`, {
    method: "DELETE",
  });
  await assertOk(response);
}

export async function getSession(sessionId: string): Promise<SessionState> {
  const response = await authFetch(`/sessions/${sessionId}`);
  await assertOk(response);
  return (await response.json()) as SessionState;
}

export async function getSemanticCatalog(sessionId: string): Promise<SemanticCatalog> {
  const response = await authFetch(`/sessions/${sessionId}/semantic-catalog`);
  await assertOk(response);
  return (await response.json()) as SemanticCatalog;
}

export async function getSemanticCatalogStatus(
  sessionId: string,
): Promise<SemanticCatalogStatusResponse> {
  const response = await authFetch(`/sessions/${sessionId}/semantic-catalog/status`);
  await assertOk(response);
  return (await response.json()) as SemanticCatalogStatusResponse;
}

export async function refreshSemanticCatalog(sessionId: string): Promise<SemanticCatalog> {
  const response = await authFetch(`/sessions/${sessionId}/semantic-catalog/refresh`, {
    method: "POST",
  });
  await assertOk(response);
  return (await response.json()) as SemanticCatalog;
}

export async function generateSemanticCatalog(
  sessionId: string,
  payload: SemanticCatalogGenerationRequest = {},
): Promise<SemanticCatalogGenerationResponse> {
  const response = await authFetch(`/sessions/${sessionId}/semantic-catalog/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await assertOk(response);
  return (await response.json()) as SemanticCatalogGenerationResponse;
}

export async function startSemanticCatalogGeneration(
  sessionId: string,
  payload: SemanticCatalogGenerationRequest = {},
): Promise<SemanticCatalogGenerationAcceptedResponse> {
  const response = await authFetch(`/sessions/${sessionId}/semantic-catalog/generate?background=true`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await assertOk(response);
  return (await response.json()) as SemanticCatalogGenerationAcceptedResponse;
}

export async function createSemanticMetric(
  sessionId: string,
  payload: SemanticMetricPayload,
): Promise<SemanticMetric> {
  const response = await authFetch(`/sessions/${sessionId}/semantic-catalog/metrics`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await assertOk(response);
  return (await response.json()) as SemanticMetric;
}

export async function updateSemanticMetric(
  sessionId: string,
  metricId: string,
  payload: Partial<SemanticMetricPayload> & { is_active?: boolean },
): Promise<SemanticMetric> {
  const response = await authFetch(
    `/sessions/${sessionId}/semantic-catalog/metrics/${encodeURIComponent(metricId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
  await assertOk(response);
  return (await response.json()) as SemanticMetric;
}

export async function deleteSemanticMetric(sessionId: string, metricId: string): Promise<void> {
  const response = await authFetch(
    `/sessions/${sessionId}/semantic-catalog/metrics/${encodeURIComponent(metricId)}`,
    { method: "DELETE" },
  );
  await assertOk(response);
}

export async function createSemanticRelationship(
  sessionId: string,
  payload: SemanticRelationshipPayload,
): Promise<SemanticRelationship> {
  const response = await authFetch(`/sessions/${sessionId}/semantic-catalog/relationships`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await assertOk(response);
  return (await response.json()) as SemanticRelationship;
}

export async function updateSemanticRelationship(
  sessionId: string,
  relationshipId: string,
  payload: Partial<SemanticRelationshipPayload>,
): Promise<SemanticRelationship> {
  const response = await authFetch(
    `/sessions/${sessionId}/semantic-catalog/relationships/${encodeURIComponent(relationshipId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
  await assertOk(response);
  return (await response.json()) as SemanticRelationship;
}

export async function deleteSemanticRelationship(
  sessionId: string,
  relationshipId: string,
): Promise<void> {
  const response = await authFetch(
    `/sessions/${sessionId}/semantic-catalog/relationships/${encodeURIComponent(relationshipId)}`,
    { method: "DELETE" },
  );
  await assertOk(response);
}

export async function createSemanticTerm(
  sessionId: string,
  payload: SemanticTermPayload,
): Promise<SemanticTerm> {
  const response = await authFetch(`/sessions/${sessionId}/semantic-catalog/terms`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await assertOk(response);
  return (await response.json()) as SemanticTerm;
}

export async function updateSemanticTerm(
  sessionId: string,
  termId: string,
  payload: Partial<SemanticTermPayload>,
): Promise<SemanticTerm> {
  const response = await authFetch(
    `/sessions/${sessionId}/semantic-catalog/terms/${encodeURIComponent(termId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
  await assertOk(response);
  return (await response.json()) as SemanticTerm;
}

export async function deleteSemanticTerm(sessionId: string, termId: string): Promise<void> {
  const response = await authFetch(
    `/sessions/${sessionId}/semantic-catalog/terms/${encodeURIComponent(termId)}`,
    { method: "DELETE" },
  );
  await assertOk(response);
}

export async function getSessionNotebook(sessionId: string): Promise<string> {
  const response = await authFetch(`/sessions/${sessionId}/notebook`);
  await assertOk(response);
  return await response.text();
}

export interface NotebookCell {
  index: number;
  entry_type: "code" | "data_source_change";
  tool_name: string;
  language: string;
  question: string;
  code: string;
  result_summary: string;
  variables_created: string[];
  timestamp: string;
}

export async function getSessionNotebookCells(sessionId: string): Promise<NotebookCell[]> {
  const response = await authFetch(`/sessions/${sessionId}/notebook/cells`);
  await assertOk(response);
  return (await response.json()) as NotebookCell[];
}

export async function getRuntimeModelProfile(): Promise<RuntimeModelProfile> {
  const response = await authFetch("/runtime/model");
  await assertOk(response);
  return (await response.json()) as RuntimeModelProfile;
}

export async function getPhoenixOverview(): Promise<PhoenixOverview> {
  const response = await authFetch("/observability/phoenix");
  await assertOk(response);
  return (await response.json()) as PhoenixOverview;
}

export async function getPhoenixTraces(
  limit: number = 50,
  offset: number = 0,
): Promise<PhoenixTracesResponse> {
  const response = await authFetch(
    `/observability/phoenix/traces?limit=${limit}&offset=${offset}`,
  );
  await assertOk(response);
  return (await response.json()) as PhoenixTracesResponse;
}

export async function getPhoenixTraceDetail(
  traceId: string,
): Promise<PhoenixTraceDetail> {
  const response = await authFetch(
    `/observability/phoenix/traces/${encodeURIComponent(traceId)}`,
  );
  await assertOk(response);
  return (await response.json()) as PhoenixTraceDetail;
}

export async function getPhoenixTracesBySession(
  sessionId: string,
): Promise<PhoenixTraceDetail> {
  const response = await authFetch(
    `/observability/phoenix/traces/by-session/${encodeURIComponent(sessionId)}`,
  );
  await assertOk(response);
  return (await response.json()) as PhoenixTraceDetail;
}

export async function uploadTabularFiles(
  sessionId: string,
  files: File[],
  preprocessingOptions?: TabularPreprocessingOptions,
): Promise<BatchUploadResponse> {
  const cleanFiles = files.filter(Boolean);
  if (cleanFiles.length === 0) {
    throw new Error("No files selected");
  }
  const formData = new FormData();
  cleanFiles.forEach((file) => {
    formData.append("files", file);
  });
  if (preprocessingOptions) {
    formData.append("preprocessing_options", JSON.stringify(preprocessingOptions));
  }
  const response = await authFetch(`/sessions/${sessionId}/data/batch`, {
    method: "POST",
    body: formData,
  });
  await assertOk(response);
  return (await response.json()) as BatchUploadResponse;
}

export async function uploadCsv(sessionId: string, file: File): Promise<void> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await authFetch(`/sessions/${sessionId}/data`, {
    method: "POST",
    body: formData,
  });
  await assertOk(response);
}

export async function listDbConnections(): Promise<DBConnection[]> {
  const response = await authFetch("/db-connections");
  await assertOk(response);
  return (await response.json()) as DBConnection[];
}

export async function createDbConnection(payload: DBConnectionFormPayload): Promise<DBConnection> {
  const response = await authFetch("/db-connections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await assertOk(response);
  return (await response.json()) as DBConnection;
}

export async function updateDbConnection(
  connectionId: string,
  payload: Partial<DBConnectionFormPayload>,
): Promise<DBConnection> {
  const response = await authFetch(`/db-connections/${connectionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await assertOk(response);
  return (await response.json()) as DBConnection;
}

export async function deleteDbConnection(connectionId: string): Promise<void> {
  const response = await authFetch(`/db-connections/${connectionId}`, {
    method: "DELETE",
  });
  await assertOk(response);
}

export async function testDbConnection(connectionId: string): Promise<DBConnectionTestResult> {
  const response = await authFetch(`/db-connections/${connectionId}/test`, {
    method: "POST",
  });
  await assertOk(response);
  return (await response.json()) as DBConnectionTestResult;
}

export async function listDbConnectionSchemas(connectionId: string): Promise<DBConnectionSchema[]> {
  const response = await authFetch(`/db-connections/${connectionId}/schemas`);
  await assertOk(response);
  return (await response.json()) as DBConnectionSchema[];
}

export async function bindDbConnectionSource(
  sessionId: string,
  connectionId: string,
  sourceMode?: string,
): Promise<SessionSourceState> {
  const response = await authFetch(`/sessions/${sessionId}/source/db-connection`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      connection_id: connectionId,
      source_mode: sourceMode,
    }),
  });
  await assertOk(response);
  return (await response.json()) as SessionSourceState;
}

export async function bindCsvSource(
  sessionId: string,
): Promise<SessionSourceState> {
  const response = await authFetch(`/sessions/${sessionId}/source/csv`, {
    method: "POST",
  });
  await assertOk(response);
  return (await response.json()) as SessionSourceState;
}

export async function uploadRagDocument(
  sessionId: string,
  file: File,
): Promise<RagUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await authFetch(`/sessions/${sessionId}/rag/documents`, {
    method: "POST",
    body: formData,
  });
  await assertOk(response);
  return (await response.json()) as RagUploadResponse;
}

export async function getRagUploadStatus(
  sessionId: string,
  trackId: string,
): Promise<RagTrackStatusResponse> {
  const response = await authFetch(`/sessions/${sessionId}/rag/uploads/${encodeURIComponent(trackId)}`);
  await assertOk(response);
  return (await response.json()) as RagTrackStatusResponse;
}

export async function listRagDocuments(
  sessionId: string,
): Promise<RagDocumentsResponse> {
  const response = await authFetch(`/sessions/${sessionId}/rag/documents`);
  await assertOk(response);
  return (await response.json()) as RagDocumentsResponse;
}

export async function deleteRagDocument(
  sessionId: string,
  documentId: string,
): Promise<RagDocumentDeleteResponse> {
  const response = await authFetch(
    `/sessions/${sessionId}/rag/documents/${encodeURIComponent(documentId)}`,
    {
      method: "DELETE",
    },
  );
  await assertOk(response);
  return (await response.json()) as RagDocumentDeleteResponse;
}

export async function bindRagSource(
  sessionId: string,
): Promise<SessionSourceState> {
  const response = await authFetch(`/sessions/${sessionId}/source/rag`, {
    method: "POST",
  });
  await assertOk(response);
  return (await response.json()) as SessionSourceState;
}

export async function bindOpenProjectSource(
  sessionId: string,
  payload?: OpenProjectSyncRequest,
): Promise<OpenProjectSyncResponse> {
  const response = await authFetch(`/sessions/${sessionId}/source/openproject`, {
    method: "POST",
    headers: payload ? { "Content-Type": "application/json" } : undefined,
    body: payload ? JSON.stringify(payload) : undefined,
  });
  await assertOk(response);
  return (await response.json()) as OpenProjectSyncResponse;
}

export async function listOpenProjectProjects(
  sessionId: string,
  payload?: OpenProjectSyncRequest,
): Promise<OpenProjectProjectsResponse> {
  const response = await authFetch(`/sessions/${sessionId}/source/openproject/projects`, {
    method: "POST",
    headers: payload ? { "Content-Type": "application/json" } : undefined,
    body: payload ? JSON.stringify(payload) : undefined,
  });
  await assertOk(response);
  return (await response.json()) as OpenProjectProjectsResponse;
}

export async function clearSessionSource(
  sessionId: string,
): Promise<SessionSourceState> {
  const response = await authFetch(`/sessions/${sessionId}/source/clear`, {
    method: "POST",
  });
  await assertOk(response);
  return (await response.json()) as SessionSourceState;
}

type ToolEvent = {
  tool_name: string;
  tool_call_id?: string;
  input_preview?: string;
  input_summary?: string;
  input_code?: string;
  output_preview?: string;
  result_summary?: string;
  status?: string;
  artifact_keys?: string[];
};

type StreamHandlers = {
  onToken: (token: string) => void;
  onFinal: (payload: QueryResponse) => void;
  onReasoning: (payload: string, mode: "chunk" | "token") => void;
  onContextUsage?: (payload: ContextUsageSnapshot) => void;
  onPhase?: (event: PhaseEvent) => void;
  /** Fired when a thinking block starts (model entered <think>). No payload. */
  onThinkingStart?: () => void;
  /** Fired when a thinking block ends. Carries the complete thinking text for this LLM call. */
  onThinkingEnd?: (text: string) => void;
  onToolStart?: (event: ToolEvent) => void;
  onToolEnd?: (event: ToolEvent) => void;
  onGraphUpdate?: (graph: ExecutionGraph) => void;
  onError: (error: string) => void;
};

function consumeSseLine(
  line: string,
  state: { event: string | null },
  handlers: StreamHandlers,
): boolean {
  if (line.startsWith("event: ")) {
    state.event = line.slice(7).trim();
    return false;
  }
  if (!line.startsWith("data: ")) {
    return false;
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
    return false;
  }
  if (currentEvent === "final" && typeof payload === "object" && payload !== null) {
    handlers.onFinal(payload as QueryResponse);
    return true;
  }
  if (currentEvent === "reasoning" && typeof payload === "string") {
    handlers.onReasoning(payload, "chunk");
    return false;
  }
  if (currentEvent === "reasoning_token" && typeof payload === "string") {
    handlers.onReasoning(payload, "token");
    return false;
  }
  if (currentEvent === "phase" && typeof payload === "object" && payload !== null) {
    handlers.onPhase?.(payload as PhaseEvent);
    return false;
  }
  if (currentEvent === "context_usage" && typeof payload === "object" && payload !== null) {
    handlers.onContextUsage?.(payload as ContextUsageSnapshot);
    return false;
  }
  if (currentEvent === "thinking_start") {
    handlers.onThinkingStart?.();
    return false;
  }
  if (currentEvent === "thinking_end" && typeof payload === "string") {
    handlers.onThinkingEnd?.(payload);
    return false;
  }
  if (currentEvent === "tool_start" && typeof payload === "object" && payload !== null) {
    handlers.onToolStart?.(payload as ToolEvent);
    return false;
  }
  if (currentEvent === "tool_end" && typeof payload === "object" && payload !== null) {
    handlers.onToolEnd?.(payload as ToolEvent);
    return false;
  }
  if (currentEvent === "execution_graph" && typeof payload === "object" && payload !== null) {
    handlers.onGraphUpdate?.(payload as ExecutionGraph);
    return false;
  }
  if (currentEvent === "error") {
    handlers.onError(typeof payload === "string" ? payload : JSON.stringify(payload));
    return true;
  }
  return false;
}

export async function streamQuery(
  sessionId: string,
  query: string,
  includeReasoning: boolean,
  useHistory: boolean,
  handlers: StreamHandlers,
  signal?: AbortSignal,
  analysisDepth?: string,
  selectedSkillIds?: string[],
): Promise<void> {
  const body = buildQueryPayload(
    query,
    includeReasoning,
    useHistory,
    analysisDepth,
    selectedSkillIds,
  );
  const response = await authFetch(`/sessions/${sessionId}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify(body),
  });
  await assertOk(response);

  if (!response.body) {
    throw new Error("SSE stream body is empty");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const state = { event: null as string | null };
  let carry = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      carry += decoder.decode(value, { stream: true });
      const lines = carry.split("\n");
      carry = lines.pop() ?? "";
      for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        if (!line) {
          continue;
        }
        if (consumeSseLine(line, state, handlers)) {
          return;
        }
      }
    }

    const tail = (carry + decoder.decode()).trim();
    if (!tail) {
      return;
    }
    for (const rawLine of tail.split("\n")) {
      const line = rawLine.trimEnd();
      if (!line) {
        continue;
      }
      if (consumeSseLine(line, state, handlers)) {
        return;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function buildQueryPayload(
  query: string,
  includeReasoning: boolean,
  useHistory: boolean,
  analysisDepth?: string,
  selectedSkillIds?: string[],
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    query,
    use_history: useHistory,
    include_reasoning: includeReasoning,
  };
  if (analysisDepth) {
    body.analysis_depth = analysisDepth;
  }
  if (selectedSkillIds && selectedSkillIds.length > 0) {
    body.selected_skill_ids = selectedSkillIds;
  }
  return body;
}

export type BoardExportSectionPayload = {
  label: string;
  artifact_ids: string[];
};

export async function exportBoardReport(
  format: BoardExportFormat,
  artifacts: ArtifactPayload[],
  title: string,
  sections: BoardExportSectionPayload[] = [],
): Promise<void> {
  const response = await authFetch("/reports/board-export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      format,
      title,
      artifacts,
      sections,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    if (text.includes("405 Not Allowed") || text.includes("<html>")) {
      throw new Error(
        "Сервер не принял запрос экспорта (405). Пересоберите frontend-образ: в nginx нужен прокси для /reports/.",
      );
    }
    throw new Error(text || `HTTP ${response.status}`);
  }

  const blob = await response.blob();
  const extension = format === "pdf" ? "pdf" : format === "xlsx" ? "xlsx" : "docx";
  const safeTitle = title
    .trim()
    .replace(/[^\wа-яА-ЯёЁ\s-]+/gu, "")
    .replace(/\s+/g, "_")
    .slice(0, 60);
  const fileName = `${safeTitle || "board_report"}.${extension}`;

  const objectUrl = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(objectUrl);
}

export async function downloadReportFile(href: string): Promise<void> {
  const headers = new Headers();
  const token = getStoredToken();

  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const cleanHref = href.trim();

  let url: string;
  if (/^https?:\/\//i.test(cleanHref)) {
    url = cleanHref;
  } else {
    const cleanPath = cleanHref.startsWith("/")
      ? cleanHref
      : `/${cleanHref}`;
    url = `${API_BASE}${cleanPath}`;
  }

  const response = await fetch(url, {
    method: "GET",
    headers,
  });

  await assertOk(response);

  const blob = await response.blob();

  const contentDisposition = response.headers.get("content-disposition") ?? "";
  const fileNameMatch = contentDisposition.match(/filename="?([^"]+)"?/i);
  const fileName = fileNameMatch?.[1] || cleanHref.split("/").pop() || "report.docx";

  const objectUrl = window.URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = objectUrl;
  link.download = decodeURIComponent(fileName);
  document.body.appendChild(link);
  link.click();
  link.remove();

  window.URL.revokeObjectURL(objectUrl);
}
