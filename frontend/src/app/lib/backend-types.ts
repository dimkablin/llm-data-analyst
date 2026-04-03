export type ArtifactPayload = {
  id: string;
  type: string;
  text?: string | null;
  role: string;
  meta: Record<string, unknown>;
  timestamp: string;
  data: {
    format: string;
    data: unknown;
  };
};

export type QueryMetrics = {
  duration_ms: number;
  artifact_count: number;
  table_count: number;
  plot_count: number;
  value_count: number;
  model: string;
};

export type QueryResponse = {
  session_id: string;
  text: string;
  reasoning?: string | null;
  artifacts: ArtifactPayload[];
  values?: Record<string, unknown> | null;
  metrics: QueryMetrics;
};

export type SessionSource = {
  alias: string;
  source_type: "csv" | "db_connection";
  display_name: string;
  variable_name: string;
  file_name?: string | null;
  connection_id?: string | null;
  connection_name?: string | null;
  bound_at: string;
  schema_hint?: Record<string, string>;
};

export type SessionState = {
  session_id: string;
  title: string;
  chat_history: Array<{
    id?: string;
    role: string;
    content: string;
    timestamp: string;
    reasoning?: string | null;
    artifacts?: ArtifactPayload[];
  }>;
  artifacts: ArtifactPayload[];
  has_dataset: boolean;
  dataset_name?: string | null;
  source_type?: "csv" | "db_connection" | null;
  source_ref_id?: string | null;
  source_label?: string | null;
  source_mode?: string | null;
  sources?: SessionSource[];
  session_memory?: string;
};

export type SessionSourceState = {
  source_type?: "csv" | "db_connection" | null;
  source_ref_id?: string | null;
  source_label?: string | null;
  source_mode?: string | null;
};

export type SessionSummary = {
  session_id: string;
  title: string;
  created_at: string;
  last_access: string;
  has_dataset: boolean;
  last_message_preview?: string | null;
};

export type DBConnectionType = "postgresql" | "clickhouse";

export type DBConnection = {
  id: string;
  name: string;
  db_type: DBConnectionType;
  host: string;
  port?: number | null;
  database?: string | null;
  username?: string | null;
  options_json?: Record<string, unknown> | null;
  password_present: boolean;
  last_test_at?: string | null;
  last_test_ok?: boolean | null;
  last_error?: string | null;
  created_at: string;
  updated_at: string;
};

export type DBConnectionTestResult = {
  ok: boolean;
  checked_at: string;
  last_test_at: string;
  last_test_ok: boolean;
  error?: string | null;
};

export type DBConnectionFormPayload = {
  name: string;
  db_type: DBConnectionType;
  host: string;
  port?: number | null;
  database?: string | null;
  username?: string | null;
  password?: string | null;
  clear_password?: boolean;
  options_json?: Record<string, unknown> | null;
};

export type AuthUser = {
  id: number;
  username: string;
  is_admin: boolean;
  created_at: string;
};

export type AuthResult = {
  access_token: string;
  token_type: "bearer";
  user: AuthUser;
};

export type AnalysisDepth = "light" | "medium" | "deep";

/** Upper bound on outer ReAct steps enforced server-side per analysis depth. */
export const ANALYSIS_DEPTH_STEP_CEILING: Record<AnalysisDepth, number> = {
  light: 20,
  medium: 30,
  deep: 50,
};

export function clampAgentMaxStepsForDepth(depth: AnalysisDepth, value: number): number {
  const cap = ANALYSIS_DEPTH_STEP_CEILING[depth];
  const v = Math.round(value);
  if (!Number.isFinite(v)) {
    return Math.min(20, cap);
  }
  return Math.min(Math.max(2, v), cap);
}

export type UserSettings = {
  theme: "light" | "dark";
  default_include_reasoning: boolean;
  default_answer_style: "concise" | "detailed";
  analysis_depth: AnalysisDepth;
  llm_temperature_chat: number;
  llm_temperature_tool: number;
  llm_max_tokens_default: number;
  llm_max_tokens_reasoning: number;
  backend_query_timeout_sec: number;
  agent_max_steps: number;
  agent_step_timeout_sec: number;
  agent_inner_recursion_limit: number;
  ui_scale: number;
};

export type RuntimeModelProfile = {
  provider: string;
  model: string;
  base_url: string;
};

export type ToolAvailability = {
  tool_key: string;
  kind: "builtin" | "integration";
  tool_label: string;
  display_name_ru?: string | null;
  description: string;
  description_ru?: string | null;
  capabilities: string[];
  requires_session_data: boolean;
  source_type?: string | null;
  source_ref_id?: string | null;
  source_mode?: string | null;
  enabled_globally: boolean;
  available_globally: boolean;
  enabled_for_user: boolean;
  effective_enabled: boolean;
  status: string;
  timeout_hint_sec?: number | null;
};

export type PhaseEvent = {
  id?: string;
  phase: "think" | "act" | "evaluate" | "finalize";
  title: string;
  content: string;
  timestamp: string;
  step_index?: number;
  max_steps?: number;
  status?: string;
};

export type GraphNode = {
  id: string;
  type: "phase" | "tool";
  label: string;
  status: "pending" | "running" | "done" | "error";
  duration_ms?: number;
  tool_name?: string;
  artifact_keys?: string[];
  parent_id?: string;
};

export type GraphEdge = {
  from: string;
  to: string;
  label?: string;
};

export type ExecutionGraph = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type ChatMessage = {
  /** Frontend-local composite ID (used as React key). */
  id: string;
  /** UUID assigned by the backend when the message was persisted. Used for server-side deletion. */
  backendId?: string;
  timestamp: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string | null;
  phases?: PhaseEvent[];
  liveReasoningTrace?: string | null;
  livePhases?: PhaseEvent[];
  metrics?: QueryMetrics;
  artifacts?: ArtifactPayload[];
  executionGraph?: ExecutionGraph;
};

export type PhoenixOverviewStats = {
  total_traces: number;
  success_rate: number;
  p50_latency_ms: number;
  unique_sessions: number;
};

export type PhoenixLatencyPoint = {
  label: string;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  trace_count: number;
};

export type PhoenixTokenUsageRow = {
  trace_id: string;
  session_id?: string | null;
  query_preview: string;
  model?: string | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  total_tokens?: number | null;
  duration_ms: number;
  started_at: string;
  token_source: string;
};

export type PhoenixTraceRow = {
  trace_id: string;
  session_id?: string | null;
  query_preview: string;
  request_kind: string;
  user?: string | null;
  status: string;
  duration_ms: number;
  tool_calls: number;
  span_count: number;
  model?: string | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  total_tokens?: number | null;
  started_at: string;
};

export type PhoenixOverview = {
  available: boolean;
  project_name: string;
  project_id?: string | null;
  generated_at: string;
  dashboard_url?: string | null;
  embed_url?: string | null;
  stats: PhoenixOverviewStats;
  latency: PhoenixLatencyPoint[];
  token_usage: PhoenixTokenUsageRow[];
  traces: PhoenixTraceRow[];
  warnings: string[];
};

export type UserMemory = {
  profile: string;
  notes: string;
};
