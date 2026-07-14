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

export type SourceType = "csv" | "db_connection" | "rag" | "openproject";

export type SessionSource = {
  alias: string;
  source_type: SourceType;
  display_name: string;
  variable_name: string;
  file_name?: string | null;
  connection_id?: string | null;
  connection_name?: string | null;
  bound_at: string;
  csv_table_names?: string[];
  schema_hint?: Record<string, string>;
};

export type SemanticCatalogStatus =
  | "pending"
  | "indexing"
  | "ready"
  | "failed"
  | "stale"
  | "degraded"
  | "unbound"
  | "empty";

export type SemanticTable = {
  table_id: string;
  qualified_name: string;
  table_name: string;
  source_kind: string;
  schema_name?: string | null;
  description: string;
  semantic_role: string;
  grain: string;
  row_count?: number | null;
  columns_count: number;
  aliases: string[];
  tags: string[];
  quality_notes: string[];
  ai_context: string;
  is_hidden: boolean;
};

export type SemanticColumn = {
  column_id: string;
  table: string;
  name: string;
  dtype: string;
  nullable?: boolean | null;
  semantic_role: string;
  description: string;
  aliases: string[];
  examples: string[];
  quality_notes: string[];
  ai_context: string;
  is_hidden: boolean;
};

export type SemanticEntity = {
  entity_id: string;
  name: string;
  table: string;
  expr: string;
  type: "primary" | "unique" | "foreign" | "natural";
  description: string;
  synonyms: string[];
  is_active: boolean;
};

export type SemanticDimension = {
  dimension_id: string;
  name: string;
  table: string;
  expr: string;
  type: "categorical" | "time" | "boolean" | "number";
  grains: string[];
  description: string;
  synonyms: string[];
  is_active: boolean;
};

export type SemanticFact = {
  fact_id: string;
  name: string;
  table: string;
  expr: string;
  type: "number" | "money" | "duration" | "count";
  description: string;
  synonyms: string[];
};

export type SemanticMetricKind =
  | "simple"
  | "derived"
  | "ratio"
  | "cumulative"
  | "conversion"
  | "filtered"
  | "non_additive"
  | "time_comparison";

export type SemanticMetricAggregation = "sum" | "avg" | "count" | "count_distinct" | "min" | "max";

export type SemanticMetric = {
  metric_id: string;
  key: string;
  name: string;
  type: SemanticMetricKind;
  base_table: string;
  expr?: string | null;
  agg?: SemanticMetricAggregation | null;
  numerator?: string | null;
  denominator?: string | null;
  formula: string;
  default_time_dimension?: string | null;
  allowed_dimensions: string[];
  filters: string[];
  format: string;
  description: string;
  synonyms: string[];
  is_active: boolean;
  created_at: string;
  updated_at: string;
};

export type SemanticRelationship = {
  relationship_id: string;
  from_table: string;
  from_column: string;
  to_table: string;
  to_column: string;
  description: string;
  cardinality: "one_to_one" | "one_to_many" | "many_to_one" | "many_to_many" | "unknown";
  is_active: boolean;
};

export type SemanticRelationshipPayload = Omit<SemanticRelationship, "relationship_id">;

export type SemanticValidationIssue = {
  code: string;
  message: string;
  object_type: string;
  object_id: string;
};

export type SemanticValidationResult = {
  errors: SemanticValidationIssue[];
  warnings: SemanticValidationIssue[];
  quality_score: number;
};

export type SemanticSavedQuery = {
  query_id: string;
  name: string;
  metrics: string[];
  dimensions: string[];
  filters: string[];
  description: string;
};

export type SemanticMetricPayload = {
  key: string;
  name: string;
  type?: SemanticMetricKind;
  base_table: string;
  expr?: string | null;
  agg?: SemanticMetricAggregation | null;
  numerator?: string | null;
  denominator?: string | null;
  formula?: string;
  default_time_dimension?: string | null;
  allowed_dimensions?: string[];
  filters?: string[];
  format?: string;
  synonyms?: string[];
  description?: string;
};

export type SemanticTerm = {
  term_id: string;
  name: string;
  description: string;
  synonyms: string[];
  entity_refs: string[];
  is_active: boolean;
  created_at: string;
  updated_at: string;
};

export type SemanticTermPayload = {
  name: string;
  description?: string;
  synonyms?: string[];
  entity_refs?: string[];
  is_active?: boolean;
};

export type SemanticCatalog = {
  catalog_id: string;
  user_id: number;
  session_id: string;
  source_key: string;
  source_type: string;
  source_ref_id: string;
  source_label: string;
  source_fingerprint: string;
  version: string;
  overlay_version: number;
  published_version: number;
  status: SemanticCatalogStatus;
  error?: string | null;
  built_at: string;
  updated_at: string;
  tables: SemanticTable[];
  columns: SemanticColumn[];
  entities: SemanticEntity[];
  dimensions: SemanticDimension[];
  facts: SemanticFact[];
  relationships: SemanticRelationship[];
  metrics: SemanticMetric[];
  saved_queries: SemanticSavedQuery[];
  terms: SemanticTerm[];
  validation: SemanticValidationResult;
};

export type SemanticCatalogStatusResponse = {
  status: SemanticCatalogStatus;
  catalog_id?: string | null;
  source_fingerprint?: string | null;
  updated_at?: string | null;
  error?: string | null;
};

export type SemanticCatalogGenerationRequest = {
  sample_rows?: number;
  max_tables?: number;
};

export type SemanticCatalogGenerationSummary = {
  tables_scanned: number;
  sample_tables: number;
  table_patches: number;
  column_patches: number;
  metrics_added: number;
  terms_added: number;
  relationships_added: number;
  rejected_items: string[];
};

export type SemanticCatalogGenerationResponse = {
  catalog: SemanticCatalog;
  summary: SemanticCatalogGenerationSummary;
};

export type SemanticCatalogGenerationAcceptedResponse = {
  accepted: true;
  status: string;
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
    reasoning_steps?: PersistedReasoningStep[] | null;
    artifacts?: ArtifactPayload[];
    tools?: PersistedToolCall[];
  }>;
  artifacts: ArtifactPayload[];
  has_dataset: boolean;
  dataset_name?: string | null;
  source_type?: SourceType | null;
  source_ref_id?: string | null;
  source_label?: string | null;
  source_mode?: string | null;
  sources?: SessionSource[];
  session_memory?: string;
  selected_skill_ids?: string[];
  context_usage?: ContextUsageSnapshot | null;
};

export type UploadedTable = {
  file_name: string;
  file_format: string;
  table_name: string;
  source_alias: string;
  variable_name: string;
  parquet_path: string;
  rows: number;
  columns: number;
  preprocessing: TabularPreprocessingSummary;
};

export type TabularPreprocessingOptions = {
  enabled: boolean;
  detect_csv_separator: boolean;
  detect_header_row: boolean;
  normalize_empty_values: boolean;
  drop_empty_rows: boolean;
  drop_empty_columns: boolean;
  drop_sparse_rows: boolean;
  unique_column_names: boolean;
  header_scan_rows: number;
  sparse_row_min_ratio: number;
};

export type TabularPreprocessingSummary = {
  enabled: boolean;
  raw_rows: number;
  raw_columns: number;
  cleaned_rows: number;
  cleaned_columns: number;
  detected_header_row?: number | null;
  removed_rows: number;
  removed_columns: number;
};

export const DEFAULT_TABULAR_PREPROCESSING_OPTIONS: TabularPreprocessingOptions = {
  enabled: true,
  detect_csv_separator: true,
  detect_header_row: true,
  normalize_empty_values: true,
  drop_empty_rows: true,
  drop_empty_columns: true,
  drop_sparse_rows: true,
  unique_column_names: true,
  header_scan_rows: 50,
  sparse_row_min_ratio: 0.5,
};

export type BatchUploadResponse = {
  session_id: string;
  csv_session_id: string;
  table_names: string[];
  files: UploadedTable[];
  expires_at: number;
  total_rows: number;
  total_columns: number;
  dataset_name: string;
};

export type SessionSourceState = {
  source_type?: SourceType | null;
  source_ref_id?: string | null;
  source_label?: string | null;
  source_mode?: string | null;
};

export type OpenProjectSyncResponse = {
  source_type: "openproject";
  source_ref_id: string;
  source_label: string;
  source_mode: "postgres_sync";
  connection_id: string;
  connection_name: string;
  schema_name: string;
  tables: Record<string, number>;
  artifact_ids: string[];
  synced_at: string;
};

export type OpenProjectSyncRequest = {
  base_url?: string | null;
  api_key?: string | null;
  host_header?: string | null;
  project?: string | null;
  all_projects?: boolean | null;
  days?: number | null;
  max_items?: number | null;
};

export type OpenProjectProject = {
  id: string;
  identifier: string;
  name: string;
};

export type OpenProjectProjectsResponse = {
  projects: OpenProjectProject[];
};

export type RagDocumentStatus = {
  id?: string | null;
  file_path?: string | null;
  status: string;
  track_id?: string | null;
  chunks_count?: number | null;
  content_length?: number | null;
  created_at?: string | null;
  updated_at?: string | null;
  error_msg?: string | null;
};

export type RagUploadResponse = {
  status: string;
  message: string;
  track_id: string;
};

export type RagTrackStatusResponse = {
  track_id: string;
  documents: RagDocumentStatus[];
  status_summary: Record<string, number>;
};

export type RagDocumentsResponse = {
  documents: RagDocumentStatus[];
};

export type RagDocumentDeleteResponse = {
  status: string;
  message: string;
  document_id: string;
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

export type DBConnectionSchema = {
  name: string;
  display_name: string;
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
export type AnalysisMode = "fast" | "deep";

/** Inner tool-call loop limit enforced server-side per analysis depth (mirrors runner.DEPTH_PROFILES). */
export const ANALYSIS_DEPTH_STEP_CEILING: Record<AnalysisDepth, number> = {
  light: 32,
  medium: 64,
  deep: 120,
};

export function clampAgentMaxStepsForDepth(depth: AnalysisDepth, value: number): number {
  const cap = ANALYSIS_DEPTH_STEP_CEILING[depth];
  const v = Math.round(value);
  if (!Number.isFinite(v)) {
    return Math.min(80, cap);
  }
  return Math.min(Math.max(2, v), cap);
}

export type UserSettings = {
  theme: "light" | "dark";
  default_include_reasoning: boolean;
  analysis_mode: AnalysisMode;
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
  llm_streaming: boolean;
  show_thinking: boolean;
  show_think_planning: boolean;
  show_think_tool: boolean;
  show_think_final: boolean;
  /** Admin: show raw planner/sql/pandas rows instead of 5 high-level stages. */
  show_detailed_tool_steps: boolean;
};

export type RuntimeModelProfile = {
  provider: string;
  model: string;
  base_url: string;
  max_context_tokens?: number | null;
  context_window_source?: ContextWindowSource;
};

export type ContextWindowSource = "settings" | "unavailable";

export type ContextUsageStatus =
  | "unavailable"
  | "normal"
  | "warning"
  | "critical"
  | "overflow";

export type ContextCompactionStatus = "idle" | "running" | "done" | "failed";

export type ContextUsageSnapshot = {
  input_tokens: number;
  reserved_response_tokens: number;
  used_tokens: number;
  max_context_tokens?: number | null;
  remaining_tokens?: number | null;
  usage_ratio?: number | null;
  usage_percent?: number | null;
  overflow: boolean;
  status: ContextUsageStatus;
  context_window_source: ContextWindowSource;
  compaction_status?: ContextCompactionStatus;
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

export type MCPServerTransport = "streamable_http" | "stdio";

export type MCPToolDescriptor = {
  server_id: string;
  tool_name: string;
  tool_key: string;
  description?: string | null;
  input_schema: Record<string, unknown>;
  output_schema?: Record<string, unknown> | null;
};

export type MCPServerAvailability = {
  server_id: string;
  name: string;
  description?: string | null;
  transport: MCPServerTransport;
  enabled_globally: boolean;
  available_globally: boolean;
  status: string;
  enabled_by_default: boolean;
  enabled_for_user: boolean;
  effective_enabled: boolean;
  tools: MCPToolDescriptor[];
  tool_count: number;
  last_error?: string | null;
};

export type AdminMCPServerConfig = {
  server_id: string;
  name: string;
  description?: string | null;
  transport: MCPServerTransport;
  url?: string | null;
  command?: string | null;
  args: string[];
  env: Record<string, string>;
  timeout_sec: number;
  enabled: boolean;
  enabled_by_default: boolean;
  created_at?: string | null;
  updated_at?: string | null;
  updated_by?: number | null;
};

export type AdminMCPServerPayload = {
  server_id: string;
  name: string;
  description?: string | null;
  transport: MCPServerTransport;
  url?: string | null;
  command?: string | null;
  args?: string[];
  env?: Record<string, string>;
  timeout_sec?: number;
  enabled?: boolean;
  enabled_by_default?: boolean;
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

/**
 * Persisted storage contract for a single tool invocation.
 * Separate from StreamToolCall (live UI DTO) so they can evolve independently.
 */
export type PersistedToolCall = {
  tool_call_id?: string;
  tool_name: string;
  /** "done" | "error" */
  status: string;
  input_summary?: string;
  /** Raw tool input shown when the tool row is expanded. */
  input_preview?: string;
  /** Full tool code when available (pandas/plotly/value), up to 4000 chars. */
  input_code?: string;
  result_summary?: string;
  output_preview?: string;
  artifact_keys?: string[];
  /** ISO timestamp of tool_start. */
  started_at?: string;
  /** ISO timestamp of tool_end. */
  finished_at?: string;
  error?: string;
  /** Thinking text that preceded this tool call. */
  pre_reasoning?: string;
};

export type StreamToolCall = {
  /** Client-side ID (uuid generated on tool_start). */
  id: string;
  tool_call_id?: string;
  tool_name: string;
  /** Short human-readable summary of the tool input, e.g. first line of code or file path. */
  input_summary: string;
  /** Raw tool input shown when the tool row is expanded. */
  input_preview?: string;
  /** Full generated code (preferred over truncated input_preview). */
  input_code?: string;
  /** Raw tool output text (up to 800 chars), shown in expand view. */
  output_preview?: string;
  result_summary?: string;
  status: "running" | "done" | "error";
  artifact_keys?: string[];
  started_at: number;
  /** Reasoning text that preceded this tool call (delta since previous tool call). */
  pre_reasoning?: string;
};

// ─── Assistant block model (Claude-like sequential rendering) ────────────────

export type ThinkingBlock = {
  type: "thinking";
  id: string;
  content: string;
  /** Mirrors PersistedReasoningStep.kind — used for per-kind visibility filtering. */
  kind?: "planning" | "tool_synthesis" | "final_synthesis" | "unknown";
};

export type TextBlock = {
  type: "text";
  id: string;
  content: string;
};

export type ToolUseBlock = {
  type: "tool_use";
  id: string;
  tool_call_id?: string;
  tool_name: string;
  input_summary: string;
  input_code?: string;
  input_preview?: string;
  status: "running" | "done" | "error";
  started_at: number;
  result_summary?: string;
  output_preview?: string;
  artifact_keys?: string[];
};

export type ToolResultBlock = {
  type: "tool_result";
  id: string;
  tool_use_id: string;
  tool_name: string;
  status: "ok" | "error";
  result_summary: string;
  output_preview?: string;
  artifact_keys?: string[];
};

export type AssistantBlock =
  | ThinkingBlock
  | TextBlock
  | ToolUseBlock
  | ToolResultBlock;

/** One LLM call's thinking block, persisted per-step for accurate reload rendering. */
export type PersistedReasoningStep = {
  step_index: number;
  kind: "planning" | "tool_synthesis" | "final_synthesis" | "unknown";
  content: string;
  tool_name?: string | null;
};

// ─── Chat message ────────────────────────────────────────────────────────────

export type ChatMessage = {
  /** Frontend-local composite ID (used as React key). */
  id: string;
  /** UUID assigned by the backend when the message was persisted. Used for server-side deletion. */
  backendId?: string;
  timestamp: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string | null;
  /** Per-step LLM thinking blocks for accurate multi-block rendering on reload. */
  reasoning_steps?: PersistedReasoningStep[] | null;
  phases?: PhaseEvent[];
  /** Tool calls that happened during this message (for inline display, Claude Code style). */
  tools?: StreamToolCall[];
  /** Ordered block sequence for Claude-like sequential rendering. */
  blocks?: AssistantBlock[];
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
  skill_ids?: string | null;
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

export type PhoenixSpanSnapshotItem = {
  span_id: string;
  parent_id?: string | null;
  name: string;
  span_kind: string;
  status_code: string;
  status_message: string;
  duration_ms: number;
  start_time: string;
  end_time: string;
  input_tokens?: number | null;
  output_tokens?: number | null;
  total_tokens?: number | null;
  input_value?: string | null;
  output_value?: string | null;
  skill_ids?: string | null;
};

export type PhoenixTraceDetail = {
  trace_id: string;
  project_id?: string | null;
  project_name?: string | null;
  spans: PhoenixSpanSnapshotItem[];
};

export type PhoenixTracesResponse = {
  total: number;
  traces: PhoenixTraceRow[];
};

export type UserMemory = {
  profile: string;
  notes: string;
};

export type Skill = {
  skill_id: string;
  name: string;
  description: string;
  triggers: string[];
  enabled_for_user: boolean;
};

export type AdminSkillDetail = {
  skill_id: string;
  name: string;
  description: string;
  triggers: string[];
  source_path: string;
  kind: string;
  tool_key: string | null;
  core_markdown: string;
  details_markdown: string | null;
  is_overridden: boolean;
  updated_by: number | null;
  updated_at: string | null;
};

export type AdminSkillUpdatePayload = {
  name?: string;
  description?: string;
  triggers?: string[];
  core_markdown?: string;
  details_markdown?: string;
};
