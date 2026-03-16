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

export type SessionState = {
  session_id: string;
  title: string;
  chat_history: Array<{
    role: string;
    content: string;
    timestamp: string;
    reasoning?: string | null;
    artifacts?: ArtifactPayload[];
  }>;
  artifacts: ArtifactPayload[];
  has_dataset: boolean;
};

export type SessionSummary = {
  session_id: string;
  title: string;
  created_at: string;
  last_access: string;
  has_dataset: boolean;
  last_message_preview?: string | null;
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
};

export type RuntimeModelProfile = {
  provider: string;
  model: string;
  base_url: string;
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

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string | null;
  phases?: PhaseEvent[];
  metrics?: QueryMetrics;
  artifacts?: ArtifactPayload[];
};
