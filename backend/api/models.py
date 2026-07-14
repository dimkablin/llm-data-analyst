from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.auth.user_settings_defaults import DEFAULT_USER_SETTINGS
from backend.core.config import DEPTH_MAX_STEPS

_MAX_STEPS = max(DEPTH_MAX_STEPS.values())


class AuthRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=4, max_length=256)


class AuthLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


class AuthUserResponse(BaseModel):
    id: int
    username: str
    is_admin: bool
    created_at: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUserResponse


class AuthChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=4, max_length=256)


class UserSettingsResponse(BaseModel):
    theme: str = Field(default=DEFAULT_USER_SETTINGS.theme, pattern="^(light|dark)$")
    default_include_reasoning: bool = DEFAULT_USER_SETTINGS.default_include_reasoning
    default_answer_style: str = Field(
        default=DEFAULT_USER_SETTINGS.default_answer_style,
        pattern="^(concise|detailed)$",
    )
    analysis_mode: str = Field(
        default=DEFAULT_USER_SETTINGS.analysis_mode, pattern="^(fast|deep)$"
    )
    analysis_depth: str = Field(
        default=DEFAULT_USER_SETTINGS.analysis_depth, pattern="^(light|medium|deep)$"
    )
    llm_temperature_chat: float = Field(
        default=DEFAULT_USER_SETTINGS.llm_temperature_chat, ge=0.0, le=2.0
    )
    llm_temperature_tool: float = Field(
        default=DEFAULT_USER_SETTINGS.llm_temperature_tool, ge=0.0, le=2.0
    )
    llm_max_tokens_default: int = Field(
        default=DEFAULT_USER_SETTINGS.llm_max_tokens_default, ge=128, le=32768
    )
    llm_max_tokens_reasoning: int = Field(
        default=DEFAULT_USER_SETTINGS.llm_max_tokens_reasoning, ge=128, le=32768
    )
    backend_query_timeout_sec: int = Field(
        default=DEFAULT_USER_SETTINGS.backend_query_timeout_sec, ge=15, le=1800
    )
    agent_max_steps: int = Field(default=DEPTH_MAX_STEPS["light"], ge=2, le=_MAX_STEPS)
    agent_step_timeout_sec: int = Field(
        default=DEFAULT_USER_SETTINGS.agent_step_timeout_sec, ge=5, le=600
    )
    agent_inner_recursion_limit: int = Field(
        default=DEFAULT_USER_SETTINGS.agent_inner_recursion_limit, ge=2, le=_MAX_STEPS
    )
    agent_react_enabled: bool = DEFAULT_USER_SETTINGS.agent_react_enabled
    ui_scale: int = Field(default=DEFAULT_USER_SETTINGS.ui_scale, ge=70, le=150)
    llm_streaming: bool = DEFAULT_USER_SETTINGS.llm_streaming
    show_thinking: bool = DEFAULT_USER_SETTINGS.show_thinking
    show_think_planning: bool = DEFAULT_USER_SETTINGS.show_think_planning
    show_think_tool: bool = DEFAULT_USER_SETTINGS.show_think_tool
    show_think_final: bool = DEFAULT_USER_SETTINGS.show_think_final
    show_detailed_tool_steps: bool = DEFAULT_USER_SETTINGS.show_detailed_tool_steps


class UserSettingsUpdateRequest(BaseModel):
    theme: str | None = Field(default=None, pattern="^(light|dark)$")
    default_include_reasoning: bool | None = None
    default_answer_style: str | None = Field(
        default=None,
        pattern="^(concise|detailed)$",
    )
    analysis_mode: str | None = Field(
        default=None,
        pattern="^(fast|deep|demo)$",
    )
    analysis_depth: str | None = Field(
        default=None,
        pattern="^(light|medium|deep)$",
    )
    llm_temperature_chat: float | None = Field(default=None, ge=0.0, le=2.0)
    llm_temperature_tool: float | None = Field(default=None, ge=0.0, le=2.0)
    llm_max_tokens_default: int | None = Field(default=None, ge=128, le=32768)
    llm_max_tokens_reasoning: int | None = Field(default=None, ge=128, le=32768)
    backend_query_timeout_sec: int | None = Field(default=None, ge=15, le=1800)
    agent_max_steps: int | None = Field(default=None, ge=2, le=_MAX_STEPS)
    agent_step_timeout_sec: int | None = Field(default=None, ge=5, le=600)
    agent_inner_recursion_limit: int | None = Field(default=None, ge=2, le=_MAX_STEPS)
    agent_react_enabled: bool | None = None
    ui_scale: int | None = Field(default=None, ge=70, le=150)
    llm_streaming: bool | None = None
    show_thinking: bool | None = None
    show_think_planning: bool | None = None
    show_think_tool: bool | None = None
    show_think_final: bool | None = None
    show_detailed_tool_steps: bool | None = None


class ToolEnabledUpdateRequest(BaseModel):
    enabled: bool


class ToolAvailabilityResponse(BaseModel):
    tool_key: str
    kind: str
    tool_label: str
    display_name_ru: str | None = None
    description: str | None = None
    description_ru: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    requires_session_data: bool = False
    source_type: str | None = None
    source_ref_id: str | None = None
    source_mode: str | None = None
    enabled_globally: bool
    available_globally: bool
    status: str
    enabled_by_default: bool = True
    enabled_for_user: bool
    effective_enabled: bool
    timeout_hint_sec: float | None = None


class AdminCreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=4, max_length=256)
    is_admin: bool = False


class AdminUpdateUserRequest(BaseModel):
    password: str | None = Field(default=None, min_length=4, max_length=256)
    is_admin: bool | None = None


class MessageResponse(BaseModel):
    ok: bool = True
    message: str


class RuntimeModelProfileResponse(BaseModel):
    provider: str
    model: str
    base_url: str
    max_context_tokens: int | None = Field(default=None, ge=1)
    context_window_source: str = "unavailable"


class SessionSummaryResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    last_access: str
    has_dataset: bool
    last_message_preview: str | None = None


class SessionTitleUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class CreateSessionResponse(BaseModel):
    session_id: str


class UploadResponse(BaseModel):
    session_id: str
    rows: int
    columns: int


class TabularPreprocessingSummaryResponse(BaseModel):
    enabled: bool
    raw_rows: int
    raw_columns: int
    cleaned_rows: int
    cleaned_columns: int
    detected_header_row: int | None = None
    removed_rows: int
    removed_columns: int


class UploadedTableResponse(BaseModel):
    file_name: str
    file_format: str
    table_name: str
    source_alias: str
    variable_name: str
    parquet_path: str
    rows: int
    columns: int
    preprocessing: TabularPreprocessingSummaryResponse


class BatchUploadResponse(BaseModel):
    session_id: str
    csv_session_id: str
    table_names: list[str] = Field(default_factory=list)
    files: list[UploadedTableResponse] = Field(default_factory=list)
    expires_at: int
    total_rows: int
    total_columns: int
    dataset_name: str


class RagDocumentUploadResponse(BaseModel):
    status: str
    message: str = ""
    track_id: str


class RagDocumentStatusResponse(BaseModel):
    id: str | None = None
    file_path: str | None = None
    status: str
    track_id: str | None = None
    chunks_count: int | None = None
    content_length: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    error_msg: str | None = None


class RagTrackStatusResponse(BaseModel):
    track_id: str
    documents: list[RagDocumentStatusResponse] = Field(default_factory=list)
    status_summary: dict[str, int] = Field(default_factory=dict)


class RagDocumentsResponse(BaseModel):
    documents: list[RagDocumentStatusResponse] = Field(default_factory=list)


class RagDocumentDeleteResponse(BaseModel):
    status: str
    message: str = ""
    document_id: str


class SessionSourceStateResponse(BaseModel):
    source_type: str | None = None
    source_ref_id: str | None = None
    source_label: str | None = None
    source_mode: str | None = None


class SourceDescriptorResponse(BaseModel):
    source_type: str
    source_ref_id: str | None = None
    source_label: str
    display_name_ru: str | None = None
    source_mode: str | None = None
    enabled: bool
    available: bool
    status: str = Field(default="disabled")
    description: str | None = None
    description_ru: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    requires_session_data: bool = False
    timeout_hint_sec: float | None = None


class SessionBindDBConnectionSourceRequest(BaseModel):
    connection_id: str = Field(..., min_length=1, max_length=120)
    source_mode: str | None = Field(default=None, max_length=40)


class OpenProjectSyncRequest(BaseModel):
    base_url: str | None = Field(default=None, max_length=500)
    api_key: str | None = Field(default=None, max_length=4096)
    host_header: str | None = Field(default=None, max_length=255)
    project: str | None = Field(default=None, max_length=255)
    all_projects: bool | None = None
    days: int | None = Field(default=None, ge=1, le=3650)
    max_items: int | None = Field(default=None, ge=0, le=1_000_000)


class OpenProjectProjectResponse(BaseModel):
    id: str
    identifier: str
    name: str


class OpenProjectProjectsResponse(BaseModel):
    projects: list[OpenProjectProjectResponse] = Field(default_factory=list)


class OpenProjectSyncResponse(BaseModel):
    source_type: str = "openproject"
    source_ref_id: str
    source_label: str
    source_mode: str = "postgres_sync"
    connection_id: str
    connection_name: str
    schema_name: str
    tables: dict[str, int] = Field(default_factory=dict)
    artifact_ids: list[str] = Field(default_factory=list)
    synced_at: str


class DBConnectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    db_type: str = Field(..., min_length=2, max_length=40)
    host: str = Field(..., min_length=1, max_length=255)
    port: int | None = Field(default=None, ge=1, le=65535)
    database: str | None = Field(default=None, max_length=255)
    username: str | None = Field(default=None, max_length=255)
    password: str | None = Field(default=None, max_length=4096)
    options_json: dict[str, Any] | None = None


class DBConnectionUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    db_type: str | None = Field(default=None, min_length=2, max_length=40)
    host: str | None = Field(default=None, min_length=1, max_length=255)
    port: int | None = Field(default=None, ge=1, le=65535)
    database: str | None = Field(default=None, max_length=255)
    username: str | None = Field(default=None, max_length=255)
    password: str | None = Field(default=None, max_length=4096)
    clear_password: bool = False
    options_json: dict[str, Any] | None = None


class DBConnectionResponse(BaseModel):
    id: str
    name: str
    db_type: str
    host: str
    port: int | None = None
    database: str | None = None
    username: str | None = None
    options_json: dict[str, Any] | None = None
    password_present: bool
    last_test_at: str | None = None
    last_test_ok: bool | None = None
    last_error: str | None = None
    created_at: str
    updated_at: str


class DBConnectionTestResponse(BaseModel):
    ok: bool
    checked_at: str
    last_test_at: str
    last_test_ok: bool
    error: str | None = None


class DBConnectionSchemaResponse(BaseModel):
    name: str
    display_name: str


class DBConnectionTableResponse(BaseModel):
    schema: str
    name: str
    table_type: str
    qualified_name: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    use_history: bool = True
    include_reasoning: bool = False
    selected_skill_ids: list[str] | None = None
    analysis_depth: str | None = Field(
        default=None,
        pattern="^(light|medium|deep)$",
    )


class ArtifactPayload(BaseModel):
    id: str
    type: str
    text: str | None = None
    role: str
    meta: dict[str, Any] = Field(default_factory=dict)
    timestamp: str
    data: dict[str, Any]


class QueryMetrics(BaseModel):
    duration_ms: int
    artifact_count: int
    table_count: int
    plot_count: int
    value_count: int
    json_count: int = 0
    model: str


class QueryResponse(BaseModel):
    session_id: str
    text: str
    reasoning: str | None = None
    artifacts: list[ArtifactPayload]
    values: dict[str, Any] | None = None
    metrics: QueryMetrics
    persistence_failed: bool = (
        False  # True when post-processing (store/artifact persistence) failed
    )


class SessionSourceResponse(BaseModel):
    """Single source in the multi-source list."""

    alias: str
    source_type: str
    display_name: str = ""
    variable_name: str = ""
    file_name: str | None = None
    connection_id: str | None = None
    connection_name: str | None = None
    bound_at: str = ""
    csv_table_names: list[str] = Field(default_factory=list)
    schema_hint: dict[str, str] = Field(default_factory=dict)


class SessionStateResponse(BaseModel):
    session_id: str
    title: str = "Новый чат"
    chat_history: list[dict[str, Any]]
    artifacts: list[ArtifactPayload]
    has_dataset: bool = False
    dataset_name: str | None = None
    source_type: str | None = None
    source_ref_id: str | None = None
    source_label: str | None = None
    source_mode: str | None = None
    selected_skill_ids: list[str] = Field(default_factory=list)
    sources: list[SessionSourceResponse] = Field(default_factory=list)
    session_memory: str = ""
    context_usage: dict[str, Any] | None = None


class SkillResponse(BaseModel):
    skill_id: str
    name: str
    description: str
    triggers: list[str] = Field(default_factory=list)
    source_path: str
    enabled_by_default: bool = True
    enabled_for_user: bool = True


class AdminSkillResponse(BaseModel):
    skill_id: str
    name: str
    description: str
    triggers: list[str] = Field(default_factory=list)
    source_path: str
    enabled_by_default: bool = True
    kind: str
    tool_key: str | None = None
    core_markdown: str
    details_markdown: str | None = None
    is_overridden: bool = False
    updated_by: int | None = None
    updated_at: str | None = None


class AdminSkillUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    triggers: list[str] | None = None
    core_markdown: str | None = None
    details_markdown: str | None = None


class PhoenixOverviewStats(BaseModel):
    total_traces: int
    success_rate: float
    p50_latency_ms: int
    unique_sessions: int


class PhoenixLatencyPoint(BaseModel):
    label: str
    p50_ms: int
    p95_ms: int
    p99_ms: int
    trace_count: int


class PhoenixTokenUsageRow(BaseModel):
    trace_id: str
    session_id: str | None = None
    query_preview: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    duration_ms: int
    started_at: str
    token_source: str = "unavailable"


class PhoenixTraceRow(BaseModel):
    trace_id: str
    session_id: str | None = None
    query_preview: str
    request_kind: str
    user: str | None = None
    status: str
    duration_ms: int
    tool_calls: int
    span_count: int
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    started_at: str


class PhoenixOverviewResponse(BaseModel):
    available: bool
    project_name: str
    project_id: str | None = None
    generated_at: str
    dashboard_url: str | None = None
    embed_url: str | None = None
    stats: PhoenixOverviewStats
    latency: list[PhoenixLatencyPoint]
    token_usage: list[PhoenixTokenUsageRow]
    traces: list[PhoenixTraceRow]
    warnings: list[str] = Field(default_factory=list)


# ── User memory ──────────────────────────────────────────────────────────────


class UserMemoryResponse(BaseModel):
    profile: str
    notes: str


class UserMemoryUpdateRequest(BaseModel):
    profile: str | None = None
    notes: str | None = None
