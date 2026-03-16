from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
    theme: str = Field(default="dark", pattern="^(light|dark)$")
    default_include_reasoning: bool = True
    default_answer_style: str = Field(default="detailed", pattern="^(concise|detailed)$")
    analysis_depth: str = Field(default="light", pattern="^(light|medium|deep)$")
    llm_temperature_chat: float = Field(default=0.5, ge=0.0, le=2.0)
    llm_temperature_tool: float = Field(default=0.15, ge=0.0, le=2.0)
    llm_max_tokens_default: int = Field(default=1200, ge=128, le=32768)
    llm_max_tokens_reasoning: int = Field(default=2200, ge=128, le=32768)
    backend_query_timeout_sec: int = Field(default=180, ge=15, le=1800)
    agent_max_steps: int = Field(default=5, ge=2, le=20)
    agent_step_timeout_sec: int = Field(default=45, ge=5, le=600)
    agent_inner_recursion_limit: int = Field(default=6, ge=2, le=20)


class UserSettingsUpdateRequest(BaseModel):
    theme: str | None = Field(default=None, pattern="^(light|dark)$")
    default_include_reasoning: bool | None = None
    default_answer_style: str | None = Field(
        default=None,
        pattern="^(concise|detailed)$",
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
    agent_max_steps: int | None = Field(default=None, ge=2, le=20)
    agent_step_timeout_sec: int | None = Field(default=None, ge=5, le=600)
    agent_inner_recursion_limit: int | None = Field(default=None, ge=2, le=20)


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


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    use_history: bool = True
    include_reasoning: bool = False
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
    model: str


class QueryResponse(BaseModel):
    session_id: str
    text: str
    reasoning: str | None = None
    artifacts: list[ArtifactPayload]
    values: dict[str, Any] | None = None
    metrics: QueryMetrics


class SessionStateResponse(BaseModel):
    session_id: str
    title: str = "Новый чат"
    chat_history: list[dict[str, Any]]
    artifacts: list[ArtifactPayload]
    has_dataset: bool = False
