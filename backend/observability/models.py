from __future__ import annotations

from pydantic import BaseModel, Field


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


