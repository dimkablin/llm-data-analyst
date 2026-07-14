from __future__ import annotations

from typing import Any

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
    skill_ids: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    started_at: str


class PhoenixSpanSnapshotItem(BaseModel):
    span_id: str
    parent_id: str | None = None
    name: str
    span_kind: str
    status_code: str
    status_message: str = ""
    duration_ms: int
    start_time: str
    end_time: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    input_value: str | None = None
    output_value: str | None = None
    skill_ids: str | None = None
    attributes: dict[str, Any] = {}


class PhoenixTraceDetailResponse(BaseModel):
    trace_id: str
    project_id: str | None = None
    project_name: str | None = None
    spans: list[PhoenixSpanSnapshotItem]


class PhoenixTracesResponse(BaseModel):
    total: int
    traces: list[PhoenixTraceRow]


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


