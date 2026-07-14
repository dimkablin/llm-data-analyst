from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SERVER_ID_PATTERN = r"^[A-Za-z0-9_.-]{1,80}$"
_TOOL_KEY_PART_RE = re.compile(r"[^A-Za-z0-9_]+")


class MCPServerTransport(StrEnum):
    streamable_http = "streamable_http"
    stdio = "stdio"


def _tool_key_part(value: str) -> str:
    normalized = _TOOL_KEY_PART_RE.sub("_", value.strip()).strip("_").lower()
    return normalized or "unnamed"


def mcp_tool_key(server_id: str, tool_name: str) -> str:
    return f"mcp__{_tool_key_part(server_id)}__{_tool_key_part(tool_name)}"


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str = Field(pattern=_SERVER_ID_PATTERN)
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    transport: MCPServerTransport
    url: str | None = Field(default=None, max_length=2048)
    command: str | None = Field(default=None, max_length=512)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_sec: float = Field(default=30.0, ge=1.0, le=300.0)
    enabled: bool = True
    enabled_by_default: bool = True
    created_at: str | None = None
    updated_at: str | None = None
    updated_by: int | None = None

    @field_validator("server_id", mode="before")
    @classmethod
    def _normalize_server_id(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: Any) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def _validate_transport_payload(self) -> MCPServerConfig:
        if self.transport == MCPServerTransport.streamable_http:
            parsed = urlparse(str(self.url or "").strip())
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("streamable_http MCP server requires an http(s) url")
        if self.transport == MCPServerTransport.stdio and not str(self.command or "").strip():
            raise ValueError("stdio MCP server requires a command")
        return self

    @property
    def display_label(self) -> str:
        return self.name.strip() or self.server_id


class MCPServerCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str = Field(pattern=_SERVER_ID_PATTERN)
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    transport: MCPServerTransport = MCPServerTransport.streamable_http
    url: str | None = Field(default=None, max_length=2048)
    command: str | None = Field(default=None, max_length=512)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_sec: float = Field(default=30.0, ge=1.0, le=300.0)
    enabled: bool = True
    enabled_by_default: bool = True

    @model_validator(mode="after")
    def _validate_as_config(self) -> MCPServerCreateRequest:
        MCPServerConfig(**self.model_dump())
        return self


class MCPServerUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    transport: MCPServerTransport | None = None
    url: str | None = Field(default=None, max_length=2048)
    command: str | None = Field(default=None, max_length=512)
    args: list[str] | None = None
    env: dict[str, str] | None = None
    timeout_sec: float | None = Field(default=None, ge=1.0, le=300.0)
    enabled: bool | None = None
    enabled_by_default: bool | None = None


class MCPServerEnabledUpdateRequest(BaseModel):
    enabled: bool


class MCPToolDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str = Field(pattern=_SERVER_ID_PATTERN)
    tool_name: str = Field(min_length=1, max_length=200)
    tool_key: str = Field(min_length=1, max_length=260)
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None

    @classmethod
    def from_mcp_tool(
        cls,
        *,
        server_id: str,
        tool_name: str,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> MCPToolDescriptor:
        return cls(
            server_id=server_id,
            tool_name=tool_name,
            tool_key=mcp_tool_key(server_id, tool_name),
            description=description,
            input_schema=dict(input_schema or {"type": "object"}),
            output_schema=dict(output_schema) if output_schema is not None else None,
        )


class MCPToolCallResult(BaseModel):
    content: list[Any] = Field(default_factory=list)
    structured_content: Any | None = None
    is_error: bool = False

    def format_for_agent(self) -> str:
        parts: list[str] = []
        if self.structured_content is not None:
            parts.append(json.dumps(self.structured_content, ensure_ascii=False))
        for item in self.content:
            if isinstance(item, str):
                parts.append(item)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            if hasattr(item, "model_dump"):
                parts.append(json.dumps(item.model_dump(), ensure_ascii=False))
            else:
                parts.append(json.dumps(item, ensure_ascii=False, default=str))
        body = "\n".join(part for part in parts if part.strip()).strip()
        if not body:
            body = "MCP tool returned no content."
        if self.is_error:
            return f"MCP tool returned an error: {body}"
        return body


class MCPServerCatalogItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str
    name: str
    description: str | None = None
    transport: MCPServerTransport
    enabled_globally: bool
    available_globally: bool
    status: str
    enabled_by_default: bool
    enabled_for_user: bool
    effective_enabled: bool
    tools: list[MCPToolDescriptor] = Field(default_factory=list)
    tool_count: int = 0
    last_error: str | None = None


class AdminMCPServerConfigResponse(MCPServerConfig):
    pass
