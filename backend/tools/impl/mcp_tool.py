from __future__ import annotations

import re
from typing import Any, ClassVar

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from backend.mcp.models import MCPServerConfig, MCPToolCallResult, MCPToolDescriptor
from backend.mcp.service import MCPToolProvider

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _MCPGenericInput(BaseModel):
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments object passed to the MCP tool.",
    )


def _json_schema_type_to_python(schema: dict[str, Any]) -> Any:
    value_type = schema.get("type")
    if value_type == "string":
        return str
    if value_type == "integer":
        return int
    if value_type == "number":
        return float
    if value_type == "boolean":
        return bool
    if value_type == "array":
        return list[Any]
    if value_type == "object":
        return dict[str, Any]
    return Any


def _args_schema_for_descriptor(descriptor: MCPToolDescriptor) -> type[BaseModel]:
    schema = descriptor.input_schema or {}
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return _MCPGenericInput
    if any(not _IDENTIFIER_RE.match(str(name)) for name in properties):
        return _MCPGenericInput

    required = schema.get("required")
    required_fields = {str(item) for item in required} if isinstance(required, list) else set()
    fields: dict[str, tuple[Any, Any]] = {}
    for name, raw_property_schema in properties.items():
        property_schema = raw_property_schema if isinstance(raw_property_schema, dict) else {}
        python_type = _json_schema_type_to_python(property_schema)
        default = ... if str(name) in required_fields else None
        description = property_schema.get("description")
        fields[str(name)] = (
            python_type,
            Field(default=default, description=str(description) if description else None),
        )

    if not fields:
        return create_model(f"{descriptor.tool_key}_Input")
    return create_model(f"{descriptor.tool_key}_Input", **fields)


class MCPTool(BaseTool):
    parallel_safe: ClassVar[bool] = False

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        *,
        config: MCPServerConfig,
        descriptor: MCPToolDescriptor,
        provider: MCPToolProvider,
    ) -> None:
        super().__init__(
            name=descriptor.tool_key,
            description=descriptor.description
            or f"MCP tool {descriptor.tool_name} from server {config.display_label}.",
            args_schema=_args_schema_for_descriptor(descriptor),
        )
        object.__setattr__(self, "_mcp_config", config)
        object.__setattr__(self, "_mcp_descriptor", descriptor)
        object.__setattr__(self, "_mcp_provider", provider)

    def _run(self, *args: Any, **kwargs: Any) -> str:
        del args
        config: MCPServerConfig = object.__getattribute__(self, "_mcp_config")
        descriptor: MCPToolDescriptor = object.__getattribute__(self, "_mcp_descriptor")
        provider: MCPToolProvider = object.__getattribute__(self, "_mcp_provider")
        arguments = self._normalize_arguments(kwargs)
        result = provider.call_tool(
            config=config,
            tool_name=descriptor.tool_name,
            arguments=arguments,
        )
        if isinstance(result, MCPToolCallResult):
            return result.format_for_agent()
        return str(result)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self._run(*args, **kwargs)

    @staticmethod
    def _normalize_arguments(kwargs: dict[str, Any]) -> dict[str, Any]:
        if set(kwargs) == {"arguments"} and isinstance(kwargs.get("arguments"), dict):
            return dict(kwargs["arguments"])
        return {
            str(key): value
            for key, value in kwargs.items()
            if value is not None and not str(key).startswith("_")
        }


class MCPToolFactory:
    def __init__(
        self,
        *,
        config: MCPServerConfig,
        descriptor: MCPToolDescriptor,
        provider: MCPToolProvider,
    ) -> None:
        self.key = descriptor.tool_key
        self.description = descriptor.description or f"MCP tool {descriptor.tool_name}"
        self._config = config
        self._descriptor = descriptor
        self._provider = provider

    def is_available(self, ctx) -> bool:
        from backend.tools.policy import is_tool_allowed

        return is_tool_allowed(self.key, ctx.allowed_tool_keys)

    def build(self, ctx) -> MCPTool:
        del ctx
        return MCPTool(
            config=self._config,
            descriptor=self._descriptor,
            provider=self._provider,
        )
