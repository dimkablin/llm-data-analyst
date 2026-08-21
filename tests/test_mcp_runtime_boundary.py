from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime

from backend.agent.tool_loop import _build_tool_node
from backend.mcp.models import (
    MCPErrorCategory,
    MCPRetrySemantics,
    MCPServerConfig,
    MCPServerTransport,
    MCPToolCallResult,
    MCPToolDescriptor,
    MCPToolError,
)
from backend.tools.impl.mcp_tool import MCPTool


class _Provider:
    def __init__(self, result: Any = "ok") -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def call_tool(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _tool(provider: _Provider) -> MCPTool:
    config = MCPServerConfig(
        server_id="synthetic",
        name="Synthetic",
        transport=MCPServerTransport.streamable_http,
        url="http://127.0.0.1:9999/mcp",
    )
    descriptor = MCPToolDescriptor.from_mcp_tool(
        server_id=config.server_id,
        tool_name="project",
        capability_key="projection",
        provider_identity="synthetic-provider",
        input_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {
                "point": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["actual", "plan"]},
                        "value": {"type": ["number", "null"]},
                    },
                    "required": ["kind", "value"],
                    "additionalProperties": False,
                },
                "inline_source": {
                    "type": "object",
                    "properties": {
                        "kind": {"const": "inline"},
                        "rows": {"type": "array"},
                    },
                    "required": ["kind", "rows"],
                    "additionalProperties": False,
                },
                "table_source": {
                    "type": "object",
                    "properties": {
                        "kind": {"const": "table"},
                        "table": {"type": "string"},
                    },
                    "required": ["kind", "table"],
                    "additionalProperties": False,
                },
                "options": {
                    "type": "object",
                    "properties": {"include_plot": {"type": "boolean"}},
                    "additionalProperties": False,
                },
            },
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/point"},
                },
                "mode": {
                    "oneOf": [
                        {"const": "fast"},
                        {"const": "accurate"},
                    ]
                },
                "label": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
                "source": {
                    "oneOf": [
                        {"$ref": "#/$defs/inline_source"},
                        {"$ref": "#/$defs/table_source"},
                    ]
                },
                "options": {
                    "anyOf": [
                        {"$ref": "#/$defs/options"},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["points", "mode"],
            "allOf": [{"properties": {"mode": {"enum": ["fast", "accurate"]}}}],
            "additionalProperties": False,
        },
    )
    return MCPTool(config=config, descriptor=descriptor, provider=provider)


@pytest.mark.parametrize(
    ("payload", "expected_path"),
    [
        ({"points": [{"kind": "other", "value": 1}], "mode": "fast"}, "$.points[0].kind"),
        ({"points": [{"kind": "actual"}], "mode": "fast"}, "$.points[0]"),
        ({"points": "wrong", "mode": "fast"}, "$.points"),
        ({"points": [], "mode": "fast", "secret": 1}, "$"),
    ],
)
def test_invalid_mcp_arguments_are_rejected_before_provider(payload, expected_path) -> None:
    provider = _Provider()
    tool = _tool(provider)

    with pytest.raises(MCPToolError) as exc_info:
        tool.invoke(payload)

    assert provider.calls == []
    assert exc_info.value.details.category is MCPErrorCategory.ARGUMENT_VALIDATION
    assert exc_info.value.details.retry_semantics is MCPRetrySemantics.MODEL_CORRECTABLE
    assert exc_info.value.details.json_path == expected_path


def test_composed_ref_schema_passes_runtime_validation() -> None:
    provider = _Provider()
    tool = _tool(provider)

    assert tool.invoke(
        {
            "points": [{"kind": "actual", "value": None}],
            "mode": "accurate",
            "label": None,
        }
    ) == "ok"
    assert len(provider.calls) == 1


def test_schema_declared_structures_accept_model_serialized_json() -> None:
    provider = _Provider()
    tool = _tool(provider)

    assert tool.invoke(
        {
            "points": '[{"kind":"actual","value":null}]',
            "mode": "accurate",
            "label": '{"remains":"a string"}',
            "source": '{"kind":"inline","rows":[{"ts":"2026-01-01","y":1}]}',
            "options": '{"include_plot":true}',
        }
    ) == "ok"

    arguments = provider.calls[0]["arguments"]
    assert arguments["points"] == [{"kind": "actual", "value": None}]
    assert arguments["source"] == {
        "kind": "inline",
        "rows": [{"ts": "2026-01-01", "y": 1}],
    }
    assert arguments["options"] == {"include_plot": True}
    assert arguments["label"] == '{"remains":"a string"}'


def test_mcp_tool_description_explains_native_structured_arguments() -> None:
    tool = _tool(_Provider())

    assert "native JSON arrays and objects" in tool.description
    assert "Required top-level fields (exact names): points, mode" in tool.description
    assert '"array_field": [0.1, 0.5, 0.9]' in tool.description
    assert '"object_field": {"key": "value"}' in tool.description
    assert "serialized JSON strings" not in tool.description


def test_provider_error_is_typed_and_sanitized() -> None:
    provider = _Provider(
        MCPToolCallResult(
            structured_content={
                "message": "bad request",
                "Authorization": "Bearer secret-token",
                "api_key": "top-secret",
            },
            is_error=True,
        )
    )
    tool = _tool(provider)

    with pytest.raises(MCPToolError) as exc_info:
        tool.invoke({"points": [], "mode": "fast"})

    details = exc_info.value.details
    assert details.category is MCPErrorCategory.PROVIDER_DOMAIN
    assert details.retry_semantics is MCPRetrySemantics.MODEL_CORRECTABLE
    assert "secret-token" not in details.message
    assert "top-secret" not in details.message


def test_transport_error_has_system_retry_semantics() -> None:
    provider = _Provider(TimeoutError("Authorization: Bearer private-token"))
    tool = _tool(provider)

    with pytest.raises(MCPToolError) as exc_info:
        tool.invoke({"points": [], "mode": "fast"})

    assert exc_info.value.details.category is MCPErrorCategory.TIMEOUT
    assert exc_info.value.details.retry_semantics is MCPRetrySemantics.SYSTEM
    assert "private-token" not in exc_info.value.details.message


def test_typed_mcp_error_preserves_tool_call_id_in_tool_message() -> None:
    provider = _Provider(
        MCPToolCallResult(structured_content={"code": "bad_input"}, is_error=True)
    )
    tool = _tool(provider)
    node = _build_tool_node([tool], tool_collector=None)

    result = node.invoke(
        [
            {
                "name": tool.name,
                "args": {"points": [], "mode": "fast"},
                "id": "call-original",
                "type": "tool_call",
            }
        ],
        runtime=Runtime(),
    )
    [message] = result["messages"] if isinstance(result, dict) else result
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-original"
    assert message.status == "error"
    assert message.artifact["error"]["category"] == "provider_domain"
