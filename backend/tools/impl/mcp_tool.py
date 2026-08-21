from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, ClassVar

from jsonschema import ValidationError, validators
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from backend.mcp.models import (
    MCPErrorCategory,
    MCPErrorDetails,
    MCPRetrySemantics,
    MCPServerConfig,
    MCPToolCallResult,
    MCPToolDescriptor,
    MCPToolError,
)
from backend.mcp.service import MCPToolProvider
from backend.tools.artifact_references import (
    artifact_reference_names,
    load_artifact_dataframe,
    resolve_artifact_references,
)
from backend.tools.impl.db_helpers import _normalize_dataframe

if TYPE_CHECKING:
    from backend.tools.sandbox import SessionSandbox


class _MCPGenericInput(BaseModel):
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments object passed to the MCP tool.",
    )


class MCPTool(BaseTool):
    parallel_safe: ClassVar[bool] = False

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        *,
        config: MCPServerConfig,
        descriptor: MCPToolDescriptor,
        provider: MCPToolProvider,
        sandbox: SessionSandbox | None = None,
        session_id: str = "",
        session_store: Any | None = None,
        execution_store: Any | None = None,
    ) -> None:
        base_description = (
            descriptor.description or f"MCP tool {descriptor.tool_name} from server {config.display_label}."
        ).rstrip()
        required = descriptor.input_schema.get("required", [])
        required_note = (
            " Required top-level fields (exact names): " + ", ".join(map(str, required)) + "."
            if required
            else ""
        )
        super().__init__(
            name=descriptor.tool_key,
            description=(
                f"{base_description} Supply every field with the type declared by the input "
                f"schema; list and object fields are native JSON arrays and objects.{required_note} "
                'Examples: "array_field": [0.1, 0.5, 0.9] and '
                '"object_field": {"key": "value"}. For a row-array '
                'from an existing dataframe artifact, pass {"$artifact": "artifact_id"} at that '
                "array field; the runtime expands it before schema validation. Materialize any "
                "filter, rename, fill, or extension as a new table artifact before the MCP call."
            ),
            args_schema=descriptor.input_schema or _MCPGenericInput,
            response_format="content_and_artifact",
        )
        object.__setattr__(self, "_mcp_config", config)
        object.__setattr__(self, "_mcp_descriptor", descriptor)
        object.__setattr__(self, "_mcp_provider", provider)
        object.__setattr__(self, "_sandbox", sandbox)
        object.__setattr__(self, "_session_id", str(session_id or ""))
        object.__setattr__(self, "_session_store", session_store)
        object.__setattr__(self, "_execution_store", execution_store)

    def _run(self, *args: Any, **kwargs: Any) -> tuple[str, dict[str, Any] | None]:
        del args
        config: MCPServerConfig = object.__getattribute__(self, "_mcp_config")
        descriptor: MCPToolDescriptor = object.__getattribute__(self, "_mcp_descriptor")
        provider: MCPToolProvider = object.__getattribute__(self, "_mcp_provider")
        arguments = self._normalize_arguments(kwargs, descriptor.input_schema)
        sandbox: SessionSandbox | None = object.__getattribute__(self, "_sandbox")
        references = artifact_reference_names(arguments)
        available_artifacts = dict(sandbox.get_user_scope()) if sandbox is not None else {}
        source_artifact_names: list[str] = []
        source_artifact_ids: list[str] = []
        for reference in references:
            if reference in available_artifacts:
                source_artifact_names.append(reference)
                continue
            dataframe, artifact = load_artifact_dataframe(
                reference,
                session_id=object.__getattribute__(self, "_session_id"),
                session_store=object.__getattribute__(self, "_session_store"),
                execution_store=object.__getattribute__(self, "_execution_store"),
            )
            available_artifacts[reference] = dataframe
            source_artifact_ids.append(str(artifact.id))
        arguments = resolve_artifact_references(
            arguments,
            artifacts=available_artifacts,
        )
        self._validate_arguments(arguments, descriptor.input_schema)
        try:
            result = provider.call_tool(
                config=config,
                tool_name=descriptor.tool_name,
                arguments=arguments,
            )
        except MCPToolError:
            raise
        except Exception as exc:
            raise MCPToolError(self._provider_exception_details(exc)) from exc
        if isinstance(result, MCPToolCallResult):
            if result.is_error:
                raise MCPToolError(
                    MCPErrorDetails(
                        category=MCPErrorCategory.PROVIDER_DOMAIN,
                        code="mcp_provider_error",
                        message=self._sanitize_payload(
                            result.structured_content
                            if result.structured_content is not None
                            else result.content
                        ),
                        retry_semantics=MCPRetrySemantics.MODEL_CORRECTABLE,
                    )
                )
            content = result.format_for_agent()
            structured = result.structured_content
            if structured is None:
                return content, None
            derived_artifacts = self._publish_structured_result(
                descriptor.tool_key,
                structured,
                sandbox,
            )
            published_names = [
                name
                for artifact_type in ("table", "plot")
                for name in derived_artifacts.get(artifact_type, {})
            ]
            if published_names:
                names = ", ".join(f"`{name}`" for name in published_names)
                content += (
                    f"\n\nPublished sandbox artifacts: {names}. Use these names directly in "
                    "downstream tools; do not copy their rows into code."
                )
            capability_key = descriptor.capability_key or f"mcp_tool:{descriptor.tool_key}"
            return content, {
                "schema_version": "1.0",
                "artifact_type": "json",
                "items": {descriptor.tool_key: structured},
                **derived_artifacts,
                "meta": {
                    "capability_key": capability_key,
                    "bound_tool_key": descriptor.tool_key,
                    "runtime_route": "mcp",
                    "provider": config.server_id,
                    "status": "ok",
                    **(
                        {
                            "lineage": {
                                **(
                                    {"source_artifact_names": source_artifact_names}
                                    if source_artifact_names
                                    else {}
                                ),
                                **(
                                    {"source_artifact_ids": source_artifact_ids}
                                    if source_artifact_ids
                                    else {}
                                ),
                            }
                        }
                        if source_artifact_names or source_artifact_ids
                        else {}
                    ),
                },
                "provenance": {
                    "source_type": "mcp",
                    "source_ref_id": config.server_id,
                    "source_label": config.display_label,
                    "source_mode": "model_inference",
                },
            }
        return str(result), None

    @staticmethod
    def _publish_structured_result(
        tool_key: str,
        structured: Any,
        sandbox: SessionSandbox | None,
    ) -> dict[str, dict[str, Any]]:
        artifacts: dict[str, dict[str, Any]] = {}
        if sandbox is None:
            return artifacts
        sandbox.put(tool_key, structured)
        if not isinstance(structured, dict):
            return artifacts

        rows = structured.get("rows")
        if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
            import pandas as pd

            name = f"{tool_key}_rows"
            table = _normalize_dataframe(pd.DataFrame(rows))
            sandbox.put(name, table)
            artifacts["table"] = {name: table}

        plot = structured.get("plot")
        figure = plot.get("figure") if isinstance(plot, dict) else None
        if isinstance(figure, dict):
            import plotly.graph_objects as go

            try:
                name = f"{tool_key}_plot_figure"
                plot_figure = go.Figure(figure)
                sandbox.put(name, plot_figure)
                artifacts["plot"] = {name: plot_figure}
            except (TypeError, ValueError):
                pass
        return artifacts

    async def _arun(self, *args: Any, **kwargs: Any) -> tuple[str, dict[str, Any] | None]:
        return self._run(*args, **kwargs)

    @classmethod
    def _normalize_arguments(
        cls,
        kwargs: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        if set(kwargs) == {"arguments"} and isinstance(kwargs.get("arguments"), dict):
            arguments = dict(kwargs["arguments"])
        else:
            arguments = {
                str(key): value
                for key, value in kwargs.items()
                if value is not None and not str(key).startswith("_")
            }
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        return {
            key: cls._decode_declared_structure(value, properties.get(key, {}), schema)
            for key, value in arguments.items()
        }

    @classmethod
    def _decode_declared_structure(
        cls,
        value: Any,
        field_schema: Any,
        root_schema: dict[str, Any],
    ) -> Any:
        if not isinstance(value, str):
            return value
        declared_types = cls._declared_json_types(field_schema, root_schema)
        if "string" in declared_types or not declared_types.intersection({"array", "object"}):
            return value
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return value
        if isinstance(decoded, list) and "array" in declared_types:
            return decoded
        if isinstance(decoded, dict) and "object" in declared_types:
            return decoded
        return value

    @classmethod
    def _declared_json_types(
        cls,
        schema: Any,
        root_schema: dict[str, Any],
        seen_refs: frozenset[str] = frozenset(),
    ) -> set[str]:
        if not isinstance(schema, dict):
            return set()
        raw_type = schema.get("type")
        declared = {raw_type} if isinstance(raw_type, str) else set(raw_type or [])
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/") and ref not in seen_refs:
            target: Any = root_schema
            for part in ref[2:].split("/"):
                target = target.get(part.replace("~1", "/").replace("~0", "~"), {})
            declared.update(cls._declared_json_types(target, root_schema, seen_refs | {ref}))
        for keyword in ("anyOf", "oneOf", "allOf"):
            for branch in schema.get(keyword, []):
                declared.update(cls._declared_json_types(branch, root_schema, seen_refs))
        return declared

    @staticmethod
    def _validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
        effective_schema = dict(schema or {"type": "object"})
        validator_class = validators.validator_for(effective_schema)
        validator_class.check_schema(effective_schema)
        errors = sorted(
            validator_class(effective_schema).iter_errors(arguments),
            key=lambda item: (list(item.absolute_path), item.message),
        )
        if not errors:
            return
        error = errors[0]
        raise MCPToolError(
            MCPErrorDetails(
                category=MCPErrorCategory.ARGUMENT_VALIDATION,
                code="mcp_arguments_invalid",
                message=MCPTool._sanitize_text(error.message),
                retry_semantics=MCPRetrySemantics.MODEL_CORRECTABLE,
                json_path=MCPTool._json_path(error),
                schema_path="/" + "/".join(str(item) for item in error.absolute_schema_path),
            )
        )

    @staticmethod
    def _json_path(error: ValidationError) -> str:
        path = "$"
        for item in error.absolute_path:
            path += f"[{item}]" if isinstance(item, int) else f".{item}"
        return path

    @staticmethod
    def _provider_exception_details(exc: Exception) -> MCPErrorDetails:
        text = str(exc or "")
        lowered = text.casefold()
        if isinstance(exc, TimeoutError):
            category = MCPErrorCategory.TIMEOUT
            code = "mcp_timeout"
            retry = MCPRetrySemantics.SYSTEM
        elif "401" in lowered or "unauthenticated" in lowered or "authentication" in lowered:
            category = MCPErrorCategory.AUTHENTICATION
            code = "mcp_authentication_failed"
            retry = MCPRetrySemantics.TERMINAL
        elif "403" in lowered or "forbidden" in lowered or "authorization" in lowered:
            category = MCPErrorCategory.AUTHORIZATION
            code = "mcp_authorization_failed"
            retry = MCPRetrySemantics.TERMINAL
        elif "protocol" in lowered or "json-rpc" in lowered:
            category = MCPErrorCategory.PROTOCOL
            code = "mcp_protocol_error"
            retry = MCPRetrySemantics.TERMINAL
        else:
            category = MCPErrorCategory.TRANSPORT
            code = "mcp_transport_error"
            retry = MCPRetrySemantics.SYSTEM
        return MCPErrorDetails(
            category=category,
            code=code,
            message=MCPTool._sanitize_text(text) or "MCP provider call failed.",
            retry_semantics=retry,
        )

    @staticmethod
    def _sanitize_payload(payload: Any) -> str:
        sensitive = {"authorization", "proxy-authorization", "token", "access_token", "api_key"}

        def redact(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    str(key): "[REDACTED]" if str(key).casefold() in sensitive else redact(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [redact(item) for item in value]
            return value

        return MCPTool._sanitize_text(json.dumps(redact(payload), ensure_ascii=False, default=str))

    @staticmethod
    def _sanitize_text(value: str) -> str:
        text = str(value or "")
        text = re.sub(
            r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;}\]]+",
            r"\1[REDACTED]",
            text,
        )
        text = re.sub(
            r"(?i)((?:api[_-]?key|access[_-]?token|token)\s*[:=]\s*)[^\s,;}\]]+",
            r"\1[REDACTED]",
            text,
        )
        return text[:1200]


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

    @property
    def descriptor(self) -> MCPToolDescriptor:
        return self._descriptor

    @property
    def config(self) -> MCPServerConfig:
        return self._config

    def is_available(self, ctx) -> bool:
        from backend.tools.policy import is_tool_allowed

        return is_tool_allowed(self.key, ctx.allowed_tool_keys)

    def build(self, ctx) -> MCPTool:
        return MCPTool(
            config=self._config,
            descriptor=self._descriptor,
            provider=self._provider,
            sandbox=getattr(ctx, "sandbox", None),
            session_id=str(getattr(ctx, "trace_context", {}).get("session_id") or ""),
            session_store=getattr(ctx, "session_store", None),
            execution_store=getattr(ctx, "execution_store", None),
        )
