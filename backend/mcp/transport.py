from __future__ import annotations

import threading
from datetime import timedelta
from typing import Any

import anyio

from backend.mcp.models import (
    MCPServerConfig,
    MCPServerTransport,
    MCPToolCallResult,
    MCPToolDescriptor,
)


class SDKMCPToolProvider:
    """Synchronous facade over the official async MCP Python SDK."""

    def list_tools(self, config: MCPServerConfig) -> list[MCPToolDescriptor]:
        return self._run_blocking(self._list_tools, config)

    def call_tool(
        self,
        *,
        config: MCPServerConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolCallResult:
        return self._run_blocking(self._call_tool, config, tool_name, arguments)

    @staticmethod
    def _run_blocking(async_fn, *args):
        try:
            anyio.get_current_task()
        except RuntimeError:
            return anyio.run(async_fn, *args)

        result: dict[str, Any] = {}
        error: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                result["value"] = anyio.run(async_fn, *args)
            except BaseException as exc:
                error["value"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "value" in error:
            raise error["value"]
        return result.get("value")

    async def _list_tools(self, config: MCPServerConfig) -> list[MCPToolDescriptor]:
        async with self._session(config) as session:
            result = await session.list_tools()
            descriptors: list[MCPToolDescriptor] = []
            for tool in result.tools:
                output_schema = tool.outputSchema
                descriptors.append(
                    MCPToolDescriptor.from_mcp_tool(
                        server_id=config.server_id,
                        tool_name=str(tool.name),
                        description=tool.description,
                        input_schema=dict(tool.inputSchema or {}),
                        output_schema=dict(output_schema) if output_schema is not None else None,
                    )
                )
            return descriptors

    async def _call_tool(
        self,
        config: MCPServerConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolCallResult:
        async with self._session(config) as session:
            result = await session.call_tool(
                tool_name,
                arguments=arguments,
                read_timeout_seconds=timedelta(seconds=config.timeout_sec),
            )
            return MCPToolCallResult(
                content=list(result.content or []),
                structured_content=getattr(result, "structuredContent", None),
                is_error=bool(getattr(result, "isError", False)),
            )

    def _session(self, config: MCPServerConfig):
        if config.transport == MCPServerTransport.streamable_http:
            return self._http_session(config)
        if config.transport == MCPServerTransport.stdio:
            return self._stdio_session(config)
        raise ValueError(f"Unsupported MCP transport: {config.transport}")

    @staticmethod
    def _http_session(config: MCPServerConfig):
        from contextlib import asynccontextmanager

        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        @asynccontextmanager
        async def _connect():
            async with streamablehttp_client(
                str(config.url),
                timeout=config.timeout_sec,
                sse_read_timeout=max(config.timeout_sec, 300.0),
            ) as (read_stream, write_stream, _get_session_id):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session

        return _connect()

    @staticmethod
    def _stdio_session(config: MCPServerConfig):
        from contextlib import asynccontextmanager

        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        @asynccontextmanager
        async def _connect():
            params = StdioServerParameters(
                command=str(config.command),
                args=list(config.args),
                env=dict(config.env) or None,
            )
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session

        return _connect()
