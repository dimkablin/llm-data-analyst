from __future__ import annotations

import json
import logging
import os
import secrets
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import ValidationError

from chronos_mcp.adapters import (
    ChronosRuntime,
    ChronosSettings,
    DataGatewaySettings,
    HttpDataGateway,
)
from chronos_mcp.application import ChronosApplication, ModelRuntime, error_response
from chronos_mcp.contracts import (
    BacktestEvaluation,
    BacktestOutput,
    BacktestRequest,
    CapabilitiesResponse,
    Covariates,
    FilterExpression,
    ForecastOptions,
    ForecastOutput,
    ForecastRequest,
    Frequency,
    InlineSource,
    MissingPolicy,
    Target,
)
from chronos_mcp.errors import ChronosMCPError, ErrorCode
from chronos_mcp.preparation import TableReader

logger = logging.getLogger(__name__)
Transport = Literal["stdio", "streamable-http"]
READ_ONLY_TOOL = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


@dataclass(frozen=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8810
    transport: Transport = "streamable-http"
    log_level: str = "INFO"
    api_key: str | None = field(default=None, repr=False)
    public_url: str = "http://127.0.0.1:8810/mcp"

    @classmethod
    def from_env(cls) -> ServerSettings:
        transport = os.getenv("CHRONOS_MCP_TRANSPORT", cls.transport)
        if transport not in {"stdio", "streamable-http"}:
            raise ValueError("CHRONOS_MCP_TRANSPORT must be stdio or streamable-http")
        host = os.getenv("CHRONOS_MCP_HOST", cls.host)
        port = _port_from_env()
        return cls(
            host=host,
            port=port,
            transport=transport,
            log_level=os.getenv("CHRONOS_MCP_LOG_LEVEL", cls.log_level).upper(),
            api_key=_api_key_from_env(transport),
            public_url=os.getenv("CHRONOS_MCP_PUBLIC_URL", f"http://{host}:{port}/mcp"),
        )


def _port_from_env() -> int:
    try:
        return int(os.getenv("CHRONOS_MCP_PORT", str(ServerSettings.port)))
    except ValueError as exc:
        raise ValueError("CHRONOS_MCP_PORT must be an integer") from exc


def _api_key_from_env(transport: Transport) -> str | None:
    if transport == "stdio":
        return None
    secret_file = os.getenv("CHRONOS_MCP_API_KEY_FILE")
    try:
        api_key = Path(secret_file).read_text(encoding="utf-8").strip() if secret_file else ""
    except OSError as exc:
        raise ValueError("CHRONOS_MCP_API_KEY_FILE cannot be read") from exc
    api_key = api_key or os.getenv("CHRONOS_MCP_API_KEY", "").strip()
    if not api_key:
        raise ValueError("CHRONOS_MCP_API_KEY or CHRONOS_MCP_API_KEY_FILE is required for HTTP")
    return api_key


class _ApiKeyVerifier:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key.encode()

    async def verify_token(self, token: str) -> AccessToken | None:
        if not secrets.compare_digest(token.encode(), self._api_key):
            return None
        return AccessToken(token=token, client_id="static-api-key", scopes=[])


def create_server(
    *,
    runtime: ModelRuntime | None = None,
    table_reader: TableReader | None = None,
    settings: ServerSettings | None = None,
) -> FastMCP:
    server_settings = settings or ServerSettings.from_env()
    if server_settings.transport == "streamable-http" and not server_settings.api_key:
        raise ValueError("api_key is required for streamable-http")
    resolved_runtime = runtime or ChronosRuntime(ChronosSettings.from_env())
    resolved_reader = table_reader or _data_gateway_from_env()
    application = ChronosApplication(
        runtime=resolved_runtime,
        table_reader=resolved_reader,
    )
    server = FastMCP(
        name="Chronos Forecasting",
        instructions=(
            "Route requests that predict, forecast, project, or extrapolate future "
            "time-series values through forecast. Use backtest to measure historical "
            "forecast quality and capabilities for model-specific limits. The caller "
            "prepares compact typed time-series rows and supplies the declared forecast "
            "fields."
        ),
        host=server_settings.host,
        port=server_settings.port,
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        log_level=_mcp_log_level(server_settings.log_level),
        token_verifier=_token_verifier(server_settings),
        auth=_auth_settings(server_settings),
    )
    _register_tools(server, application)
    return server


def _token_verifier(settings: ServerSettings) -> _ApiKeyVerifier | None:
    return _ApiKeyVerifier(settings.api_key) if settings.api_key else None


def _auth_settings(settings: ServerSettings) -> AuthSettings | None:
    if not settings.api_key:
        return None
    return AuthSettings(
        issuer_url=settings.public_url,
        resource_server_url=settings.public_url,
    )


def _data_gateway_from_env() -> TableReader | None:
    settings = DataGatewaySettings.from_env()
    return HttpDataGateway(settings) if settings is not None else None


def _mcp_log_level(value: str) -> Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    normalized = value if value in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO"
    return normalized  # type: ignore[return-value]


def _register_tools(server: FastMCP, application: ChronosApplication) -> None:
    @server.tool(
        name="forecast",
        title="Chronos forecast",
        description=(
            "Use for every request to predict or forecast future time-series values, "
            "including projections, extrapolation, and forecasts for the next N periods. "
            "Historical summaries use the active source-analysis actions. The caller "
            "first prepares compact time-series rows from the active data source: clean "
            "required values, aggregate to the requested frequency, order periods, and "
            "select missing_policy from the metric semantics and source completeness. "
            "Pass the prepared rows as source.kind='inline' with time_column, targets, "
            "horizon, frequency, and missing_policy. The structured result provides "
            "forecast rows, uncertainty intervals, and, when options.include_plot is "
            "true, a ready-to-use Plotly figure."
        ),
        annotations=READ_ONLY_TOOL,
        structured_output=True,
    )
    def forecast(
        source: InlineSource,
        time_column: str,
        targets: list[Target],
        horizon: int,
        frequency: Frequency,
        missing_policy: MissingPolicy,
        request_id: str | None = None,
        series_id_columns: list[str] | None = None,
        filter: FilterExpression | None = None,
        history_start: date | datetime | None = None,
        history_end: date | datetime | None = None,
        timezone: str = "UTC",
        quantiles: list[float] | None = None,
        model_alias: str | None = None,
        covariates: Covariates | None = None,
        options: ForecastOptions | None = None,
    ) -> Annotated[CallToolResult, ForecastOutput]:
        return _execute(
            lambda: application.forecast(
                _forecast_request(
                    source=source,
                    time_column=time_column,
                    targets=targets,
                    horizon=horizon,
                    frequency=frequency,
                    missing_policy=missing_policy,
                    request_id=request_id,
                    series_id_columns=series_id_columns,
                    filter=filter,
                    history_start=history_start,
                    history_end=history_end,
                    timezone=timezone,
                    quantiles=quantiles,
                    model_alias=model_alias,
                    covariates=covariates,
                    options=options,
                )
            ),
            request_id=request_id,
            prefix="fc",
        )

    @server.tool(
        name="backtest",
        title="Chronos backtest",
        description=(
            "Measure Chronos forecast quality over rolling historical windows "
            "using the same typed source contract as forecast."
        ),
        annotations=READ_ONLY_TOOL,
        structured_output=True,
    )
    def backtest(
        source: InlineSource,
        time_column: str,
        targets: list[Target],
        horizon: int,
        frequency: Frequency,
        missing_policy: MissingPolicy,
        request_id: str | None = None,
        series_id_columns: list[str] | None = None,
        filter: FilterExpression | None = None,
        history_start: date | datetime | None = None,
        history_end: date | datetime | None = None,
        timezone: str = "UTC",
        quantiles: list[float] | None = None,
        model_alias: str | None = None,
        covariates: Covariates | None = None,
        options: ForecastOptions | None = None,
        evaluation: BacktestEvaluation | None = None,
    ) -> Annotated[CallToolResult, BacktestOutput]:
        return _execute(
            lambda: application.backtest(
                BacktestRequest(
                    **_forecast_request(
                        source=source,
                        time_column=time_column,
                        targets=targets,
                        horizon=horizon,
                        frequency=frequency,
                        missing_policy=missing_policy,
                        request_id=request_id,
                        series_id_columns=series_id_columns,
                        filter=filter,
                        history_start=history_start,
                        history_end=history_end,
                        timezone=timezone,
                        quantiles=quantiles,
                        model_alias=model_alias,
                        covariates=covariates,
                        options=options,
                    ).model_dump(),
                    evaluation=evaluation or BacktestEvaluation(),
                )
            ),
            request_id=request_id,
            prefix="bt",
        )

    @server.tool(
        name="capabilities",
        title="Chronos capabilities",
        description="List configured model capabilities, limits, and source availability.",
        annotations=READ_ONLY_TOOL,
        structured_output=True,
    )
    def capabilities() -> CapabilitiesResponse:
        return application.capabilities()


def _forecast_request(
    *,
    source: InlineSource,
    time_column: str,
    targets: list[Target],
    horizon: int,
    frequency: Frequency,
    missing_policy: MissingPolicy,
    request_id: str | None,
    series_id_columns: list[str] | None,
    filter: FilterExpression | None,
    history_start: date | datetime | None,
    history_end: date | datetime | None,
    timezone: str,
    quantiles: list[float] | None,
    model_alias: str | None,
    covariates: Covariates | None,
    options: ForecastOptions | None,
) -> ForecastRequest:
    values = {
        "request_id": request_id,
        "source": source,
        "time_column": time_column,
        "targets": targets,
        "horizon": horizon,
        "frequency": frequency,
        "missing_policy": missing_policy,
        "series_id_columns": series_id_columns or [],
        "filter": filter,
        "history_start": history_start,
        "history_end": history_end,
        "timezone": timezone,
        "model_alias": model_alias,
        "covariates": covariates,
        "options": options or ForecastOptions(),
    }
    if quantiles is not None:
        values["quantiles"] = quantiles
    return ForecastRequest.model_validate(values)


def _execute(
    operation: Callable[[], object],
    *,
    request_id: str | None,
    prefix: str,
) -> CallToolResult:
    try:
        payload = operation()
    except ValidationError as exc:
        error = ChronosMCPError(
            ErrorCode.invalid_argument,
            "Tool arguments failed semantic validation.",
            details={"errors": exc.errors(include_url=False, include_input=False)},
        )
        return _call_result(error_response(error, request_id=request_id, prefix=prefix), is_error=True)
    except ChronosMCPError as exc:
        return _call_result(error_response(exc, request_id=request_id, prefix=prefix), is_error=True)
    except Exception:
        correlation_id = uuid.uuid4().hex
        logger.exception("Unexpected Chronos MCP failure correlation_id=%s", correlation_id)
        error = ChronosMCPError(
            ErrorCode.internal_error,
            "Unexpected Chronos MCP failure.",
            details={"correlation_id": correlation_id},
        )
        return _call_result(error_response(error, request_id=request_id, prefix=prefix), is_error=True)
    return _call_result(payload, is_error=False)


def _call_result(payload: object, *, is_error: bool) -> CallToolResult:
    if not hasattr(payload, "model_dump"):
        raise TypeError("Tool payload must be a Pydantic model.")
    structured = payload.model_dump(mode="json")  # type: ignore[attr-defined]
    serialized = json.dumps(structured, ensure_ascii=False, allow_nan=False)
    return CallToolResult(
        content=[TextContent(type="text", text=serialized)],
        structuredContent=structured,
        isError=is_error,
    )


def run() -> None:
    settings = ServerSettings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    create_server(settings=settings).run(transport=settings.transport)
