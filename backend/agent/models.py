from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from backend.sessions.session_memory import StructuredSessionMemory


class CapabilityOutcome(BaseModel):
    capability_key: str
    tool_key: str
    status: Literal["ok", "error"]
    artifact_types: list[str] = Field(default_factory=list)
    provenance: str
    error_fingerprint: str | None = None


class AgentRuntimeEffects(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_memory_notes: tuple[str, ...] = ()
    session_memory_notes: tuple[str, ...] = ()
    session_memory: StructuredSessionMemory | None = None


class TerminalStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorCategory(StrEnum):
    NONE = "none"
    TASK_CONTRACT = "task_contract"
    GRAPH = "graph"
    MODEL = "model"
    TOOL = "tool"
    VALIDATION = "validation"
    TRANSPORT = "transport"
    INTERNAL = "internal"
    CANCELLED = "cancelled"


class AgentOutcome(BaseModel):
    """Canonical outcome dimensions used by runtime, cache, API and telemetry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_envelope_valid: bool
    task_contract_satisfied: bool
    terminal_status: TerminalStatus
    error_category: ErrorCategory
    partial_result: bool = False

    @model_validator(mode="after")
    def _validate_success_invariants(self) -> AgentOutcome:
        if self.terminal_status is TerminalStatus.SUCCESS:
            if not self.response_envelope_valid or not self.task_contract_satisfied:
                raise ValueError("success requires a valid envelope and satisfied task contract")
            if self.error_category is not ErrorCategory.NONE:
                raise ValueError("success cannot carry an error category")
        elif self.task_contract_satisfied:
            raise ValueError("non-success outcomes cannot satisfy the task contract")
        return self

    @classmethod
    def success(cls) -> AgentOutcome:
        return cls(
            response_envelope_valid=True,
            task_contract_satisfied=True,
            terminal_status=TerminalStatus.SUCCESS,
            error_category=ErrorCategory.NONE,
        )

    @classmethod
    def partial(cls, category: ErrorCategory = ErrorCategory.TASK_CONTRACT) -> AgentOutcome:
        return cls(
            response_envelope_valid=True,
            task_contract_satisfied=False,
            terminal_status=TerminalStatus.PARTIAL,
            error_category=category,
            partial_result=True,
        )

    @classmethod
    def unavailable(cls, category: ErrorCategory = ErrorCategory.TOOL) -> AgentOutcome:
        return cls(
            response_envelope_valid=True,
            task_contract_satisfied=False,
            terminal_status=TerminalStatus.UNAVAILABLE,
            error_category=category,
        )

    @classmethod
    def failed(cls, category: ErrorCategory = ErrorCategory.INTERNAL) -> AgentOutcome:
        return cls(
            response_envelope_valid=True,
            task_contract_satisfied=False,
            terminal_status=TerminalStatus.FAILED,
            error_category=category,
        )

    @classmethod
    def cancelled(cls) -> AgentOutcome:
        return cls(
            response_envelope_valid=True,
            task_contract_satisfied=False,
            terminal_status=TerminalStatus.CANCELLED,
            error_category=ErrorCategory.CANCELLED,
        )

    @property
    def cacheable_success(self) -> bool:
        return self.terminal_status is TerminalStatus.SUCCESS and self.task_contract_satisfied


class AgentResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_text: str
    reasoning: str | None
    artifacts: list
    route: Literal["analysis", "summary"] = "analysis"
    tool_calls: int = 0
    tool_names: list[str] = Field(default_factory=list)
    llm_unreachable: bool = False
    outcome: AgentOutcome = Field(default_factory=AgentOutcome.failed)
    capability_outcomes: list[CapabilityOutcome] = Field(default_factory=list)
    error_fingerprints: list[str] = Field(default_factory=list)
    retry_count: int = 0
    tool_error_count: int = 0
    reasoning_steps: list[str] = Field(default_factory=list)
    runtime_effects: AgentRuntimeEffects = Field(default_factory=AgentRuntimeEffects)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_outcome_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "outcome" in value:
            return value
        data = dict(value)
        legacy_status = data.pop("terminal_status", None)
        legacy_contract = data.pop("contract_valid", None)
        if legacy_status is None and legacy_contract is None:
            return data
        status = TerminalStatus(legacy_status or ("success" if legacy_contract else "failed"))
        if status is TerminalStatus.SUCCESS and legacy_contract is not False:
            data["outcome"] = AgentOutcome.success()
        elif status is TerminalStatus.PARTIAL:
            data["outcome"] = AgentOutcome.partial()
        elif status is TerminalStatus.UNAVAILABLE:
            data["outcome"] = AgentOutcome.unavailable()
        elif status is TerminalStatus.CANCELLED:
            data["outcome"] = AgentOutcome.cancelled()
        else:
            data["outcome"] = AgentOutcome.failed()
        return data

    @computed_field
    @property
    def response_envelope_valid(self) -> bool:
        return self.outcome.response_envelope_valid

    @computed_field
    @property
    def task_contract_satisfied(self) -> bool:
        return self.outcome.task_contract_satisfied

    @computed_field
    @property
    def terminal_status(self) -> TerminalStatus:
        return self.outcome.terminal_status

    @computed_field
    @property
    def error_category(self) -> ErrorCategory:
        return self.outcome.error_category

    @computed_field
    @property
    def partial_result(self) -> bool:
        return self.outcome.partial_result

    @computed_field
    @property
    def contract_valid(self) -> bool:
        """Deprecated compatibility alias for task_contract_satisfied."""
        return self.outcome.task_contract_satisfied


class QueryCacheEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    created_at: float
    response: AgentResponse
