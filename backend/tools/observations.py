from __future__ import annotations

from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

_MAX_SCHEMA_COLUMNS = 80


class ToolObservationStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class DataFrameSchemaSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    shape: tuple[int, int]
    columns: dict[str, str] = Field(default_factory=dict)
    warning: str | None = None

    @classmethod
    def from_value(cls, name: str, value: Any) -> DataFrameSchemaSummary:
        if not isinstance(value, pd.DataFrame):
            return cls(
                name=name,
                shape=(0, 0),
                warning="Referenced variable is not a pandas DataFrame.",
            )
        columns = list(value.dtypes.items())
        shown_columns = columns[:_MAX_SCHEMA_COLUMNS]
        warning = None
        if len(columns) > _MAX_SCHEMA_COLUMNS:
            warning = f"Only first {_MAX_SCHEMA_COLUMNS} of {len(columns)} columns are shown."
        return cls(
            name=name,
            shape=(int(value.shape[0]), int(value.shape[1])),
            columns={str(column): str(dtype) for column, dtype in shown_columns},
            warning=warning,
        )

    def to_text(self) -> str:
        lines = [f"{self.name}:", f"  shape: {self.shape[0]}x{self.shape[1]}"]
        if self.columns:
            lines.append("  columns:")
            lines.extend(f"    {name}: {dtype}" for name, dtype in self.columns.items())
        if self.warning:
            lines.append(f"  warning: {self.warning}")
        return "\n".join(lines)


class ToolExecutionObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ToolObservationStatus
    tool_name: str
    input_code: str
    executed_code: str | None = None
    error: str | None = None
    contract_hint: str | None = None
    available_variables: list[str] = Field(default_factory=list)
    referenced_dataframe_schemas: list[DataFrameSchemaSummary] = Field(default_factory=list)
    artifact_summary: str | None = None

    def to_message_text(self) -> str:
        if self.status is ToolObservationStatus.OK:
            lines = [f"{self.tool_name} succeeded"]
            if self.artifact_summary:
                lines.append(self.artifact_summary)
            return "\n\n".join(lines)

        lines = [
            f"{self.tool_name} failed",
            "input_code:",
            self.input_code.strip(),
        ]
        if self.executed_code and self.executed_code != self.input_code:
            lines.extend(["", "executed_code:", self.executed_code.strip()])
        if self.error:
            lines.extend(["", "error:", self.error.strip()])
        if self.contract_hint:
            lines.extend(["", "contract_hint:", self.contract_hint.strip()])
        if self.available_variables:
            lines.extend(["", "available_variables:"])
            lines.extend(f"- {name}" for name in self.available_variables)
        if self.referenced_dataframe_schemas:
            lines.extend(["", "referenced_dataframe_schemas:"])
            lines.extend(schema.to_text() for schema in self.referenced_dataframe_schemas)
        return "\n".join(lines).strip()
