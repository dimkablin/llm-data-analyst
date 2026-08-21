from __future__ import annotations

import json
from typing import ClassVar

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from backend.agent.working_memory import AnalysisWorkingMemory, PlanItem
from backend.tools.instructions import tool_description


class UpdatePlanInput(BaseModel):
    plan: list[PlanItem] = Field(
        min_length=1,
        max_length=8,
        description="The complete current analysis checklist. Replaces the previous checklist.",
    )

    @model_validator(mode="after")
    def _validate_plan(self) -> UpdatePlanInput:
        normalized_steps = [item.step.casefold() for item in self.plan]
        if len(normalized_steps) != len(set(normalized_steps)):
            raise ValueError("plan steps must be unique")
        if sum(item.status == "in_progress" for item in self.plan) > 1:
            raise ValueError("at most one plan step may be in_progress")
        return self


class UpdatePlanTool(BaseTool):
    """Replace the main agent's current per-request checklist."""

    name: str = "update_plan"
    description: str = tool_description("update_plan")
    args_schema: type[BaseModel] = UpdatePlanInput
    parallel_safe: ClassVar[bool] = True

    _working_memory: AnalysisWorkingMemory = PrivateAttr()

    def __init__(self, working_memory: AnalysisWorkingMemory) -> None:
        super().__init__()
        self._working_memory = working_memory

    def _run(self, plan: list[PlanItem]) -> str:
        self._working_memory.current_plan = list(plan)
        completed = sum(item.status == "completed" for item in plan)
        return json.dumps(
            {
                "plan": [item.model_dump(mode="json") for item in plan],
                "completed": completed,
                "total": len(plan),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    async def _arun(self, plan: list[PlanItem]) -> str:
        return self._run(plan)
