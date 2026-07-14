from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from backend.agent.contracts import AnalysisTaskContract
from backend.skills import SkillRegistry


class AgentPromptContextBuilder(BaseModel):
    """Build prompt fragments used by the generic agent node."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skill_registry: SkillRegistry
    enabled_analytical_skill_ids: set[str] | None = None

    @staticmethod
    def task_contract_prompt_block(contract: AnalysisTaskContract) -> str:
        requirements = list(contract.required_outputs or [])
        if not requirements:
            return ""
        lines = [
            "[ROLE: ANALYSIS_CONTRACT]",
            "- Treat the required outputs below as planning guidance, not as a hard finalization gate.",
            "- If a requested artifact cannot be produced, still answer from available evidence "
            "and do not claim that missing artifact exists.",
        ]
        for item in requirements:
            reason = str(item.reason or "").strip()
            suffix = f" - {reason}" if reason else ""
            lines.append(f"- required `{item.kind}`{suffix}")
        lines.append(
            "- Do not claim a chart, joined table, aggregation, metric, or catalog step "
            "unless the corresponding tool output exists."
        )
        return "\n".join(lines)
