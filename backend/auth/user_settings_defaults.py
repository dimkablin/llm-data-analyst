from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.core.config import DEPTH_MAX_STEPS

if TYPE_CHECKING:
    from backend.auth.auth_db import UserSettings
    from backend.core.config import Settings


@dataclass(frozen=True)
class UserSettingsDefaults:
    theme: str = "dark"
    default_include_reasoning: bool = True
    default_answer_style: str = "detailed"
    analysis_mode: str = "fast"
    analysis_depth: str = "light"
    llm_temperature_chat: float = 0.7
    llm_temperature_tool: float = 0.5
    llm_max_tokens_default: int = 4096
    llm_max_tokens_reasoning: int = 4096
    backend_query_timeout_sec: int = 180
    agent_max_steps: int = DEPTH_MAX_STEPS["light"]
    agent_step_timeout_sec: int = 45
    agent_inner_recursion_limit: int = DEPTH_MAX_STEPS["light"]
    agent_react_enabled: bool = False
    ui_scale: int = 100
    llm_streaming: bool = True
    show_thinking: bool = True
    show_think_planning: bool = True
    show_think_tool: bool = True
    show_think_final: bool = True
    always_use_analysis_plan: bool = False
    show_detailed_tool_steps: bool = False
    show_rag_errors: bool = True
    anomaly_check_enabled: bool = False

    def to_user_settings(self) -> UserSettings:
        from backend.auth.auth_db import UserSettings

        return UserSettings(**self.__dict__)


DEFAULT_USER_SETTINGS = UserSettingsDefaults()


def user_settings_defaults_from_runtime(settings: Settings) -> UserSettingsDefaults:
    depth = str(settings.agent_analysis_depth or "light").strip().lower()
    if depth not in DEPTH_MAX_STEPS:
        depth = "light"
    depth_cap = DEPTH_MAX_STEPS[depth]
    return UserSettingsDefaults(
        analysis_depth=depth,
        llm_temperature_chat=settings.llm_temperature_chat,
        llm_temperature_tool=settings.llm_temperature_tool,
        llm_max_tokens_default=settings.llm_max_tokens_default,
        llm_max_tokens_reasoning=settings.llm_max_tokens_reasoning,
        backend_query_timeout_sec=settings.backend_query_timeout_sec,
        agent_max_steps=min(max(2, settings.agent_max_steps), depth_cap),
        agent_step_timeout_sec=settings.agent_step_timeout_sec,
        agent_inner_recursion_limit=min(
            max(2, settings.agent_inner_recursion_limit),
            depth_cap,
        ),
        llm_streaming=settings.llm_streaming,
    )
