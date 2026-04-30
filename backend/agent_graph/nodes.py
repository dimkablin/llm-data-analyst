from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.agent.callbacks import LLMTextCollector
from backend.agent_graph.analysis_context import AnalysisContextBuilder
from backend.agent_graph.llm import LlmFactory, MessageBuilder, build_runtime_metadata
from backend.agent_graph.message_codec import (
    message_to_state,
    messages_to_states,
    states_to_messages,
)
from backend.agent_graph.prompting import ExecutionPromptBuilder
from backend.agent_graph.runtime import RuntimeContextStore
from backend.agent_graph.routing import RouteClassifier
from backend.agent_graph.state import AgentGraphState, WorkingMemoryState
from backend.agent_graph.tool_execution import ToolCallExecutor, ToolResultMapper
from backend.observability.phoenix import record_llm_usage_on_active_span


def initial_working_memory(prompt: str) -> WorkingMemoryState:
    return {
        "goal": prompt,
        "step_index": 0,
        "tool_call_count": 0,
        "artifact_refs": [],
        "sandbox_var_names": [],
        "current_plan": [],
        "completed_actions": [],
        "last_tool_result_summary": "",
    }


@dataclass(slots=True)
class AgentGraphNodes:
    """Callable node collection for the LangGraph agent runtime.

    LangGraph accepts plain callables, but grouping nodes in a class gives us a
    clean dependency boundary.  Shared services such as runtime context lookup,
    tool execution and telemetry can be injected here without turning module
    globals into hidden dependencies.
    """

    runtime_context_store: RuntimeContextStore
    route_classifier: RouteClassifier = field(default_factory=RouteClassifier)
    analysis_context_builder: AnalysisContextBuilder = field(
        default_factory=AnalysisContextBuilder,
    )

    def prepare(self, state: AgentGraphState) -> dict[str, Any]:
        """Normalize request state before routing.

        This node is deliberately behavior-light in the first migration slice.
        The next slices will move context/tool construction here from the
        legacy runner.
        """

        prompt = str(state.get("prompt") or "")
        return {
            "prompt": prompt,
            "history": list(state.get("history") or []),
            "use_history": bool(state.get("use_history", True)),
            "include_reasoning": bool(state.get("include_reasoning", False)),
            "trace_context": dict(state.get("trace_context") or {}),
            "session_source": dict(state.get("session_source") or {}),
            "selected_skill_ids": list(state.get("selected_skill_ids") or []),
            "runtime_context_key": str(state.get("runtime_context_key") or ""),
            "route": state.get("route") or "analysis",
            "done": bool(state.get("done", False)),
            "stop_reason": str(state.get("stop_reason") or ""),
            "step_index": int(state.get("step_index") or 0),
            "max_steps": max(1, int(state.get("max_steps") or 1)),
            "messages": list(state.get("messages") or []),
            "pending_tool_calls": list(state.get("pending_tool_calls") or []),
            "available_tool_keys": list(state.get("available_tool_keys") or []),
            "capability_context": dict(state.get("capability_context") or {}),
            "tool_call_count": int(state.get("tool_call_count") or 0),
            "tool_names": list(state.get("tool_names") or []),
            "reasoning_steps": list(state.get("reasoning_steps") or []),
            "artifact_refs": list(state.get("artifact_refs") or []),
            "working_memory": state.get("working_memory") or initial_working_memory(prompt),
            "status": "running",
        }

    def prepare_analysis(self, state: AgentGraphState) -> dict[str, Any]:
        """Build tools, sandbox and capability context for analysis requests."""

        context = self._runtime_context(state)
        services = context.services if context is not None else None
        if context is None or services is None:
            return {
                "available_tool_keys": [],
                "capability_context": {},
                "max_steps": max(1, int(state.get("max_steps") or 1)),
            }

        analysis_context = self.analysis_context_builder.build(
            services=services,
            runtime_context=context,
            state=dict(state),
        )
        context.execution_system_prompt = ExecutionPromptBuilder(services).build(
            capability_context=analysis_context.capability_context,
            sandbox=analysis_context.sandbox,
            selected_skill_ids=list(state.get("selected_skill_ids") or []),
            df=context.df,
            session_source=dict(state.get("session_source") or {}),
            tool_db_runtime=analysis_context.tool_db_runtime,
            user_prompt=str(state.get("prompt") or ""),
        )
        return {
            "available_tool_keys": analysis_context.available_tool_keys,
            "capability_context": analysis_context.capability_context,
            "max_steps": analysis_context.max_steps,
        }

    def route(self, state: AgentGraphState) -> dict[str, Any]:
        """Classify the request before expensive analysis setup."""

        route = self.route_classifier.classify(
            str(state.get("prompt") or ""),
            has_data=self._has_data_context(state),
        )
        if route in {"chat", "summary"}:
            return {
                "done": True,
                "route": route,
                "stop_reason": f"{route}_route",
            }
        return {"done": False, "route": "analysis"}

    def chat(self, state: AgentGraphState) -> dict[str, Any]:
        """Run lightweight chat through the graph runtime."""

        context = self._runtime_context(state)
        services = context.services if context is not None else None
        if services is None:
            return {
                "final_text": "Привет. Я на связи.",
                "route": "chat",
                "status": "done",
            }

        callbacks = context.callbacks if context is not None else []
        prompt = str(state.get("prompt") or "")
        message_builder = MessageBuilder(
            settings=services.settings,
            user_memory=services.user_memory,
            session_memory=services.session_memory,
        )
        messages = message_builder.build_chat_messages(
            prompt=prompt,
            history=list(state.get("history") or []),
            use_history=bool(state.get("use_history", True)),
        )
        runtime_config: dict[str, Any] = {"callbacks": callbacks}
        metadata = build_runtime_metadata(state.get("trace_context") or {})
        if metadata:
            runtime_config["metadata"] = metadata

        try:
            response = LlmFactory(services.settings).build(
                role="chat",
                include_reasoning=bool(state.get("include_reasoning", False)),
            ).invoke(messages, config=runtime_config)
            record_llm_usage_on_active_span(
                response,
                fallback_model=services.settings.llm_model,
                fallback_provider=services.settings.llm_provider,
            )
        except Exception as exc:
            return {
                "final_text": "Привет. Я на связи, но языковая модель сейчас недоступна.",
                "reasoning": f"chat failed: {exc}",
                "route": "chat",
                "status": "done",
            }

        text_collector = next((cb for cb in callbacks if isinstance(cb, LLMTextCollector)), None)
        final_text = ""
        reasoning = None
        if text_collector and text_collector.messages:
            latest = text_collector.messages[-1]
            final_text = str(latest.get("text") or "")
            reasoning = latest.get("reasoning") or None
        if not final_text:
            final_text = message_builder.content_to_text(getattr(response, "content", ""))
        if reasoning is None:
            reasoning = getattr(response, "additional_kwargs", {}).get("reasoning") or None
        return {
            "final_text": final_text.strip() or "Привет. Я на связи.",
            "reasoning": reasoning,
            "route": "chat",
            "status": "done",
        }

    def summary(self, state: AgentGraphState) -> dict[str, Any]:
        """Temporary summary endpoint until the old summary path is migrated."""

        return {
            "final_text": "Summary runtime is not migrated yet.",
            "route": "summary",
            "status": "done",
        }

    def plan(self, state: AgentGraphState) -> dict[str, Any]:
        """Run planner_tool once before entering the model/tool loop."""

        context = self._runtime_context(state)
        if context is None:
            return {"status": "running"}

        planner = next(
            (tool for tool in context.tools if getattr(tool, "name", "") == "planner_tool"),
            None,
        )
        if planner is None:
            return {"status": "running"}

        runtime_config: dict[str, Any] = {"callbacks": context.callbacks}
        metadata = build_runtime_metadata(state.get("trace_context") or {})
        if metadata:
            runtime_config["metadata"] = metadata

        try:
            plan_result = planner.invoke(
                {
                    "name": "planner_tool",
                    "args": {
                        "question": str(state.get("prompt") or ""),
                        "context": self._planner_history_context(state),
                    },
                    "id": "pre_plan_0",
                    "type": "tool_call",
                },
                config=runtime_config,
            )
            plan_text = ToolResultMapper(tool_name="planner_tool").map(plan_result).text.strip()
        except Exception as exc:
            plan_text = f"Planner failed: {exc}"

        if not plan_text:
            return {"status": "running"}

        context.execution_system_prompt = (
            context.execution_system_prompt
            + "\n\n## Preliminary Analysis Plan\n"
            + plan_text
        ).strip()

        working_memory = dict(state.get("working_memory") or initial_working_memory(""))
        working_memory["current_plan"] = [
            line.strip() for line in plan_text.splitlines() if line.strip()
        ]
        return {"working_memory": working_memory, "status": "running"}

    def call_llm(self, state: AgentGraphState) -> dict[str, Any]:
        """Call the tool-bound LLM and store either tool calls or final text."""

        context = self._runtime_context(state)
        services = context.services if context is not None else None
        if context is None or services is None:
            return {"status": "done", "pending_tool_calls": []}

        callbacks = context.callbacks
        message_builder = MessageBuilder(
            settings=services.settings,
            user_memory=services.user_memory,
            session_memory=services.session_memory,
        )
        message_states = list(state.get("messages") or [])
        if message_states:
            messages = states_to_messages(message_states)
        else:
            messages = message_builder.build_messages(
                prompt=str(state.get("prompt") or ""),
                history=list(state.get("history") or []),
                use_history=bool(state.get("use_history", True)),
                system_prompt=context.execution_system_prompt,
            )
            message_states = messages_to_states(messages)

        tools_for_loop = [
            tool
            for tool in context.tools
            if str(getattr(tool, "name", "")).strip() not in {"planner_tool", "review_tool"}
        ]
        llm_factory = context.llm_factory or LlmFactory(services.settings)
        bound_llm = llm_factory.build(
            role="tool",
            include_reasoning=bool(state.get("include_reasoning", False)),
            timeout_sec=min(
                services.settings.agent_step_timeout_sec,
                services.settings.backend_query_timeout_sec,
            ),
        ).bind_tools(tools_for_loop)

        runtime_config: dict[str, Any] = {"callbacks": callbacks}
        metadata = build_runtime_metadata(state.get("trace_context") or {})
        if metadata:
            runtime_config["metadata"] = metadata

        try:
            response = bound_llm.invoke(messages, config=runtime_config)
            record_llm_usage_on_active_span(
                response,
                fallback_model=services.settings.llm_model,
                fallback_provider=services.settings.llm_provider,
            )
        except Exception as exc:
            return {
                "final_text": "Language model is unavailable.",
                "reasoning": f"LLM invoke failed: {exc}",
                "pending_tool_calls": [],
                "status": "error",
            }

        message_states.append(message_to_state(response))
        reasoning_steps = list(state.get("reasoning_steps") or [])
        reasoning = getattr(response, "additional_kwargs", {}).get("reasoning") or None
        if reasoning:
            reasoning_steps.append(str(reasoning))

        tool_calls = [
            {
                "id": str(call.get("id") or ""),
                "name": str(call.get("name") or ""),
                "args": dict(call.get("args") or {}),
                "type": str(call.get("type") or "tool_call"),
            }
            for call in (getattr(response, "tool_calls", None) or [])
            if isinstance(call, dict)
        ]
        if tool_calls:
            return {
                "messages": message_states,
                "pending_tool_calls": tool_calls,
                "reasoning": reasoning or state.get("reasoning"),
                "reasoning_steps": reasoning_steps,
                "status": "running",
            }

        final_text = message_builder.content_to_text(getattr(response, "content", ""))
        return {
            "messages": message_states,
            "pending_tool_calls": [],
            "final_text": final_text.strip(),
            "reasoning": reasoning or state.get("reasoning"),
            "reasoning_steps": reasoning_steps,
            "status": "done",
        }

    def execute_tools(self, state: AgentGraphState) -> dict[str, Any]:
        """Execute pending tool calls and append ToolMessages to graph state."""

        context = self._runtime_context(state)
        if context is None:
            return {"pending_tool_calls": []}

        tool_calls = list(state.get("pending_tool_calls") or [])
        summary = ToolCallExecutor(
            tools=context.tools,
            callbacks=context.callbacks,
            metadata=build_runtime_metadata(state.get("trace_context") or {}),
        ).execute(tool_calls)

        working_memory = dict(state.get("working_memory") or initial_working_memory(""))
        completed_actions = list(working_memory.get("completed_actions") or [])
        completed_actions.extend(summary.completed_actions)
        working_memory["completed_actions"] = completed_actions
        working_memory["tool_call_count"] = int(working_memory.get("tool_call_count") or 0) + summary.tool_call_count
        working_memory["step_index"] = int(working_memory.get("step_index") or 0) + summary.tool_call_count
        existing_artifacts = list(working_memory.get("artifact_refs") or [])
        next_step = int(working_memory.get("step_index") or 0)
        for ref in summary.artifact_refs:
            ref["step_index"] = max(0, next_step - 1)
            existing_artifacts.append(ref)
        working_memory["artifact_refs"] = existing_artifacts
        if summary.completed_actions:
            working_memory["last_tool_result_summary"] = summary.completed_actions[-1]

        tool_names = list(state.get("tool_names") or [])
        for name in summary.tool_names:
            if name and name not in tool_names:
                tool_names.append(name)

        return {
            "messages": list(state.get("messages") or []) + summary.messages,
            "pending_tool_calls": [],
            "tool_call_count": int(state.get("tool_call_count") or 0) + summary.tool_call_count,
            "tool_names": tool_names,
            "artifact_refs": list(state.get("artifact_refs") or []) + summary.artifact_refs,
            "working_memory": working_memory,
        }

    def finalize(self, state: AgentGraphState) -> dict[str, Any]:
        """Finalize graph output without touching the production runner yet."""

        final_text = str(state.get("final_text") or "")
        return {
            "final_text": final_text,
            "reasoning": state.get("reasoning"),
            "artifact_refs": list(state.get("artifact_refs") or []),
            "status": "done",
        }

    def _runtime_context(self, state: AgentGraphState) -> Any | None:
        key = str(state.get("runtime_context_key") or "")
        if not key:
            return None
        return self.runtime_context_store.get(key)

    @staticmethod
    def _planner_history_context(state: AgentGraphState) -> str:
        history = list(state.get("history") or [])
        rows: list[str] = []
        for item in history[-4:]:
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            role = "User" if item.get("role") == "user" else "Assistant"
            rows.append(f"{role}: {content[:200]}")
        return "\n".join(rows)

    @staticmethod
    def _has_data_context(state: AgentGraphState) -> bool:
        source = state.get("session_source") or {}
        return bool(
            source.get("csv_loaded")
            or source.get("source_type") in {"csv", "db_connection"}
            or (state.get("trace_context") or {}).get("db_connection_id")
        )


def should_skip_to_finalize(state: AgentGraphState) -> str:
    route = state.get("route")
    if route == "chat":
        return "chat"
    if route == "summary":
        return "summary"
    return "prepare_analysis"


def should_continue_tools(state: AgentGraphState) -> str:
    if state.get("pending_tool_calls"):
        return "tools"
    return "finalize"
