from pydantic import BaseModel

from backend.agent.runner import AgentRunner
from backend.auth import UserMemory
from backend.core.config import Settings
from backend.sessions.session_memory import SessionMemory


def test_message_build_request_is_pydantic_contract() -> None:
    from backend.agent.services.message_builder import MessageBuildRequest

    request = MessageBuildRequest(
        prompt="hello",
        history=[],
        use_history=False,
        settings=Settings(),
        user_memory=UserMemory(profile="Data lead", notes=""),
        session_memory=SessionMemory(),
    )

    assert isinstance(request, BaseModel)
    assert request.prompt == "hello"


def test_build_messages_injects_memory_without_runner() -> None:
    from langchain_core.messages import SystemMessage

    from backend.agent.services.message_builder import MessageBuildRequest, build_messages

    messages = build_messages(
        MessageBuildRequest(
            prompt="hello",
            history=[],
            use_history=False,
            settings=Settings(),
            user_memory=UserMemory(profile="Data lead", notes=""),
            session_memory=SessionMemory(),
        )
    )

    system_messages = [message for message in messages if isinstance(message, SystemMessage)]

    assert any("Data lead" in str(message.content) for message in system_messages)


def test_build_messages_keeps_system_prompt_and_memory_in_first_system_message() -> None:
    from langchain_core.messages import SystemMessage

    from backend.agent.context_window import trim_context_messages
    from backend.agent.services.message_builder import MessageBuildRequest, build_messages

    messages = build_messages(
        MessageBuildRequest(
            prompt="hello",
            history=[],
            use_history=False,
            settings=Settings(),
            user_memory=UserMemory(profile="", notes="concise answers"),
            session_memory=SessionMemory(notes="current dataset"),
            system_prompt="base system prompt",
        )
    )

    system_messages = [message for message in messages if isinstance(message, SystemMessage)]

    assert len(system_messages) == 1
    assert system_messages[0] is messages[0]
    assert "base system prompt" in str(system_messages[0].content)
    assert "concise answers" in str(system_messages[0].content)
    assert "current dataset" in str(system_messages[0].content)

    trimmed = trim_context_messages(messages, max_input_tokens=1000)

    assert "concise answers" in str(trimmed[0].content)
    assert "current dataset" in str(trimmed[0].content)


def test_build_messages_skips_history_already_in_context_summary() -> None:
    from backend.agent.services.message_builder import MessageBuildRequest, build_messages

    memory = SessionMemory(
        context_summary="old-marker already summarized",
        compacted_message_count=2,
    )
    messages = build_messages(
        MessageBuildRequest(
            prompt="current prompt",
            history=[
                {"role": "user", "content": "old-marker-0"},
                {"role": "assistant", "content": "old-marker-1"},
                {"role": "user", "content": "fresh-marker"},
            ],
            use_history=True,
            settings=Settings(),
            user_memory=UserMemory(profile="", notes=""),
            session_memory=memory,
        )
    )

    text = "\n".join(str(message.content) for message in messages)

    assert "old-marker already summarized" in text
    assert "old-marker-0" not in text
    assert "old-marker-1" not in text
    assert "fresh-marker" in text


def test_build_messages_keeps_artifact_context_internal() -> None:
    from langchain_core.messages import AIMessage, SystemMessage

    from backend.agent.services.message_builder import MessageBuildRequest, build_messages

    messages = build_messages(
        MessageBuildRequest(
            prompt="current prompt",
            history=[
                {"role": "user", "content": "source question"},
                {
                    "role": "assistant",
                    "content": (
                        "Result table:\n\n"
                        "| branch | value |\n"
                        "|:--|--:|\n"
                        "| A | 10 |\n\n"
                        "Branch A has the highest value."
                    ),
                    "artifacts": [
                        {
                            "id": "artifact-1",
                            "execution_artifact_id": "artifact-1",
                            "type": "table",
                            "text": "category_metrics",
                            "execution": {
                                "data_complete": True,
                                "schema": {
                                    "columns": ["branch", "value"],
                                    "dtypes": {"branch": "object", "value": "int64"},
                                    "row_count": 1,
                                },
                            },
                            "data": {
                                "format": "split",
                                "data": {
                                    "columns": ["branch", "value"],
                                    "index": [0],
                                    "data": [["A", 10]],
                                },
                            },
                        },
                        {
                            "id": "artifact-preview",
                            "type": "table",
                            "text": "preview_only",
                            "execution": {
                                "data_complete": False,
                                "schema": {"columns": ["value"], "row_count": 1},
                            },
                        },
                    ],
                },
            ],
            use_history=True,
            settings=Settings(),
            user_memory=UserMemory(profile="", notes=""),
            session_memory=SessionMemory(),
        )
    )

    text = "\n".join(str(message.content) for message in messages)
    assert "Branch A has the highest value." in text
    assert "| A | 10 |" not in text
    assert "Result table" in text
    assert "Контекст предыдущих артефактов" not in text
    assert "Запрос, породивший артефакты" not in text
    internal = [
        message
        for message in messages
        if isinstance(message, SystemMessage) and "[INTERNAL_ARTIFACT_CONTEXT]" in str(message.content)
    ]
    assert len(internal) == 1
    assert "category_metrics" in str(internal[0].content)
    assert "artifact_id=artifact-1" in str(internal[0].content)
    assert "branch:object" in str(internal[0].content)
    assert 'data_preview=[{"branch": "A", "value": 10}]' in str(internal[0].content)
    assert "not current sandbox variables" in str(internal[0].content)
    assert "preview_only" not in str(internal[0].content)
    assert "source question" in str(internal[0].content)
    assistant = [message for message in messages if isinstance(message, AIMessage)]
    assert [str(message.content) for message in assistant] == [
        "Result table:\n\n\nBranch A has the highest value."
    ]


def test_build_messages_keeps_history_context_in_one_leading_system_message() -> None:
    from langchain_core.messages import SystemMessage

    from backend.agent.services.message_builder import MessageBuildRequest, build_messages

    history: list[dict[str, object]] = []
    for index in range(10):
        history.extend(
            [
                {"role": "user", "content": f"question-{index}"},
                {
                    "role": "assistant",
                    "content": f"answer-{index}",
                    "artifacts": [
                        {
                            "id": f"artifact-{index}",
                            "execution_artifact_id": f"artifact-{index}",
                            "type": "table",
                            "text": f"table-{index}",
                            "execution": {
                                "data_complete": True,
                                "schema": {
                                    "columns": ["value"],
                                    "dtypes": {"value": "int64"},
                                    "row_count": 1,
                                },
                            },
                            "data": {
                                "format": "split",
                                "data": {
                                    "columns": ["value"],
                                    "index": [0],
                                    "data": [[index]],
                                },
                            },
                        }
                    ],
                },
            ]
        )

    messages = build_messages(
        MessageBuildRequest(
            prompt="continue",
            history=history,
            use_history=True,
            settings=Settings(),
            user_memory=UserMemory(profile="", notes=""),
            session_memory=SessionMemory(),
            system_prompt="base system prompt",
        )
    )

    system_messages = [message for message in messages if isinstance(message, SystemMessage)]
    assert len(system_messages) == 1
    assert system_messages[0] is messages[0]
    assert "base system prompt" in str(system_messages[0].content)
    assert "Краткая сводка предыдущего диалога" in str(system_messages[0].content)
    assert "[INTERNAL_ARTIFACT_CONTEXT]" in str(system_messages[0].content)


def test_build_messages_keeps_explanation_before_terminal_artifact_table() -> None:
    from langchain_core.messages import AIMessage

    from backend.agent.services.message_builder import MessageBuildRequest, build_messages

    messages = build_messages(
        MessageBuildRequest(
            prompt="continue",
            history=[
                {
                    "role": "assistant",
                    "content": (
                        "The metric increased because every observed segment grew.\n\n"
                        "| month | metric |\n"
                        "|---|---:|\n"
                        "| Jan | 12 |"
                    ),
                    "artifacts": [{"type": "table", "text": "monthly_metric"}],
                }
            ],
            use_history=True,
            settings=Settings(),
            user_memory=UserMemory(profile="", notes=""),
            session_memory=SessionMemory(),
        )
    )

    assistant = [message for message in messages if isinstance(message, AIMessage)]
    assert [str(message.content) for message in assistant] == [
        "The metric increased because every observed segment grew."
    ]


def test_build_messages_omits_context_summary_when_history_disabled() -> None:
    from backend.agent.services.message_builder import MessageBuildRequest, build_messages

    memory = SessionMemory(
        notes="persistent session note",
        context_summary="old-marker already summarized",
        compacted_message_count=2,
    )
    messages = build_messages(
        MessageBuildRequest(
            prompt="current prompt",
            history=[{"role": "user", "content": "fresh-marker"}],
            use_history=False,
            settings=Settings(),
            user_memory=UserMemory(profile="", notes=""),
            session_memory=memory,
        )
    )

    text = "\n".join(str(message.content) for message in messages)

    assert "persistent session note" in text
    assert "old-marker already summarized" not in text
    assert "fresh-marker" not in text
    assert "current prompt" in text


def test_execution_prompt_builder_matches_runner_contract() -> None:
    from backend.agent.services.message_builder import (
        ExecutionSystemPromptRequest,
        build_execution_system_prompt,
    )

    runner = AgentRunner()
    capability_context = {
        "source_mode": "dataset",
        "tool_descriptions": "",
        "available_tool_keys": ["sql_tool"],
    }

    request = ExecutionSystemPromptRequest(
        settings=runner.settings,
        skill_registry=runner.skill_registry,
        enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
        capability_context=capability_context,
    )
    prompt = build_execution_system_prompt(request)

    assert "sql_tool" in prompt
    assert "Thought:" not in prompt


def test_execution_prompt_does_not_reload_base_workflow_before_routine_sql() -> None:
    from backend.agent.services.message_builder import (
        ExecutionSystemPromptRequest,
        build_execution_system_prompt,
    )
    from backend.agent.tool_loop import ToolLoopRequest

    runner = AgentRunner()
    capability_context = {
        "source_mode": "dataset",
        "tool_descriptions": "",
        "available_tool_keys": ["get_tool_instructions", "sql_tool"],
    }

    request = ExecutionSystemPromptRequest(
        settings=runner.settings,
        skill_registry=runner.skill_registry,
        enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
        capability_context=capability_context,
    )
    prompt = build_execution_system_prompt(request)

    assert "active_workflow" not in prompt
    assert "CSV/XLSX" in prompt
    assert "do not reload the base `general_analytics` workflow" in prompt
    assert not hasattr(runner.skill_registry.get("general_analytics"), "execution_contract")
    assert "required_tools" not in ToolLoopRequest.model_fields


def test_execution_prompt_explains_core_tool_pipeline_contract() -> None:
    from backend.agent.services.message_builder import (
        ExecutionSystemPromptRequest,
        build_execution_system_prompt,
    )

    runner = AgentRunner()
    capability_context = {
        "source_mode": "dataset",
        "tool_descriptions": "",
        "available_tool_keys": ["sql_tool", "pandas_tool", "plotly_tool"],
    }

    prompt = build_execution_system_prompt(
        ExecutionSystemPromptRequest(
            settings=runner.settings,
            skill_registry=runner.skill_registry,
            enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
            capability_context=capability_context,
        )
    )

    assert "DATA FLOW POLICY" in prompt
    assert "`sql_tool`" in prompt
    assert "`pandas_tool`" in prompt
    assert "`plotly_tool`" in prompt
    assert "Successful outputs become named artifacts" in prompt
    assert "failed calls create no data artifact" in prompt


def test_postgresql_source_prompt_uses_typed_portable_sql_patterns() -> None:
    from types import SimpleNamespace

    from backend.agent.services.message_builder import db_session_prompt_block

    block = db_session_prompt_block(
        session_source={},
        runtime=SimpleNamespace(
            name="Demo",
            db_type="postgresql",
            database="demo",
            options={"schema": "analytics"},
        ),
        df=None,
    )

    assert "COUNT(*) FILTER (WHERE predicate)" in block
    assert "DATE missingness uses `IS NULL` or `IS NOT NULL`" in block
    assert "AVG(value) AS avg_value" in block
    assert "`ROUND(double precision, digits)` is invalid" in block
    assert "PostgreSQL has no `UNPIVOT` syntax" in block
    assert "project the LATERAL value-table alias columns" in block
    assert "switch to explicit `UNION ALL` branches" in block


def test_tool_data_flow_policy_is_positive_atomic_execution_protocol() -> None:
    from backend.agent.services.message_builder import tool_data_flow_policy_block

    policy = tool_data_flow_policy_block({"sql_tool", "pandas_tool", "plotly_tool", "mcp__chronos__forecast"})

    for stage in ("OBJECTIVE", "GROUND", "EXECUTE", "RECOVER", "VERIFY", "COMPLETE"):
        assert stage in policy
    assert "dimension, measure, identifier" in policy
    assert "Reuse complete semantic context without another lookup" in policy
    assert "missing, incomplete, or ambiguous terms, formulas, and relationships" in policy
    assert "database schema only for unresolved physical fields" in policy
    assert "human-readable labels before ranking" in policy
    assert "if none exists, keep the code" in policy
    assert "error as evidence about the attempted call, not about the underlying data" in policy
    assert "Never resend an equivalent failing payload" in policy
    assert "wait for its result before constructing the next call" in policy
    assert "empty result proves only that exact table and filter scope" in policy
    assert "observed facts separate from inference or recommendation" in policy
    assert "each requested measure, comparison" in policy
    assert "aggregate each requested measure with the same stated per-time-grain statistic" in policy
    assert "Name that statistic and its included periods" in policy
    assert "rank requested growth by delta rather than raw level" in policy
    assert "prose and charts from the same final table artifact" in policy
    assert "do not recompute measures inside a chart" in policy
    assert "sort by the exact cited measure" in policy
    assert "verify it against the reference distribution" in policy
    assert "otherwise report the value neutrally" in policy
    assert "must coexist in the same final evidence row" in policy
    assert "another entity's value" in policy
    assert "choose one directly observed comparable measure" in policy
    assert "invent a combined priority score" in policy
    assert "future action horizon is not an evidence window" in policy
    assert "observed baselines separate from proposed targets" in policy
    assert "If evidence is missing, report a partial outcome, not assumptions" in policy
    assert "prefer the newest authoritative evidence" in policy
    assert "resolve the conflict by authority and event date" in policy
    assert "direct source URL" in policy
    assert "search with today's year" in policy
    assert "historical report proves only its own reporting period" in policy
    assert "latest complete comparable window" in policy
    assert "claim visible in its snippet" in policy
    assert "every item must appear in retrieved passages" in policy
    assert "make a complementary query before answering" in policy
    assert "Never fill list gaps from memory" in policy
    assert "`json_path` and `schema_path`" in policy
    assert "native `targets=" in policy
    assert "singular `target`" in policy
    assert 'pass {"$artifact": "artifact_name"}' in policy
    assert "Reuse named artifacts published by the MCP result" in policy
    assert "Future observations are not required inputs" in policy
    assert "missing plan does not block the forecast" in policy
    assert "stored comparison evidence, never a substitute" in policy
    assert "selects its operation with `mode`, not `action`" in policy
    assert "do not copy dataframe dtype names into SQL casts" in policy
    assert "synthesize conclusion, interpretation, caveats" in policy
    assert "artifacts are separate" in policy
    assert "answer_ready" not in policy
    assert "do not call pandas only to sort, round, relabel, or format" in policy
    assert "mode `semantic_query`" in policy
    assert "Different metric base tables" in policy
    assert "complete period and final grain" in policy
    assert "executable contract including its metric dependencies" in policy
    assert "do not repeat `month`, `year`" in policy
    assert "conditionally aggregate measures in a CTE" in policy
    assert "not the scenario discriminator" in policy
    assert "aliases exposed by its immediate input CTE" in policy
    assert "ask for the formula instead of inventing one" in policy
    assert "unique, descriptive artifact key" in policy
    assert "never overwrite generic `result` or `table`" in policy
    assert "Publish a chart when requested or when it is the smallest useful artifact" in policy
    assert "never duplicate a sufficient table" in policy
    assert "explicit chart prohibition wins" in policy
    assert "categoryorder" in policy
    assert "`category_order` is not a Plotly axis property" in policy
    assert 'artifact_name="segment_metrics"' not in policy
    assert len(policy) < 8_000


def test_tool_data_flow_policy_can_request_a_plan_for_every_query() -> None:
    from backend.agent.services.message_builder import tool_data_flow_policy_block

    adaptive = tool_data_flow_policy_block({"update_plan", "sql_tool"})
    always = tool_data_flow_policy_block(
        {"update_plan", "sql_tool"},
        always_use_analysis_plan=True,
    )

    assert "Send a single-step request directly" in adaptive
    assert "checklist for every request" in always
    assert "before any other tool or final answer" in always
    assert "In later iterations" in always
    assert "alongside independent tool work" in always
    assert "mark work launched in the same batch as in_progress, not completed" in always
    assert "alongside independent tool work" in adaptive
    assert "never blocks an honest final or partial answer" in always


def test_execution_prompt_does_not_include_domain_specific_route_airport_rule() -> None:
    from backend.agent.services.message_builder import (
        ExecutionSystemPromptRequest,
        build_execution_system_prompt,
    )

    runner = AgentRunner()
    capability_context = {
        "source_mode": "db",
        "tool_descriptions": "",
        "available_tool_keys": ["database_tool", "sql_tool", "pandas_tool"],
    }

    prompt = build_execution_system_prompt(
        ExecutionSystemPromptRequest(
            settings=runner.settings,
            skill_registry=runner.skill_registry,
            enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
            capability_context=capability_context,
        )
    )

    assert "Маршруты и аэропорты" not in prompt
    assert "direct_info" not in prompt
    assert "origin/destination" not in prompt


def test_execution_prompt_keeps_planning_in_the_main_agent_loop() -> None:
    from backend.agent.services.message_builder import (
        ExecutionSystemPromptRequest,
        build_execution_context_messages,
        build_execution_system_prompt,
    )

    runner = AgentRunner()
    capability_context = {
        "source_mode": "dataset",
        "tool_descriptions": "",
        "available_tool_keys": ["get_tool_instructions", "sql_tool", "update_plan"],
        "prompt_block": "DYNAMIC_CAPABILITY_CONTEXT",
    }

    request = ExecutionSystemPromptRequest(
        settings=runner.settings,
        skill_registry=runner.skill_registry,
        enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
        capability_context=capability_context,
    )
    prompt = build_execution_system_prompt(request)
    context_text = "\n\n".join(str(message.content) for message in build_execution_context_messages(request))

    plan_instruction = "MUST call `update_plan` as the only tool call"
    assert plan_instruction in prompt
    assert "Before any data or retrieval tool" in prompt
    assert "If complexity becomes clear only after the first result" in prompt
    assert "add, remove, reorder, or replace remaining steps" in prompt
    assert "Send a single-step request directly to its sufficient tool" in prompt
    assert "active skill explicitly provides the matching executable route" not in prompt
    assert prompt.index(plan_instruction) < prompt.index("DYNAMIC_CAPABILITY_CONTEXT")
    assert "no source is specified and both capabilities are available, use both" in prompt
    assert "A bound table or database does not substitute" in prompt
    assert "do not reload the base `general_analytics` workflow" in prompt
    assert "semantic metric contract already provides the complete calculation" not in prompt
    assert "unless an active skill provides the matching executable route" not in prompt
    assert "SKILL_CATALOG_CONTEXT:" in prompt
    assert "SKILL_CATALOG_CONTEXT:" not in context_text


def test_execution_prompt_routes_knowledge_questions_by_requested_source() -> None:
    from backend.agent.services.message_builder import (
        ExecutionSystemPromptRequest,
        build_execution_system_prompt,
    )

    runner = AgentRunner()
    prompt = build_execution_system_prompt(
        ExecutionSystemPromptRequest(
            settings=runner.settings,
            skill_registry=runner.skill_registry,
            enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
            capability_context={
                "source_mode": "knowledge",
                "tool_descriptions": "",
                "available_tool_keys": ["rag_tool", "public_web_search"],
            },
        )
    )

    assert "explicitly asks to search the knowledge base" in prompt
    assert "Availability or session binding alone does not make a source exclusive" in prompt
    assert "explicitly asks for public internet or web sources" in prompt
    assert "no source is specified and both capabilities are available, use both" in prompt
    assert "a request for both requires both" in prompt


def test_data_context_contains_rag_no_context_clarification_rule() -> None:
    from backend.agent.services.message_builder import (
        ExecutionSystemPromptRequest,
        _build_data_context_messages,
    )

    runner = AgentRunner()
    request = ExecutionSystemPromptRequest(
        settings=runner.settings,
        skill_registry=runner.skill_registry,
        capability_context={"prompt_block": "x"},
        session_source={"source_type": "rag", "source_label": "Demo KB"},
        tool_db_runtime=None,
        df=None,
    )

    messages = _build_data_context_messages(request)
    assert messages, "RAG data context should include source guidance"
    assert any("no relevant context" in str(message.content).lower() for message in messages)


def test_execution_context_does_not_add_hidden_semantic_lookup_route() -> None:
    from backend.agent.services.message_builder import (
        ExecutionSystemPromptRequest,
        build_execution_context_messages,
    )

    runner = AgentRunner()
    request = ExecutionSystemPromptRequest(
        settings=runner.settings,
        skill_registry=runner.skill_registry,
        enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
        requested_tool_key=None,
        capability_context={
            "source_mode": "db",
            "tool_descriptions": "",
            "available_tool_keys": ["semantic_catalog_read_tool", "rag_tool", "public_web_search"],
        },
    )

    context_text = "\n\n".join(str(message.content) for message in build_execution_context_messages(request))

    assert "SEMANTIC_LOOKUP_CONTEXT" not in context_text
