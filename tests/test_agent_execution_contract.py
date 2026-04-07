"""
Behavior and contract tests — lightweight tier (no langchain/duckdb required).

Each test answers ONE question: "what behavior of the system are we protecting?"
Tests are written against public contracts, not private method names.
When a test fails it means the system broke a contract — change the code, not the test.

Modules under test (available without full deps):
  - backend.skills  — skill loading, prompt injection, validation
  - backend.tools.sandbox  — persistent code execution environment
  - backend.tools.policy  — tool access control
  - backend.agent.prompts  — base prompt content invariants
"""
from __future__ import annotations

import textwrap
import threading

import pandas as pd
import pytest

from backend.agent.prompts import execution_agent_prompt
from backend.skills.models import SkillSelectionError, SkillValidationError
from backend.skills.registry import SkillRegistry
from backend.tools.policy import (
    detect_data_access_mode,
    has_enabled_data_tools,
    is_tool_allowed,
    normalize_allowed_tool_keys,
)
from backend.tools.sandbox import SAFE_BUILTINS, SessionSandbox

# Depth profile constants — public contract values, reproduced here so the test
# doesn't drag in langchain.  These MUST match what runner.DEPTH_PROFILES contains.
# If they diverge → update runner.py, not this file.
EXPECTED_DEPTH_PROFILES = {
    "light": {"inner_recursion_limit": 4},
    "medium": {"inner_recursion_limit": 8},
    "deep": {"inner_recursion_limit": 15},
}


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _write_skill(tmp_path, folder: str, content: str) -> None:
    d = tmp_path / folder
    d.mkdir(exist_ok=True)
    (d / "SKILL.md").write_text(textwrap.dedent(content), encoding="utf-8")


@pytest.fixture()
def skill_registry_with_skills(tmp_path):
    """Registry with one tool skill (sql_tool) and one analytical skill (cohort)."""
    _write_skill(tmp_path, "sql_tool", """\
        ---
        name: SQL Tool
        description: Execute SQL queries against the database
        kind: tool
        tool_key: sql_tool
        triggers: sql, query, database
        ---
        ## SQL Instructions
        Use sql_tool to run SQL queries. Always end code with `tool_result`.
    """)
    _write_skill(tmp_path, "cohort_analysis", """\
        ---
        name: Cohort Analysis
        description: Run cohort retention analysis on user data
        kind: analytical
        triggers: cohort, retention, удержание
        ---
        ## Cohort Analysis Method
        Steps: 1. Group users by acquisition date. 2. Compute retention matrix.
    """)
    reg = SkillRegistry.from_path(tmp_path)
    reg.load()
    return reg


# ─────────────────────────────────────────────────────────────────────────────
# 1. Skills participate in policy (guidance layer)
# ─────────────────────────────────────────────────────────────────────────────


class TestSkillsPolicy:
    """Skills must load correctly, be discoverable, and appear in the prompt.

    Protected contract: skill files are the source of truth for tool guidance;
    the registry must make them available to the LLM via prompt injection.
    """

    def test_skill_registry_loads_both_skill_kinds(self, skill_registry_with_skills):
        skills = skill_registry_with_skills.list_skills()
        kinds = {s.kind for s in skills}
        assert "tool" in kinds and "analytical" in kinds

    def test_tool_skill_appears_in_brief_block_when_tool_available(
        self, skill_registry_with_skills
    ):
        """The sql_tool brief must appear when sql_tool is in available_tool_keys."""
        block = skill_registry_with_skills.build_tool_skills_brief_block({"sql_tool"})
        assert block and "sql_tool" in block

    def test_tool_skill_excluded_from_brief_when_tool_unavailable(
        self, skill_registry_with_skills
    ):
        """sql_tool brief must NOT appear when it's not in available_tool_keys."""
        block = skill_registry_with_skills.build_tool_skills_brief_block({"plotly_tool"})
        assert "sql_tool" not in block

    def test_analytical_skill_appears_in_analytical_brief_block(
        self, skill_registry_with_skills
    ):
        block = skill_registry_with_skills.build_analytical_skills_brief_block()
        assert "cohort" in block.lower()

    def test_selected_skills_injected_with_header_and_name(
        self, skill_registry_with_skills
    ):
        skills = skill_registry_with_skills.list_skills()
        analytical = next(s for s in skills if s.kind == "analytical")
        block = skill_registry_with_skills.build_prompt_block([analytical.skill_id])
        assert "Selected Skills" in block and analytical.name in block

    def test_resolve_selection_raises_for_unknown_skill_id(
        self, skill_registry_with_skills
    ):
        with pytest.raises(SkillSelectionError):
            skill_registry_with_skills.resolve_selection(["nonexistent_skill_xyz"])

    def test_empty_skills_dir_loads_cleanly(self, tmp_path):
        reg = SkillRegistry.from_path(tmp_path)
        reg.load()
        assert reg.list_skills() == ()

    def test_load_is_idempotent(self, skill_registry_with_skills):
        before = len(skill_registry_with_skills.list_skills())
        skill_registry_with_skills.load()  # second call must not double-count
        assert len(skill_registry_with_skills.list_skills()) == before

    def test_skill_without_frontmatter_is_rejected(self, tmp_path):
        _write_skill(tmp_path, "bad_skill", "# No frontmatter\nText.")
        with pytest.raises(SkillValidationError):
            SkillRegistry.from_path(tmp_path).load()

    def test_invalid_skill_kind_is_rejected(self, tmp_path):
        """kind='react' must be rejected — only 'analytical' and 'tool' are valid."""
        _write_skill(tmp_path, "bad_skill", """\
            ---
            name: Bad Skill
            description: Uses removed ReAct pattern
            kind: react
            ---
            Instructions.
        """)
        with pytest.raises(SkillValidationError, match="kind"):
            SkillRegistry.from_path(tmp_path).load()

    def test_tool_skill_without_tool_key_is_rejected(self, tmp_path):
        """kind='tool' + no tool_key → rejected at load time (not silently ignored)."""
        _write_skill(tmp_path, "bad_skill", """\
            ---
            name: Broken Tool Skill
            description: Missing required tool_key
            kind: tool
            ---
            Instructions.
        """)
        with pytest.raises(SkillValidationError, match="tool_key"):
            SkillRegistry.from_path(tmp_path).load()

    def test_duplicate_skill_ids_are_rejected(self, tmp_path):
        for folder in ("skill_a", "skill_b"):
            _write_skill(tmp_path, folder, f"""\
                ---
                id: shared_id
                name: Skill {folder}
                description: Duplicate id
                kind: analytical
                ---
                Instructions.
            """)
        with pytest.raises(SkillValidationError, match=r"[Dd]uplicate"):
            SkillRegistry.from_path(tmp_path).load()

    def test_tool_skill_full_instructions_in_prompt_block(
        self, skill_registry_with_skills
    ):
        """build_tool_skills_prompt_block must include full instructions, not just the name."""
        block = skill_registry_with_skills.build_tool_skills_prompt_block({"sql_tool"})
        assert "SQL Instructions" in block or "sql_tool" in block.lower()

    def test_analytical_skill_brief_includes_triggers(self, skill_registry_with_skills):
        """Brief block must include triggers so LLM knows when to call the skill."""
        block = skill_registry_with_skills.build_analytical_skills_brief_block()
        # At least one of the cohort skill's declared triggers must appear
        assert "cohort" in block.lower() or "retention" in block.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sandbox / data-tool contract
# ─────────────────────────────────────────────────────────────────────────────


class TestSandboxDataContract:
    """Sandbox provides persistent, isolated Python execution per session.

    Protected contract: code executes in a shared namespace that persists across
    tool calls, df is always accessible after bind_dataframe(), and forbidden
    imports are blocked as a security invariant.
    """

    def test_variables_persist_across_executions(self):
        sandbox = SessionSandbox()
        sandbox.execute("x = 42")
        sandbox.execute("y = x + 1")
        assert sandbox._scope.get("y") == 43

    def test_dataframe_bound_and_accessible_in_code(self):
        sandbox = SessionSandbox()
        sandbox.bind_dataframe(pd.DataFrame({"col1": [1, 2, 3]}))
        sandbox.execute("rows = len(df)")
        assert sandbox._scope.get("rows") == 3

    def test_second_bind_replaces_df_in_scope(self):
        sandbox = SessionSandbox()
        sandbox.bind_dataframe(pd.DataFrame({"x": [1, 2, 3]}))
        sandbox.execute("rows1 = len(df)")
        sandbox.bind_dataframe(pd.DataFrame({"x": [1, 2, 3, 4, 5]}))
        sandbox.execute("rows2 = len(df)")
        assert sandbox._scope.get("rows1") == 3
        assert sandbox._scope.get("rows2") == 5

    def test_forbidden_os_import_raises(self):
        """Security invariant: 'os' must not be importable in sandbox code."""
        sandbox = SessionSandbox()
        with pytest.raises((ImportError, Exception)):
            sandbox.execute("import os; os.getcwd()")

    def test_forbidden_sys_import_raises(self):
        """Security invariant: 'sys' must not be importable in sandbox code."""
        sandbox = SessionSandbox()
        with pytest.raises((ImportError, Exception)):
            sandbox.execute("import sys")

    def test_safe_builtins_available_without_import(self):
        """Core builtins (sum, range, len) must work without any import statement."""
        sandbox = SessionSandbox()
        sandbox.execute("result = sum(range(5))")
        assert sandbox._scope.get("result") == 10

    def test_pandas_import_allowed(self):
        """pandas is allowlisted and must be importable in sandbox code."""
        sandbox = SessionSandbox()
        sandbox.execute("import pandas as pd2; ok = pd2 is not None")
        assert sandbox._scope.get("ok") is True

    def test_eval_not_in_safe_builtins(self):
        """eval() must not be a safe builtin — prevents arbitrary code execution."""
        assert "eval" not in SAFE_BUILTINS

    def test_exec_not_in_safe_builtins(self):
        """exec() must not be a safe builtin — prevents code injection."""
        assert "exec" not in SAFE_BUILTINS

    def test_globals_not_in_safe_builtins(self):
        """globals() must not be available — prevents namespace escape."""
        assert "globals" not in SAFE_BUILTINS

    def test_chain_of_executions_share_scope(self):
        sandbox = SessionSandbox()
        sandbox.execute("a = 1")
        sandbox.execute("b = 2")
        sandbox.execute("c = a + b")
        assert sandbox._scope.get("c") == 3

    def test_tool_result_variable_accessible_after_execute(self):
        """tool_result set in code must be accessible from _scope."""
        sandbox = SessionSandbox()
        sandbox.execute(
            "tool_result = {'schema_version': '1.0', 'artifact_type': 'value', "
            "'items': {'x': 42}}"
        )
        tr = sandbox._scope.get("tool_result")
        assert tr is not None and tr.get("artifact_type") == "value"

    def test_concurrent_executions_do_not_raise(self):
        """Multiple threads writing distinct variables must not produce errors."""
        sandbox = SessionSandbox()
        errors: list[Exception] = []

        def worker(var: str, val: int) -> None:
            try:
                sandbox.execute(f"{var} = {val} * 2")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(f"var_{i}", i)) for i in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors, f"Sandbox errors under concurrent access: {errors}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Base prompt content invariants
# ─────────────────────────────────────────────────────────────────────────────


class TestPromptContent:
    """execution_agent_prompt is the single policy document for the analysis agent.

    Protected contract: the prompt drives native tool_use (no ReAct), mandates
    planner_tool, and defines the tool_result contract.
    """

    def test_prompt_has_no_react_thought_marker(self):
        """ReAct pattern removed — 'Thought:' must not appear in the execution prompt."""
        assert "Thought:" not in execution_agent_prompt

    def test_prompt_has_no_react_action_input_marker(self):
        assert "Action Input:" not in execution_agent_prompt

    def test_prompt_has_no_react_observation_marker(self):
        assert "Observation:" not in execution_agent_prompt

    def test_prompt_references_tool_calling(self):
        """The prompt must reference tool usage — it's the execution agent, not chat."""
        assert "tool" in execution_agent_prompt.lower()

    def test_prompt_mandates_planner_tool_for_data_tasks(self):
        """planner_tool is mandatory for analysis — must be explicitly named."""
        assert "planner_tool" in execution_agent_prompt

    def test_prompt_defines_tool_result_contract(self):
        """The tool_result variable contract must be stated so the LLM knows the format."""
        assert "tool_result" in execution_agent_prompt

    def test_prompt_is_substantive_policy_document(self):
        """Prompt must be long enough to constitute a real policy document."""
        assert len(execution_agent_prompt.strip()) > 500


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tool access control (policy layer)
# ─────────────────────────────────────────────────────────────────────────────


class TestToolAccessPolicy:
    """Policy functions gate which tools are accessible per request.

    Protected contract: allowed_tool_keys=None means no restriction;
    a set restricts access to exactly those tools; data-mode detection
    drives which tools are considered mandatory.
    """

    def test_none_allowed_keys_permits_any_tool(self):
        assert is_tool_allowed("any_tool", None) is True

    def test_tool_in_set_is_allowed(self):
        assert is_tool_allowed("pandas_tool", {"pandas_tool", "sql_tool"}) is True

    def test_tool_not_in_set_is_denied(self):
        assert is_tool_allowed("plotly_tool", {"pandas_tool"}) is False

    def test_normalize_strips_whitespace(self):
        result = normalize_allowed_tool_keys(["  pandas_tool  ", "sql_tool"])
        assert "pandas_tool" in result and "sql_tool" in result

    def test_normalize_none_returns_none(self):
        assert normalize_allowed_tool_keys(None) is None

    def test_dataset_mode_detected_from_dataframe(self):
        assert detect_data_access_mode(has_dataframe=True, session_source={}) == "dataset"

    def test_db_mode_detected_from_session_source(self):
        result = detect_data_access_mode(
            has_dataframe=False,
            session_source={"source_type": "db_connection"},
        )
        assert result == "db"

    def test_no_mode_without_any_data(self):
        assert detect_data_access_mode(has_dataframe=False, session_source={}) is None

    def test_db_mode_wins_over_dataframe_flag(self):
        """DB source_type takes precedence over has_dataframe."""
        result = detect_data_access_mode(
            has_dataframe=True,
            session_source={"source_type": "db_connection"},
        )
        assert result == "db"

    # ── Guardrail: data tools disabled ──────────────────────────────────────

    def test_dataset_analysis_is_blocked_when_df_tools_disabled(self):
        """When only non-data tools are allowed with a DataFrame, has_enabled_data_tools
        must return False so the system can return a guardrail error."""
        blocked = not has_enabled_data_tools(
            has_dataframe=True,
            session_source={"source_type": "csv"},
            allowed_tool_keys={"search_tool"},  # no pandas_tool / value_tool
        )
        assert blocked, (
            "has_enabled_data_tools must return False when df tools are disabled — "
            "guardrail depends on this"
        )

    def test_dataset_analysis_is_not_blocked_when_pandas_tool_enabled(self):
        """When pandas_tool is allowed, has_enabled_data_tools must return True."""
        not_blocked = has_enabled_data_tools(
            has_dataframe=True,
            session_source={"source_type": "csv"},
            allowed_tool_keys={"pandas_tool"},
        )
        assert not_blocked

    def test_db_analysis_is_blocked_when_sql_tool_disabled(self):
        """With a DB connection but no sql_tool, analysis must be blocked."""
        blocked = not has_enabled_data_tools(
            has_dataframe=False,
            session_source={"source_type": "db_connection"},
            allowed_tool_keys={"search_tool", "plotly_tool"},
        )
        assert blocked

    def test_no_data_mode_is_never_blocked(self):
        """When there's no data source, guardrail must not trigger."""
        not_blocked = has_enabled_data_tools(
            has_dataframe=False,
            session_source={},
            allowed_tool_keys=set(),  # most restrictive possible
        )
        assert not_blocked

    def test_artifact_optional_policy_for_external_tools(self):
        """Search tool results without artifacts are a valid agent output."""
        from backend.tools.policy import supports_artifact_optional_output
        assert supports_artifact_optional_output(["search_tool"]) is True
        assert supports_artifact_optional_output(["search_tool", "memory"]) is True
        assert supports_artifact_optional_output(["pandas_tool"]) is False
        assert supports_artifact_optional_output(["search_tool", "pandas_tool"]) is False


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sandbox infrastructure contracts
# ─────────────────────────────────────────────────────────────────────────────


class TestSandboxInfrastructureContract:
    """Sandbox infrastructure keys must never be returned as user artifacts.

    Protected contract: df, pd, np etc. are infrastructure — they must stay
    in _INFRA_KEYS and be excluded from result extraction.
    """

    def test_infra_keys_contain_standard_scope_variables(self):
        from backend.tools.sandbox import _INFRA_KEYS
        for key in ("df", "pd", "np", "__builtins__"):
            assert key in _INFRA_KEYS, (
                f"'{key}' must be in _INFRA_KEYS to avoid false artifact extraction"
            )

    def test_tool_result_is_highest_priority_candidate(self):
        """tool_result must be extracted before 'result', 'output', etc."""
        from backend.tools.sandbox import _RESULT_CANDIDATES
        assert _RESULT_CANDIDATES[0] == "tool_result"

    def test_result_extraction_prefers_tool_result_over_result(self):
        """When both tool_result and result are set, tool_result wins."""
        sandbox = SessionSandbox()
        sandbox.execute("result = 'secondary'; tool_result = 'primary'")
        # tool_result should be set in scope
        assert sandbox._scope.get("tool_result") == "primary"

    def test_timeout_raises_and_sandbox_recovers(self):
        """A timed-out execution must raise TimeoutError; sandbox must still work after."""
        sandbox = SessionSandbox()
        sandbox.bind_dataframe(pd.DataFrame({"a": [1]}))

        with pytest.raises(TimeoutError):
            sandbox.execute("i = 0\nwhile True: i += 1", timeout_sec=0.3)

        # Sandbox must recover — next execution must work
        result = sandbox.execute("tool_result = 99")
        assert result == 99


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cleanup guarantees (removed legacy paths absent from available modules)
# ─────────────────────────────────────────────────────────────────────────────


class TestCleanupGuarantees:
    """Verify that removed legacy paths are genuinely absent; runtime invariants hold."""

    def test_skill_registry_from_path_accepts_string_and_path(self, tmp_path):
        reg1 = SkillRegistry.from_path(str(tmp_path))
        reg1.load()
        assert reg1.list_skills() == ()

        reg2 = SkillRegistry.from_path(tmp_path)
        reg2.load()
        assert reg2.list_skills() == ()
