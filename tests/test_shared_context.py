"""Tests for SessionSandbox — persistent per-session execution environment."""

from __future__ import annotations

import multiprocessing
import time
import unittest

import pandas as pd

from backend.tools.sandbox import SessionSandbox
from backend.tools.sandbox_manager import SandboxManager


class SandboxScopePersistenceTests(unittest.TestCase):
    """Variables survive between execute() calls."""

    def test_variable_persists_across_calls(self) -> None:
        sb = SessionSandbox()
        sb.bind_dataframe(pd.DataFrame({"a": [1, 2, 3]}))

        sb.execute("x = df['a'].sum()", tool_name="pandas_tool")
        result = sb.execute("tool_result = {'v': {'total': x}}", tool_name="pandas_tool")

        assert isinstance(result, dict)
        assert result["v"]["total"] == 6

    def test_dataframe_available_after_bind(self) -> None:
        sb = SessionSandbox()
        df = pd.DataFrame({"col": [10, 20, 30]})
        sb.bind_dataframe(df, source_label="test.csv")

        result = sb.execute("tool_result = {'v': {'rows': len(df)}}", tool_name="pandas_tool")
        assert result["v"]["rows"] == 3

    def test_source_change_clears_derived_variables(self) -> None:
        sb = SessionSandbox()
        sb.bind_dataframe(pd.DataFrame({"a": [1]}), source_label="first.csv")
        sb.execute("stale_total = df['a'].sum()", tool_name="pandas_tool")

        replacement = pd.DataFrame({"b": [2]})
        sb.bind_dataframe(replacement, source_label="second.csv")

        assert "stale_total" not in sb.get_user_scope()
        assert sb.execute("tool_result = df", tool_name="pandas_tool").equals(replacement)

    def test_manager_reuses_sandbox_only_for_same_source_identity(self) -> None:
        manager = SandboxManager()
        first = manager.get_or_create_for_source(
            "session",
            {"source_type": "db_connection", "source_ref_id": "db-a"},
        )
        first.put("derived", pd.DataFrame({"value": [1]}))

        same = manager.get_or_create_for_source(
            "session",
            {
                "source_type": "db_connection",
                "source_ref_id": "db-a",
                "source_label": "renamed display label",
            },
        )

        assert same is first
        assert "derived" in same.get_user_scope()

        changed = manager.get_or_create_for_source(
            "session",
            {"source_type": "db_connection", "source_ref_id": "db-b"},
        )

        assert changed is not first
        assert "derived" not in changed.get_user_scope()

    def test_manager_clears_sandbox_when_source_is_unbound(self) -> None:
        manager = SandboxManager()
        bound = manager.get_or_create_for_source(
            "session",
            {"source_type": "csv", "source_ref_id": "dataset-a"},
        )
        bound.put("derived", pd.DataFrame({"value": [1]}))

        cleared = manager.get_or_create_for_source("session", {})

        assert cleared is not bound
        assert "derived" not in cleared.get_user_scope()

    def test_put_cannot_overwrite_infrastructure_dataframe(self) -> None:
        sb = SessionSandbox()
        original = pd.DataFrame({"source": [1]})
        sb.bind_dataframe(original)

        sb.put("df", pd.DataFrame({"foreign": [2]}))

        assert sb.execute("tool_result = df", tool_name="pandas_tool").equals(original)

    def test_plotly_available_when_include_plotly(self) -> None:
        sb = SessionSandbox()
        sb.bind_dataframe(pd.DataFrame({"x": [1], "y": [2]}))

        result = sb.execute(
            "fig = px.bar(df, x='x', y='y')\ntool_result = type(fig).__name__",
            tool_name="plotly_tool",
            include_plotly=True,
        )
        assert result == "Figure"


class SandboxDescribeForPromptTests(unittest.TestCase):
    def test_empty_sandbox_returns_empty(self) -> None:
        sb = SessionSandbox()
        assert sb.describe_for_prompt() == ""

    def test_variables_appear_in_prompt(self) -> None:
        sb = SessionSandbox()
        sb.bind_dataframe(pd.DataFrame({"region": ["A"], "revenue": [100]}))
        sb.execute("agg = df.groupby('region')['revenue'].sum()", tool_name="pandas_tool")

        prompt = sb.describe_for_prompt()
        assert "agg" in prompt

    def test_notebook_logs_data_source_change(self) -> None:
        sb = SessionSandbox()
        sb.bind_dataframe(pd.DataFrame({"a": [1]}), source_label="first.csv")
        sb.bind_dataframe(pd.DataFrame({"a": [1], "b": [2]}), source_label="second.csv")

        prompt = sb.describe_for_prompt()
        assert "second.csv" in prompt

    def test_uses_directly_instruction(self) -> None:
        sb = SessionSandbox()
        sb.bind_dataframe(pd.DataFrame({"a": [1]}))
        sb.execute("my_var = 42", tool_name="pandas_tool")

        prompt = sb.describe_for_prompt()
        assert "напрямую" in prompt


class SandboxTimeoutTests(unittest.TestCase):
    def test_timeout_terminates_worker_and_remains_usable(self) -> None:
        sb = SessionSandbox()
        sb.bind_dataframe(pd.DataFrame({"a": [1]}))
        children_before = {child.pid for child in multiprocessing.active_children()}

        with self.assertRaises(TimeoutError):
            # Use a busy-wait loop (no import needed) to trigger timeout.
            sb.execute("i = 0\nwhile True: i += 1", tool_name="test", timeout_sec=0.5)

        assert {child.pid for child in multiprocessing.active_children()} <= children_before
        result = sb.execute("tool_result = df['a'].sum()", tool_name="test")
        assert result == 1

    def test_late_timeout_write_cannot_reenter_new_scope(self) -> None:
        sb = SessionSandbox()
        with self.assertRaises(TimeoutError):
            sb.execute(
                "import time\ntime.sleep(0.05)\nlate_value = 99",
                tool_name="test",
                timeout_sec=0.01,
            )

        time.sleep(0.08)

        assert "late_value" not in sb.get_user_scope()


class SandboxResultExtractionTests(unittest.TestCase):
    def test_tool_result_priority(self) -> None:
        sb = SessionSandbox()
        sb.bind_dataframe(pd.DataFrame())
        result = sb.execute("tool_result = 'primary'\nresult = 'secondary'", tool_name="test")
        assert result == "primary"

    def test_last_expr_fallback(self) -> None:
        sb = SessionSandbox()
        sb.bind_dataframe(pd.DataFrame())
        result = sb.execute("42 + 1", tool_name="test")
        assert result == 43

    def test_result_alias(self) -> None:
        sb = SessionSandbox()
        sb.bind_dataframe(pd.DataFrame())
        result = sb.execute("result = 'hello'", tool_name="test")
        assert result == "hello"


class SandboxClearTests(unittest.TestCase):
    def test_clear_wipes_state(self) -> None:
        sb = SessionSandbox()
        sb.bind_dataframe(pd.DataFrame({"a": [1]}))
        sb.execute("x = 100", tool_name="test")
        sb.clear()

        assert sb.describe_for_prompt() == ""
        assert sb.execution_count == 0


class SandboxNotebookPersistenceTests(unittest.TestCase):
    def test_render_notebook_md_empty(self) -> None:
        sb = SessionSandbox()
        assert sb.render_notebook_md() == ""

    def test_render_notebook_md_has_entries(self) -> None:
        sb = SessionSandbox()
        sb.bind_dataframe(pd.DataFrame({"a": [1, 2]}), source_label="test.csv")
        sb.execute("x = df['a'].sum()", tool_name="pandas_tool")

        md = sb.render_notebook_md()
        assert "# Notebook" in md
        assert "test.csv" in md
        assert "pandas_tool" in md
        assert "```python" in md

    def test_persist_notebook_writes_file(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            sb = SessionSandbox()
            sb.set_storage_dir(Path(tmp))
            sb.bind_dataframe(pd.DataFrame({"a": [1]}), source_label="f.csv")
            sb.execute("x = 1", tool_name="test")

            nb_path = Path(tmp) / "notebook.md"
            assert nb_path.exists()
            content = nb_path.read_text(encoding="utf-8")
            assert "f.csv" in content
            assert "test" in content


class SkillsIntegrationTests(unittest.TestCase):
    """Verify that skill files load and pass registry validation."""

    def test_skills_dir_loads_without_error(self) -> None:
        from pathlib import Path

        from backend.skills import SkillRegistry

        skills_dir = Path(__file__).parent.parent / "skills"
        registry = SkillRegistry.from_path(skills_dir)
        skills = registry.list_skills()

        skill_ids = {s.skill_id for s in skills}
        assert "general_analytics" in skill_ids
        assert "plotly_tool" not in skill_ids
        assert "sql_tool" not in skill_ids

    def test_sql_tool_instructions_describe_data_flow(self) -> None:
        from backend.tools.instructions import get_default_tool_instruction_registry

        document = get_default_tool_instruction_registry().get("sql_tool")

        assert "mode" in document.body
        assert "artifact name" in document.body

    def test_tool_prompt_block_includes_plotly_instructions(self) -> None:
        from backend.tools.instructions import get_default_tool_instruction_registry

        block = get_default_tool_instruction_registry().build_brief_block({"plotly_tool", "sql_tool"})

        assert "get_tool_instructions" in block
        assert "plotly_tool" in block
        assert "sql_tool" in block
