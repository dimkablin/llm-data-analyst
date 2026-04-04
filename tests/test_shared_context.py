"""Tests for SessionSandbox — persistent per-session execution environment."""
from __future__ import annotations

import unittest

import pandas as pd

from backend.tools.sandbox import SessionSandbox


class SandboxScopePersistenceTests(unittest.TestCase):
    """Variables survive between execute() calls."""

    def test_variable_persists_across_calls(self) -> None:
        sb = SessionSandbox()
        sb.bind_dataframe(pd.DataFrame({"a": [1, 2, 3]}))

        sb.execute("x = df['a'].sum()", tool_name="pandas_tool")
        result = sb.execute("tool_result = {'v': {'total': x}}", tool_name="value_tool")

        assert isinstance(result, dict)
        assert result["v"]["total"] == 6

    def test_dataframe_available_after_bind(self) -> None:
        sb = SessionSandbox()
        df = pd.DataFrame({"col": [10, 20, 30]})
        sb.bind_dataframe(df, source_label="test.csv")

        result = sb.execute("tool_result = {'v': {'rows': len(df)}}", tool_name="value_tool")
        assert result["v"]["rows"] == 3

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
    def test_timeout_raises_and_resets(self) -> None:
        sb = SessionSandbox()
        sb.bind_dataframe(pd.DataFrame({"a": [1]}))

        with self.assertRaises(TimeoutError):
            # Use a busy-wait loop (no import needed) to trigger timeout.
            sb.execute("i = 0\nwhile True: i += 1", tool_name="test", timeout_sec=0.5)

        # Sandbox should still work after timeout (scope was reset).
        result = sb.execute("tool_result = 42", tool_name="test")
        assert result == 42


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
        assert "plotly_tool" in skill_ids
        assert "sql_tool" in skill_ids

    def test_sql_tool_skill_describes_data_flow(self) -> None:
        from pathlib import Path

        from backend.skills import SkillRegistry

        skills_dir = Path(__file__).parent.parent / "skills"
        registry = SkillRegistry.from_path(skills_dir)
        skill = registry.get("sql_tool")

        assert "db.query_dataframe" in skill.instructions_markdown

    def test_tool_skills_prompt_block_includes_plotly_instructions(self) -> None:
        from pathlib import Path

        from backend.skills import SkillRegistry

        skills_dir = Path(__file__).parent.parent / "skills"
        registry = SkillRegistry.from_path(skills_dir)
        block = registry.build_tool_skills_prompt_block({"plotly_tool", "sql_tool"})

        assert "db.query_dataframe" in block


if __name__ == "__main__":
    unittest.main()
