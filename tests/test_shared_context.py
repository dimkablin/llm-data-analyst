from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backend.tools.shared_context import SharedContext, SharedVarMeta


class SharedContextDescribeForPromptTests(unittest.TestCase):
    def test_empty_context_returns_empty_string(self) -> None:
        ctx = SharedContext()
        assert ctx.describe_for_prompt() == ""

    def test_dataframe_columns_appear_in_prompt(self) -> None:
        ctx = SharedContext()
        df = pd.DataFrame({"region": ["A", "B"], "revenue": [100, 200]})
        ctx.put("shared_agg", df, "pandas_tool")

        prompt = ctx.describe_for_prompt()

        assert "shared_agg" in prompt
        assert "region" in prompt
        assert "revenue" in prompt

    def test_dataframe_shape_appears_in_prompt(self) -> None:
        ctx = SharedContext()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ctx.put("shared_df", df, "pandas_tool")

        prompt = ctx.describe_for_prompt()

        assert "(3, 2)" in prompt

    def test_scalar_has_no_columns_in_prompt(self) -> None:
        ctx = SharedContext()
        ctx.put("shared_total", 42.0, "value_tool")

        prompt = ctx.describe_for_prompt()

        assert "shared_total" in prompt
        assert "columns" not in prompt

    def test_series_name_appears_as_column(self) -> None:
        ctx = SharedContext()
        s = pd.Series([1, 2, 3], name="sales")
        ctx.put("shared_sales", s, "pandas_tool")

        prompt = ctx.describe_for_prompt()

        assert "sales" in prompt

    def test_producer_tool_appears_in_prompt(self) -> None:
        ctx = SharedContext()
        df = pd.DataFrame({"x": [1]})
        ctx.put("shared_x", df, "pandas_tool")

        prompt = ctx.describe_for_prompt()

        assert "pandas_tool" in prompt

    def test_multiple_vars_all_appear(self) -> None:
        ctx = SharedContext()
        ctx.put("shared_agg", pd.DataFrame({"cat": ["A"], "val": [1]}), "pandas_tool")
        ctx.put("shared_count", 5, "value_tool")

        prompt = ctx.describe_for_prompt()

        assert "shared_agg" in prompt
        assert "shared_count" in prompt

    def test_footer_instruction_present(self) -> None:
        ctx = SharedContext()
        ctx.put("shared_x", 1, "pandas_tool")

        prompt = ctx.describe_for_prompt()

        assert "Используй их напрямую" in prompt


class SharedVarMetaColumnsTests(unittest.TestCase):
    def test_extract_columns_from_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        ctx = SharedContext()
        ctx.put("shared_df", df, "tool")

        meta = ctx.describe()[0]
        assert meta.columns == ["a", "b", "c"]

    def test_extract_columns_empty_for_scalar(self) -> None:
        ctx = SharedContext()
        ctx.put("shared_val", 99, "tool")

        meta = ctx.describe()[0]
        assert meta.columns == []

    def test_extract_columns_empty_for_list(self) -> None:
        ctx = SharedContext()
        ctx.put("shared_list", [1, 2, 3], "tool")

        meta = ctx.describe()[0]
        assert meta.columns == []

    def test_extract_columns_from_named_series(self) -> None:
        s = pd.Series([10, 20], name="revenue")
        ctx = SharedContext()
        ctx.put("shared_revenue", s, "tool")

        meta = ctx.describe()[0]
        assert meta.columns == ["revenue"]

    def test_extract_columns_empty_for_unnamed_series(self) -> None:
        s = pd.Series([1, 2, 3])
        ctx = SharedContext()
        ctx.put("shared_s", s, "tool")

        meta = ctx.describe()[0]
        assert meta.columns == []


class SharedContextSkillsIntegrationTests(unittest.TestCase):
    """Verify that skill files load and plotly_tool.md passes SkillRegistry validation."""

    def test_skills_dir_loads_without_error(self) -> None:
        from pathlib import Path
        from backend.skills import SkillRegistry

        skills_dir = Path(__file__).parent.parent / "skills"
        registry = SkillRegistry.from_path(skills_dir)
        skills = registry.list_skills()

        skill_ids = {s.skill_id for s in skills}
        assert "plotly_tool" in skill_ids
        assert "sql_table_tool" in skill_ids

    def test_plotly_tool_skill_describes_shared_vars(self) -> None:
        from pathlib import Path
        from backend.skills import SkillRegistry

        skills_dir = Path(__file__).parent.parent / "skills"
        registry = SkillRegistry.from_path(skills_dir)
        skill = registry.get("plotly_tool")

        assert "shared_" in skill.instructions_markdown
        assert "db.query_dataframe" in skill.instructions_markdown

    def test_sql_table_tool_skill_describes_data_flow(self) -> None:
        from pathlib import Path
        from backend.skills import SkillRegistry

        skills_dir = Path(__file__).parent.parent / "skills"
        registry = SkillRegistry.from_path(skills_dir)
        skill = registry.get("sql_table_tool")

        assert "shared_context" in skill.instructions_markdown
        assert "db.query_dataframe" in skill.instructions_markdown

    def test_tool_skills_prompt_block_includes_plotly_instructions(self) -> None:
        from pathlib import Path
        from backend.skills import SkillRegistry

        skills_dir = Path(__file__).parent.parent / "skills"
        registry = SkillRegistry.from_path(skills_dir)
        block = registry.build_tool_skills_prompt_block({"plotly_tool", "sql_table_tool"})

        assert "db.query_dataframe" in block
        assert "shared_" in block
        assert "Сценарий" in block


if __name__ == "__main__":
    unittest.main()
