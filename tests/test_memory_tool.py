"""Tests for MemoryTool and MemoryToolFactory."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from backend.tools.impl.factory import MemoryToolFactory
from backend.tools.impl.memory_tool import MemoryTool
from backend.core import Settings
from backend.tools import ToolBuildContext, ToolRegistry


class MemoryToolTests(unittest.TestCase):
    def test_save_note_calls_callback(self) -> None:
        collected: list[str] = []
        tool = MemoryTool(on_note=collected.append)
        result = tool._run("User prefers monthly aggregations")  # noqa: SLF001
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0], "User prefers monthly aggregations")
        self.assertIn("Saved to user memory", result)

    def test_empty_note_not_saved(self) -> None:
        collected: list[str] = []
        tool = MemoryTool(on_note=collected.append)
        result = tool._run("   ")  # noqa: SLF001
        self.assertEqual(len(collected), 0)
        self.assertIn("empty", result)

    def test_note_stripped_before_callback(self) -> None:
        collected: list[str] = []
        tool = MemoryTool(on_note=collected.append)
        tool._run("  some note  ")  # noqa: SLF001
        self.assertEqual(collected[0], "some note")

    def test_long_note_truncated_in_confirmation(self) -> None:
        collected: list[str] = []
        tool = MemoryTool(on_note=collected.append)
        long_note = "x" * 200
        result = tool._run(long_note)  # noqa: SLF001
        self.assertEqual(len(collected[0]), 200)
        self.assertTrue(result.startswith("Saved to user memory: "))
        self.assertLess(len(result), len("Saved to user memory: ") + 130)

    def test_multiple_calls_all_collected(self) -> None:
        collected: list[str] = []
        tool = MemoryTool(on_note=collected.append)
        tool._run("Note 1")  # noqa: SLF001
        tool._run("Note 2")  # noqa: SLF001
        tool._run("Note 3")  # noqa: SLF001
        self.assertEqual(len(collected), 3)
        self.assertIn("Note 2", collected)

    def test_tool_name_is_memory(self) -> None:
        tool = MemoryTool(on_note=lambda _: None)
        self.assertEqual(tool.name, "memory")


class MemoryToolFactoryTests(unittest.TestCase):
    def _make_ctx(self) -> ToolBuildContext:
        return ToolBuildContext(settings=MagicMock(spec=Settings))

    def test_is_always_available(self) -> None:
        factory = MemoryToolFactory(on_note=lambda _: None)
        self.assertTrue(factory.is_available(self._make_ctx()))

    def test_build_returns_memory_tool(self) -> None:
        factory = MemoryToolFactory(on_note=lambda _: None)
        tool = factory.build(self._make_ctx())
        self.assertIsInstance(tool, MemoryTool)

    def test_build_wires_callback(self) -> None:
        collected: list[str] = []
        factory = MemoryToolFactory(on_note=collected.append)
        tool = factory.build(self._make_ctx())
        tool._run("wired note")  # noqa: SLF001
        self.assertIn("wired note", collected)

    def test_key_is_memory_tool(self) -> None:
        factory = MemoryToolFactory(on_note=lambda _: None)
        self.assertEqual(factory.key, "memory_tool")


class MemoryToolRegistryTests(unittest.TestCase):
    def _make_registry(self, **kwargs) -> ToolRegistry:
        return ToolRegistry.from_services(**kwargs)

    def _make_ctx(self) -> ToolBuildContext:
        return ToolBuildContext(settings=MagicMock(spec=Settings))

    def test_memory_tool_in_registry_no_services(self) -> None:
        registry = self._make_registry()
        ctx = self._make_ctx()
        self.assertTrue(registry.is_available("memory_tool", ctx))

    def test_memory_tool_built_in_build_tools(self) -> None:
        registry = self._make_registry()
        ctx = self._make_ctx()
        tools = registry.build_tools(ctx)
        tool_names = [t.name for t in tools]
        self.assertIn("memory", tool_names)

    def test_custom_callback_flows_through_registry(self) -> None:
        collected: list[str] = []
        registry = self._make_registry(memory_note_callback=collected.append)
        ctx = self._make_ctx()
        tools = registry.build_tools(ctx)
        mem_tool = next(t for t in tools if t.name == "memory")
        mem_tool._run("registry callback test")  # noqa: SLF001
        self.assertIn("registry callback test", collected)


if __name__ == "__main__":
    unittest.main()
