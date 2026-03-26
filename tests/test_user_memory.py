"""Tests for UserMemoryService and the user_memories SQLite table."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from backend.auth import AuthDB, MEM_NOTES, MEM_PROFILE, UserMemory, UserMemoryService


def _make_db(tmpdir: str) -> AuthDB:
    return AuthDB(str(Path(tmpdir) / "app.db"), token_ttl_days=30)


class UserMemoryDBTests(unittest.TestCase):
    """Test the raw SQLite CRUD in AuthDB."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._db = _make_db(self._tmpdir)
        self._user = self._db.create_user("alice", "pass123", is_admin=False)

    def tearDown(self) -> None:
        del self._db
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_empty_memory_returns_empty_string(self) -> None:
        self.assertEqual(self._db.get_user_memory(self._user.id, MEM_PROFILE), "")
        self.assertEqual(self._db.get_user_memory(self._user.id, MEM_NOTES), "")

    def test_set_and_get_profile(self) -> None:
        self._db.set_user_memory(self._user.id, MEM_PROFILE, "Data scientist at Acme Corp")
        self.assertEqual(
            self._db.get_user_memory(self._user.id, MEM_PROFILE),
            "Data scientist at Acme Corp",
        )

    def test_overwrite_profile(self) -> None:
        self._db.set_user_memory(self._user.id, MEM_PROFILE, "first")
        self._db.set_user_memory(self._user.id, MEM_PROFILE, "second")
        self.assertEqual(self._db.get_user_memory(self._user.id, MEM_PROFILE), "second")

    def test_notes_independent_from_profile(self) -> None:
        self._db.set_user_memory(self._user.id, MEM_PROFILE, "profile text")
        self._db.set_user_memory(self._user.id, MEM_NOTES, "notes text")
        self.assertEqual(self._db.get_user_memory(self._user.id, MEM_PROFILE), "profile text")
        self.assertEqual(self._db.get_user_memory(self._user.id, MEM_NOTES), "notes text")

    def test_different_users_isolated(self) -> None:
        user2 = self._db.create_user("bob", "pass456", is_admin=False)
        self._db.set_user_memory(self._user.id, MEM_NOTES, "alice notes")
        self._db.set_user_memory(user2.id, MEM_NOTES, "bob notes")
        self.assertEqual(self._db.get_user_memory(self._user.id, MEM_NOTES), "alice notes")
        self.assertEqual(self._db.get_user_memory(user2.id, MEM_NOTES), "bob notes")

    def test_delete_user_cascades_memory(self) -> None:
        self._db.set_user_memory(self._user.id, MEM_PROFILE, "will be gone")
        self._db.delete_user(self._user.id)
        # After user deletion the row should be gone; get returns ""
        self.assertEqual(self._db.get_user_memory(self._user.id, MEM_PROFILE), "")


class UserMemoryServiceTests(unittest.TestCase):
    """Test UserMemoryService high-level API."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._db = _make_db(self._tmpdir)
        self._user = self._db.create_user("carol", "pass789", is_admin=False)
        self._svc = UserMemoryService(self._db)

    def tearDown(self) -> None:
        del self._db
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_load_returns_empty_by_default(self) -> None:
        mem = self._svc.load(self._user.id)
        self.assertEqual(mem.profile, "")
        self.assertEqual(mem.notes, "")
        self.assertTrue(mem.is_empty())

    def test_set_profile_persists(self) -> None:
        self._svc.set_profile(self._user.id, "  Senior analyst  ")
        mem = self._svc.load(self._user.id)
        self.assertEqual(mem.profile, "Senior analyst")

    def test_set_notes_persists(self) -> None:
        self._svc.set_notes(self._user.id, "Prefers bar charts")
        mem = self._svc.load(self._user.id)
        self.assertEqual(mem.notes, "Prefers bar charts")

    def test_append_note_adds_bullet(self) -> None:
        self._svc.append_note(self._user.id, "Uses monthly aggregations")
        mem = self._svc.load(self._user.id)
        self.assertIn("Uses monthly aggregations", mem.notes)

    def test_append_note_multiple_times(self) -> None:
        self._svc.append_note(self._user.id, "Note A")
        self._svc.append_note(self._user.id, "Note B")
        mem = self._svc.load(self._user.id)
        self.assertIn("Note A", mem.notes)
        self.assertIn("Note B", mem.notes)

    def test_is_empty_false_when_profile_set(self) -> None:
        self._svc.set_profile(self._user.id, "has profile")
        mem = self._svc.load(self._user.id)
        self.assertFalse(mem.is_empty())

    def test_is_empty_false_when_notes_set(self) -> None:
        self._svc.set_notes(self._user.id, "has notes")
        mem = self._svc.load(self._user.id)
        self.assertFalse(mem.is_empty())


class UserMemoryBuildBlockTests(unittest.TestCase):
    """Test the markdown block builder for system prompt injection."""

    def test_empty_memory_returns_empty_string(self) -> None:
        mem = UserMemory(profile="", notes="")
        self.assertEqual(mem.build_block(), "")

    def test_only_profile(self) -> None:
        mem = UserMemory(profile="Data engineer", notes="")
        block = mem.build_block()
        self.assertIn("## User memory", block)
        self.assertIn("User profile", block)
        self.assertIn("Data engineer", block)
        self.assertNotIn("Agent notes", block)

    def test_only_notes(self) -> None:
        mem = UserMemory(profile="", notes="- Prefers dark theme")
        block = mem.build_block()
        self.assertIn("## User memory", block)
        self.assertIn("Agent notes", block)
        self.assertIn("Prefers dark theme", block)
        self.assertNotIn("User profile", block)

    def test_both_sections(self) -> None:
        mem = UserMemory(profile="Analytics lead", notes="- Likes Plotly")
        block = mem.build_block()
        self.assertIn("User profile", block)
        self.assertIn("Agent notes", block)
        self.assertIn("Analytics lead", block)
        self.assertIn("Likes Plotly", block)

    def test_whitespace_only_treated_as_empty(self) -> None:
        mem = UserMemory(profile="   \n  ", notes="  ")
        self.assertEqual(mem.build_block(), "")


class UserMemoryConsolidationTests(unittest.TestCase):
    """Test schedule_consolidation fire-and-forget logic (sync mock)."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._db = _make_db(self._tmpdir)
        self._user = self._db.create_user("dave", "passabc", is_admin=False)
        self._svc = UserMemoryService(self._db)

    def tearDown(self) -> None:
        del self._db
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_schedule_consolidation_noop_when_no_notes(self) -> None:
        # Empty new_notes → should not crash, not schedule anything
        mock_llm = MagicMock()
        self._svc.schedule_consolidation(self._user.id, [], mock_llm)
        mock_llm.assert_not_called()

    def test_consolidation_merges_via_llm(self) -> None:
        """Direct call to _consolidate_async (sync via asyncio.run) verifies LLM is called."""
        import asyncio

        self._svc.set_notes(self._user.id, "- Old note")
        merged_result = MagicMock()
        merged_result.content = "- Old note\n- New insight"

        async def _run():
            def llm_invoke(msgs):  # noqa: ARG001
                return merged_result
            await self._svc._consolidate_async(  # noqa: SLF001
                self._user.id,
                ["New insight"],
                llm_invoke,
            )

        asyncio.run(_run())
        stored = self._db.get_user_memory(self._user.id, MEM_NOTES)
        self.assertIn("New insight", stored)

    def test_consolidation_handles_llm_error_gracefully(self) -> None:
        """If LLM raises, the existing notes must remain unchanged."""
        import asyncio

        self._svc.set_notes(self._user.id, "- Safe note")

        async def _run():
            def bad_llm(msgs):  # noqa: ARG001
                raise RuntimeError("LLM down")
            await self._svc._consolidate_async(  # noqa: SLF001
                self._user.id,
                ["Something new"],
                bad_llm,
            )

        asyncio.run(_run())
        stored = self._db.get_user_memory(self._user.id, MEM_NOTES)
        self.assertEqual(stored, "- Safe note")


if __name__ == "__main__":
    unittest.main()
