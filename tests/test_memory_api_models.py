"""Tests for UserMemory Pydantic models and API contract."""
from __future__ import annotations

import unittest

from backend.models import UserMemoryResponse, UserMemoryUpdateRequest


class UserMemoryResponseTests(unittest.TestCase):
    def test_valid_full(self) -> None:
        r = UserMemoryResponse(profile="Data engineer", notes="- Prefers dark theme")
        self.assertEqual(r.profile, "Data engineer")
        self.assertEqual(r.notes, "- Prefers dark theme")

    def test_empty_strings_valid(self) -> None:
        r = UserMemoryResponse(profile="", notes="")
        self.assertEqual(r.profile, "")
        self.assertEqual(r.notes, "")

    def test_serialises_to_dict(self) -> None:
        r = UserMemoryResponse(profile="a", notes="b")
        d = r.model_dump()
        self.assertEqual(d, {"profile": "a", "notes": "b"})


class UserMemoryUpdateRequestTests(unittest.TestCase):
    def test_patch_profile_only(self) -> None:
        req = UserMemoryUpdateRequest(profile="new profile")
        self.assertEqual(req.profile, "new profile")
        self.assertIsNone(req.notes)

    def test_patch_notes_only(self) -> None:
        req = UserMemoryUpdateRequest(notes="new notes")
        self.assertIsNone(req.profile)
        self.assertEqual(req.notes, "new notes")

    def test_both_none_is_valid_noop_patch(self) -> None:
        req = UserMemoryUpdateRequest()
        self.assertIsNone(req.profile)
        self.assertIsNone(req.notes)

    def test_both_fields_set(self) -> None:
        req = UserMemoryUpdateRequest(profile="p", notes="n")
        self.assertEqual(req.profile, "p")
        self.assertEqual(req.notes, "n")


if __name__ == "__main__":
    unittest.main()
