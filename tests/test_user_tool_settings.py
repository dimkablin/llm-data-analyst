from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.api.models import ToolAvailabilityResponse
from backend.auth import AuthDB
from backend.tools import build_tool_catalog, effective_enabled_tool_keys, is_tool_allowed


class UserToolSettingsTests(unittest.TestCase):
    def test_auth_db_persists_user_tool_settings(self) -> None:
        tmpdir = tempfile.mkdtemp()
        auth_db = None
        try:
            db_path = str(Path(tmpdir) / "app.db")
            auth_db = AuthDB(db_path, token_ttl_days=30)
            user = auth_db.create_user("alice_test", "secret", is_admin=False)

            self.assertEqual(auth_db.list_user_tool_settings(user.id), {})

            auth_db.set_user_tool_enabled(user.id, "rag_tool", False)
            auth_db.set_user_tool_enabled(user.id, "plotly_tool", True)

            stored = auth_db.list_user_tool_settings(user.id)
            self.assertEqual(stored["rag_tool"], False)
            self.assertEqual(stored["plotly_tool"], True)
        finally:
            if auth_db is not None:
                del auth_db
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_tool_catalog_builds_effective_user_state(self) -> None:
        source_descriptors = [
            {
                "source_type": "rag",
                "source_ref_id": "rag",
                "source_label": "RAG",
                "source_mode": "external",
                "enabled": True,
                "available": True,
                "status": "available",
                "description": "RAG",
                "capabilities": ["knowledge_base_search"],
                "requires_session_data": False,
                "timeout_hint_sec": 25.0,
            },
            {
                "source_type": "forecast",
                "source_ref_id": "forecast",
                "source_label": "Forecast",
                "source_mode": "external",
                "enabled": True,
                "available": False,
                "status": "misconfigured",
                "description": "Forecast",
                "capabilities": ["forecast"],
                "requires_session_data": True,
                "timeout_hint_sec": 45.0,
            },
            {
                "source_type": "anomaly_planfact",
                "source_ref_id": "anomaly_planfact",
                "source_label": "Anomaly",
                "source_mode": "external",
                "enabled": True,
                "available": True,
                "status": "available",
                "description": "Anomaly",
                "capabilities": ["anomaly_detection"],
                "requires_session_data": True,
                "timeout_hint_sec": 50.0,
            },
        ]

        rows = build_tool_catalog(
            source_descriptors=source_descriptors,
            user_settings={
                "rag_tool": False,
                "plotly_tool": False,
                "forecast_tool": True,
            },
        )
        payloads = [ToolAvailabilityResponse(**item) for item in rows]
        by_key = {item.tool_key: item for item in payloads}

        self.assertFalse(by_key["rag_tool"].enabled_for_user)
        self.assertFalse(by_key["rag_tool"].effective_enabled)
        self.assertFalse(by_key["plotly_tool"].effective_enabled)
        self.assertFalse(by_key["forecast_tool"].available_globally)
        self.assertFalse(by_key["forecast_tool"].effective_enabled)
        self.assertTrue(by_key["anomaly_planfact_tool"].effective_enabled)
        self.assertTrue(by_key["sql_tool"].enabled_by_default)
        self.assertIn("SQL", by_key["sql_tool"].display_name_ru)

    def test_tool_catalog_uses_tool_markdown_metadata(self) -> None:
        rows = build_tool_catalog(source_descriptors=[], user_settings={})
        by_key = {str(item["tool_key"]): item for item in rows}

        self.assertTrue(by_key["sql_tool"]["description"].startswith("Run read-only SQL"))
        self.assertTrue(by_key["sql_tool"]["enabled_by_default"])

    def test_tool_catalog_does_not_hide_broken_instruction_registry(self) -> None:
        from backend.tools import catalog as tool_catalog

        def _broken_registry():
            raise RuntimeError("broken tool metadata")

        original = tool_catalog.get_default_tool_instruction_registry
        tool_catalog.get_default_tool_instruction_registry = _broken_registry
        try:
            with self.assertRaisesRegex(RuntimeError, "broken tool metadata"):
                build_tool_catalog(source_descriptors=[], user_settings={})
        finally:
            tool_catalog.get_default_tool_instruction_registry = original

    def test_effective_tool_keys_feed_runtime_policy_layer(self) -> None:
        source_descriptors = [
            {
                "source_type": "rag",
                "source_ref_id": "rag",
                "source_label": "RAG",
                "source_mode": "external",
                "enabled": True,
                "available": True,
                "status": "available",
                "description": "RAG",
                "capabilities": ["knowledge_base_search"],
                "requires_session_data": False,
                "timeout_hint_sec": 25.0,
            },
            {
                "source_type": "forecast",
                "source_ref_id": "forecast",
                "source_label": "Forecast",
                "source_mode": "external",
                "enabled": True,
                "available": True,
                "status": "available",
                "description": "Forecast",
                "capabilities": ["forecast"],
                "requires_session_data": True,
                "timeout_hint_sec": 45.0,
            },
            {
                "source_type": "anomaly_planfact",
                "source_ref_id": "anomaly_planfact",
                "source_label": "Anomaly",
                "source_mode": "external",
                "enabled": True,
                "available": True,
                "status": "available",
                "description": "Anomaly",
                "capabilities": ["anomaly_detection"],
                "requires_session_data": True,
                "timeout_hint_sec": 50.0,
            },
        ]
        catalog = build_tool_catalog(
            source_descriptors=source_descriptors,
            user_settings={
                "rag_tool": False,
                "plotly_tool": False,
            },
        )

        allowed_tool_keys = effective_enabled_tool_keys(catalog)

        self.assertFalse(is_tool_allowed("rag_tool", allowed_tool_keys))
        self.assertFalse(is_tool_allowed("plotly_tool", allowed_tool_keys))
        self.assertTrue(is_tool_allowed("forecast_tool", allowed_tool_keys))
        self.assertTrue(is_tool_allowed("sql_tool", allowed_tool_keys))
