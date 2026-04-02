"""Kernel lifecycle — ephemeral execution runtime over notebook.

The kernel is a ``SessionSandbox`` that can be created, evicted, and
restored at any time.  The notebook is the source of truth; the kernel
is a performance cache.

Typical flow::

    1. Query arrives for session_id
    2. KernelManager.get_or_restore(session_id)
       → hot kernel exists?  return it
       → no kernel?  create one, replay notebook cells
    3. Tool executes code in kernel (sandbox.execute)
    4. Result recorded as NotebookCell via orchestrator
    5. After 3 hours of inactivity, kernel is evicted
    6. Next query triggers step 2 again
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from backend.notebook.manifest_store import ManifestStore
from backend.notebook.models import NotebookDocument
from backend.notebook.session_source import SessionManifest, SessionSource
from backend.notebook.store import NotebookStore

logger = logging.getLogger(__name__)

# Default kernel TTL: 3 hours.
DEFAULT_KERNEL_TTL_SEC = 3 * 3600


@dataclass
class KernelState:
    """Metadata about a live kernel — not persisted to disk."""

    session_id: str
    last_execution: float = field(default_factory=time.monotonic)
    cell_execution_count: int = 0
    restored: bool = False
    restore_errors: list[str] = field(default_factory=list)


class KernelManager:
    """Manages ephemeral kernel lifecycles for notebook sessions.

    Wraps ``SandboxManager`` with restore-from-notebook logic.
    """

    def __init__(
        self,
        notebook_store: NotebookStore,
        manifest_store: ManifestStore,
        storage_dir: str | Path,
        *,
        ttl_sec: float = DEFAULT_KERNEL_TTL_SEC,
    ) -> None:
        self._notebook_store = notebook_store
        self._manifest_store = manifest_store
        self._storage_dir = Path(storage_dir)
        self._ttl_sec = ttl_sec
        self._kernel_states: dict[str, KernelState] = {}

    # ── Public API ───────────────────────────────────────────────────────

    def get_or_restore(self, session_id: str) -> KernelState:
        """Return live kernel or restore from notebook.

        This is the primary entry point for the agent runner.  It guarantees
        that a kernel is warm and ready for tool execution.
        """
        from backend.tools.sandbox_manager import SandboxManager

        manager = SandboxManager.get_instance()
        sandbox = manager.get(session_id)

        if sandbox is not None:
            state = self._kernel_states.get(session_id)
            if state is not None:
                state.last_execution = time.monotonic()
                return state
            # Sandbox exists but no KernelState — wrap it.
            state = KernelState(session_id=session_id)
            self._kernel_states[session_id] = state
            return state

        # No live kernel — restore from notebook.
        return self._restore(session_id)

    def has_kernel(self, session_id: str) -> bool:
        from backend.tools.sandbox_manager import SandboxManager

        return SandboxManager.get_instance().get(session_id) is not None

    def evict(self, session_id: str) -> None:
        """Manually evict a kernel.  Notebook is already persisted."""
        from backend.tools.sandbox_manager import SandboxManager

        SandboxManager.get_instance().remove(session_id)
        self._kernel_states.pop(session_id, None)
        logger.info("Kernel evicted: %s", session_id)

    def cleanup_expired(self) -> int:
        """Evict kernels that have been idle longer than TTL."""
        from backend.tools.sandbox_manager import SandboxManager

        now = time.monotonic()
        expired: list[str] = []

        for sid, state in list(self._kernel_states.items()):
            if (now - state.last_execution) > self._ttl_sec:
                expired.append(sid)

        manager = SandboxManager.get_instance()
        for sid in expired:
            manager.remove(sid)
            self._kernel_states.pop(sid, None)
            logger.info("Kernel expired (TTL): %s", sid)

        return len(expired)

    def kernel_state(self, session_id: str) -> KernelState | None:
        return self._kernel_states.get(session_id)

    # ── Restore logic ────────────────────────────────────────────────────

    def _restore(self, session_id: str) -> KernelState:
        """Create a fresh kernel and replay notebook cells."""
        from backend.tools.sandbox import SessionSandbox
        from backend.tools.sandbox_manager import SandboxManager

        logger.info("Restoring kernel for session %s", session_id)

        manifest = self._manifest_store.load(session_id)
        notebook = self._notebook_store.load(session_id)

        sandbox = SandboxManager.get_instance().get_or_create(session_id)
        session_dir = self._storage_dir / "sessions" / session_id
        sandbox.ensure_storage_dir(session_dir)

        errors: list[str] = []

        # Step 1: Pre-bind source DataFrames from persisted files.
        self._bind_sources(sandbox, manifest, session_dir, errors)

        # Step 2: Replay ALL code cells (including source_binding for
        # anything not covered by _bind_sources, plus analysis cells).
        cell_count = self._replay_cells(sandbox, notebook, errors)

        state = KernelState(
            session_id=session_id,
            cell_execution_count=cell_count,
            restored=True,
            restore_errors=errors,
        )
        self._kernel_states[session_id] = state

        if errors:
            logger.warning(
                "Kernel restored with %d error(s) for %s: %s",
                len(errors), session_id, errors[:3],
            )
        else:
            logger.info(
                "Kernel restored successfully for %s (%d cells replayed)",
                session_id, cell_count,
            )

        return state

    def _bind_sources(
        self,
        sandbox: Any,
        manifest: SessionManifest,
        session_dir: Path,
        errors: list[str],
    ) -> None:
        """Load each source into the kernel scope."""
        for source in manifest.sources:
            try:
                if source.source_type == "csv" and source.parquet_path:
                    parquet = Path(source.parquet_path)
                    if not parquet.is_absolute():
                        parquet = session_dir / parquet
                    if parquet.is_file():
                        try:
                            df = pd.read_parquet(parquet)
                            sandbox.put(source.variable_name, df)
                            logger.debug(
                                "Source %s loaded: %s (%d rows)",
                                source.alias, source.variable_name, len(df),
                            )
                        except Exception:
                            # parquet engine unavailable or file not readable
                            # as parquet — source_binding cell will handle
                            # loading during replay.
                            logger.debug(
                                "Source %s: parquet read failed, deferring to cell replay",
                                source.alias,
                            )
                    else:
                        # File may not exist yet or path may be relative to
                        # something else — source_binding cell will handle it.
                        logger.debug(
                            "Source %s: file not found at %s, deferring to cell replay",
                            source.alias, parquet,
                        )
                elif source.source_type == "db_connection" and source.connection_id:
                    # DB connections are resolved lazily by tools — just mark as available.
                    sandbox.put(source.variable_name, {"connection_id": source.connection_id})
                    logger.debug("Source %s registered: %s", source.alias, source.variable_name)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Source {source.alias}: {exc}")

    def _replay_cells(
        self,
        sandbox: Any,
        notebook: NotebookDocument,
        errors: list[str],
    ) -> int:
        """Execute all code cells sequentially.

        Source_binding cells are replayed too — their load code is the
        definitive way to restore data into scope.  _bind_sources is a
        fast pre-step that may have already loaded some variables;
        source_binding cells that re-assign them are harmless (idempotent).
        """
        replayed = 0
        for cell in notebook.code_cells:
            if not cell.source.strip():
                continue
            try:
                sandbox.execute(
                    code=cell.source,
                    tool_name=cell.metadata.tool_name or "replay",
                    include_plotly="plotly" in cell.source.lower(),
                    timeout_sec=30.0,
                )
                replayed += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Cell {cell.id}: {exc}")
                # Continue — dependent cells may fail gracefully.
        return replayed
