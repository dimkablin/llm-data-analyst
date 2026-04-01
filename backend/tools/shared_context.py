"""Cross-tool shared variable context.

Variables with the ``shared_`` prefix created in one tool execution
are automatically captured and injected into subsequent tool executions
within the same ``act`` step.
"""
from __future__ import annotations

import io
import logging
import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Types safe to unpickle from tool subprocesses.
_ALLOWED_UNPICKLE_MODULES: dict[str, set[str]] = {
    "builtins": {"int", "float", "str", "bool", "bytes", "list", "dict", "tuple", "set", "frozenset", "complex", "NoneType"},
    "numpy": {"ndarray", "int64", "float64", "int32", "float32", "bool_", "dtype"},
    "numpy.core.multiarray": {"_reconstruct", "scalar"},
    "numpy._core.multiarray": {"_reconstruct", "scalar"},
    "pandas.core.frame": {"DataFrame"},
    "pandas.core.series": {"Series"},
    "pandas.core.indexes.base": {"_new_Index"},
    "pandas.core.indexes.range": {"RangeIndex"},
    "pandas.core.indexes.api": {"_new_Index"},
    "pandas.core.internals.managers": {"BlockManager"},
    "pandas.core.internals.blocks": {"new_block_2d", "new_block"},
    "pandas.core.arrays.categorical": {"Categorical"},
    "pandas.core.arrays.numpy_": {"NumpyExtensionArray"},
    "pandas.core.arrays.masked": {"BaseMaskedArray"},
    "pandas.core.indexes.frozen": {"FrozenList"},
    "pandas.core.dtypes.dtypes": {"CategoricalDtype"},
    "collections": {"OrderedDict"},
}

SHARED_PREFIX = "shared_"
MAX_TOTAL_BYTES = 200_000_000  # 200 MB
MAX_VAR_BYTES = 50_000_000  # 50 MB


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows whitelisted types."""

    def find_class(self, module: str, name: str) -> Any:
        allowed = _ALLOWED_UNPICKLE_MODULES.get(module)
        if allowed is not None and name in allowed:
            return super().find_class(module, name)
        # Allow pandas internal reconstruction helpers.
        if module.startswith("pandas.") and name.startswith("_"):
            return super().find_class(module, name)
        # Allow numpy dtypes.
        if module.startswith("numpy") and ("dtype" in name or name.startswith("_")):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Unpickle denied: {module}.{name}"
        )


def _safe_loads(data: bytes) -> Any:
    return _RestrictedUnpickler(io.BytesIO(data)).load()


@dataclass
class SharedVarMeta:
    """Metadata about a single shared variable."""
    name: str
    type_name: str
    shape: str  # e.g. "(12, 4)" for DataFrame, "" for scalar
    size_bytes: int
    producer_tool: str
    columns: list[str] = field(default_factory=list)  # column names for DataFrame/Series


class SharedContext:
    """Session-scoped variable store for cross-tool data sharing.

    Variables are stored as pickle bytes in the parent process memory.
    Before each tool execution, matching variables are deserialized
    into the tool's ``local_scope``. After execution, variables with
    the ``shared_`` prefix are captured and stored.
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}
        self._metadata: dict[str, SharedVarMeta] = {}
        self._total_bytes: int = 0

    # ── Write ────────────────────────────────────────────────────────

    def put(self, name: str, value: Any, producer_tool: str) -> bool:
        """Pickle and store *value*. Returns False if size limit exceeded."""
        try:
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            logger.warning("SharedContext: cannot pickle '%s' — skipped", name)
            return False

        if len(data) > MAX_VAR_BYTES:
            logger.warning(
                "SharedContext: '%s' too large (%d bytes, limit %d) — skipped",
                name, len(data), MAX_VAR_BYTES,
            )
            return False

        # Evict old entry if replacing.
        if name in self._store:
            self._total_bytes -= len(self._store[name])

        if self._total_bytes + len(data) > MAX_TOTAL_BYTES:
            logger.warning("SharedContext: total size limit reached — '%s' skipped", name)
            return False

        self._store[name] = data
        self._total_bytes += len(data)
        self._metadata[name] = SharedVarMeta(
            name=name,
            type_name=type(value).__name__,
            shape=self._describe_shape(value),
            size_bytes=len(data),
            producer_tool=producer_tool,
            columns=self._extract_columns(value),
        )
        return True

    # ── Read ─────────────────────────────────────────────────────────

    def get(self, name: str) -> Any:
        data = self._store.get(name)
        if data is None:
            return None
        return _safe_loads(data)

    def inject_all(self) -> dict[str, Any]:
        """Deserialize all stored variables into a dict suitable for local_scope."""
        result: dict[str, Any] = {}
        for name, data in self._store.items():
            try:
                result[name] = _safe_loads(data)
            except Exception:
                logger.warning("SharedContext: failed to unpickle '%s'", name)
        return result

    # ── Capture ──────────────────────────────────────────────────────

    def capture_from_scope(self, scope: dict[str, Any], producer_tool: str) -> list[str]:
        """Auto-capture variables with ``shared_`` prefix from scope.

        Returns list of captured variable names.
        """
        captured: list[str] = []
        for name, value in scope.items():
            if not name.startswith(SHARED_PREFIX):
                continue
            if self.put(name, value, producer_tool):
                captured.append(name)
        return captured

    def capture_from_result(
        self, shared_vars: dict[str, bytes], producer_tool: str,
    ) -> list[str]:
        """Store pre-pickled variables received from a subprocess.

        Returns list of captured variable names.
        """
        captured: list[str] = []
        for name, data in shared_vars.items():
            if not isinstance(data, bytes):
                continue
            if len(data) > MAX_VAR_BYTES:
                continue
            if name in self._store:
                self._total_bytes -= len(self._store[name])
            if self._total_bytes + len(data) > MAX_TOTAL_BYTES:
                continue
            # Validate by trying to unpickle.
            try:
                value = _safe_loads(data)
            except Exception:
                logger.warning("SharedContext: refused to accept '%s' (unpickle denied)", name)
                continue
            self._store[name] = data
            self._total_bytes += len(data)
            self._metadata[name] = SharedVarMeta(
                name=name,
                type_name=type(value).__name__,
                shape=self._describe_shape(value),
                size_bytes=len(data),
                producer_tool=producer_tool,
                columns=self._extract_columns(value),
            )
            captured.append(name)
        return captured

    # ── Describe (for LLM prompts) ───────────────────────────────────

    def describe(self) -> list[SharedVarMeta]:
        return list(self._metadata.values())

    def describe_for_prompt(self) -> str:
        """Human-readable description for injection into execution prompt."""
        if not self._metadata:
            return ""
        lines = ["Доступные общие переменные (из предыдущих tool-вызовов):"]
        for meta in self._metadata.values():
            shape_part = f", shape {meta.shape}" if meta.shape else ""
            cols_part = f", columns: {meta.columns}" if meta.columns else ""
            lines.append(
                f"- `{meta.name}`: {meta.type_name}{shape_part}{cols_part} (от {meta.producer_tool})"
            )
        lines.append("Используй их напрямую вместо повторного вычисления.")
        return "\n".join(lines)

    # ── Housekeeping ─────────────────────────────────────────────────

    def clear(self) -> None:
        self._store.clear()
        self._metadata.clear()
        self._total_bytes = 0

    def __len__(self) -> int:
        return len(self._store)

    def __bool__(self) -> bool:
        return bool(self._store)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_columns(value: Any) -> list[str]:
        """Return column names for DataFrame/Series, empty list otherwise."""
        if isinstance(value, pd.DataFrame):
            return [str(c) for c in value.columns]
        if isinstance(value, pd.Series) and value.name is not None:
            return [str(value.name)]
        return []

    @staticmethod
    def _describe_shape(value: Any) -> str:
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return str(value.shape)
        if isinstance(value, np.ndarray):
            return str(value.shape)
        if isinstance(value, (list, tuple)):
            return f"({len(value)},)"
        if isinstance(value, dict):
            return f"({len(value)} keys)"
        return ""
