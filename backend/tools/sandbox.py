"""Session-scoped persistent Python sandbox.

One ``SessionSandbox`` lives for the entire chat session. Successful executions
commit their user variables back into the shared namespace; timed-out worker
processes are terminated without mutating that state.
"""

from __future__ import annotations

import ast
import builtins
import copy
import keyword
import logging
import marshal
import multiprocessing
import pickle
import symtable
import threading
import types
from dataclasses import dataclass, field
from datetime import UTC, datetime
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.tools.schema_registry import DataFrameSchemaEntry, DataFrameSchemaRegistry

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Safe builtins (same set as before, centralised here)
# ------------------------------------------------------------------
SAFE_BUILTINS: dict[str, Any] = {
    "abs": builtins.abs,
    "all": builtins.all,
    "any": builtins.any,
    "bool": builtins.bool,
    "dict": builtins.dict,
    "enumerate": builtins.enumerate,
    "filter": builtins.filter,
    "float": builtins.float,
    "int": builtins.int,
    "len": builtins.len,
    "list": builtins.list,
    "map": builtins.map,
    "max": builtins.max,
    "min": builtins.min,
    "pow": builtins.pow,
    "print": builtins.print,
    "range": builtins.range,
    "reversed": builtins.reversed,
    "round": builtins.round,
    "set": builtins.set,
    "sorted": builtins.sorted,
    "str": builtins.str,
    "sum": builtins.sum,
    "tuple": builtins.tuple,
    "zip": builtins.zip,
    "type": builtins.type,
    "isinstance": builtins.isinstance,
    "hasattr": builtins.hasattr,
    "getattr": builtins.getattr,
    "Exception": builtins.Exception,
    "ValueError": builtins.ValueError,
    "TypeError": builtins.TypeError,
}

# Union of all libs any tool may need.
_ALL_ALLOWED_LIBS: frozenset[str] = frozenset(
    {
        "pandas",
        "numpy",
        "plotly",
        "json",
        "re",
        "math",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "statistics",
        "time",
        "_strptime",
    }
)

# Result-extraction priority (same logic as the old _execute_tool_code).
_RESULT_CANDIDATES: tuple[str, ...] = (
    "tool_result",
    "result",
    "output",
    "final_result",
    "artifact",
    "artifacts",
    "value",
    "values",
    "table",
    "plot",
    "fig",
    "payload",
    "data",
    "__tool_last_expr__",
)

# Keys that belong to the sandbox infrastructure, not user variables.
_INFRA_KEYS: frozenset[str] = frozenset(
    {
        "df",
        "db_connection",
        "db_runtime",
        "pd",
        "np",
        "px",
        "go",
        "__builtins__",
        "__tool_last_expr__",
    }
)


def is_user_variable_name(name: str) -> bool:
    """Return whether *name* can be referenced safely in sandbox code."""
    return bool(
        name
        and name.isidentifier()
        and not keyword.iskeyword(name)
        and not name.startswith("_")
        and name not in _INFRA_KEYS
    )


def _make_safe_import(allowed: frozenset[str]):
    def _safe_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in allowed:
            raise ImportError(f"Импорт библиотеки '{root}' запрещен")
        return builtins.__import__(name, globals_, locals_, fromlist, level)

    return _safe_import


def _worker_scope(
    values: dict[str, Any],
    module_names: dict[str, str],
    *,
    include_plotly: bool,
) -> dict[str, Any]:
    safe = dict(SAFE_BUILTINS)
    safe["__import__"] = _make_safe_import(_ALL_ALLOWED_LIBS)
    scope: dict[str, Any] = {
        "__builtins__": safe,
        "pd": pd,
        "np": np,
        "df": pd.DataFrame(),
        **values,
    }
    for name, module_name in module_names.items():
        scope[name] = builtins.__import__(module_name)
    if include_plotly:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        from backend.tools.plotly_express_compat import wrap_plotly_express

        scope.update({"px": wrap_plotly_express(px), "go": go, "make_subplots": make_subplots})
    return scope


def _picklable_scope(
    scope: dict[str, Any],
    names: set[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    values: dict[str, Any] = {}
    modules: dict[str, str] = {}
    for name in names:
        if name not in scope or name == "__builtins__":
            continue
        value = scope[name]
        if isinstance(value, types.ModuleType):
            modules[name] = value.__name__
            continue
        try:
            pickle.dumps(value)
        except Exception:
            continue
        values[name] = value
    return values, modules


def _extract_worker_result(
    scope: dict[str, Any],
    assigned_names: set[str],
    stdout_text: str,
) -> Any:
    for candidate in _RESULT_CANDIDATES:
        if candidate in assigned_names and candidate in scope:
            value = scope[candidate]
            if candidate != "__tool_last_expr__" or value is not None:
                return value
    for key, value in scope.items():
        key_lower = str(key).strip().lower()
        if key in assigned_names and (key_lower.endswith("_result") or key_lower == "result"):
            return value
    return stdout_text.strip() or None


def _execute_worker(
    connection: Any,
    compiled_payload: bytes,
    values: dict[str, Any],
    module_names: dict[str, str],
    assigned_names: set[str],
    persistent_names: set[str],
    include_plotly: bool,
) -> None:
    stdout_parts: list[str] = []
    try:
        scope = _worker_scope(values, module_names, include_plotly=include_plotly)
        original_print = scope["__builtins__"]["print"]

        def _capture_print(*args, sep=" ", end="\n", file=None, flush=False):
            if file is not None:
                original_print(*args, sep=sep, end=end, file=file, flush=flush)
                return
            stdout_parts.append(sep.join(str(arg) for arg in args) + end)

        scope["__builtins__"]["print"] = _capture_print
        exec(marshal.loads(compiled_payload), scope)  # pylint: disable=exec-used
        stdout_text = "".join(stdout_parts)
        result = _extract_worker_result(scope, assigned_names, stdout_text)
        updated_values, updated_modules = _picklable_scope(scope, persistent_names)
        connection.send(("ok", result, stdout_text, updated_values, updated_modules))
    except Exception as exc:
        stdout_text = "".join(stdout_parts)
        try:
            connection.send(("error", exc, stdout_text, {}, {}))
        except Exception:
            connection.send(("error", RuntimeError(str(exc)), stdout_text, {}, {}))
    finally:
        connection.close()


# ------------------------------------------------------------------
# Notebook entries
# ------------------------------------------------------------------
@dataclass
class NotebookEntry:
    timestamp: str
    entry_type: str  # "code" | "data_source_change"
    tool_name: str = ""
    language: str = "python"  # "python" | "sql"
    question: str = ""  # natural-language question (sql_tool only)
    code: str = ""
    result_summary: str = ""
    variables_created: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entry_type": self.entry_type,
            "tool_name": self.tool_name,
            "language": self.language,
            "question": self.question,
            "code": self.code,
            "result_summary": self.result_summary,
            "variables_created": self.variables_created,
            "timestamp": self.timestamp[:19].replace("T", " "),
        }


# ------------------------------------------------------------------
# SessionSandbox
# ------------------------------------------------------------------
class SessionSandbox:
    """Persistent Python execution environment for a single chat session.

    The ``_scope`` dict is the shared namespace committed between calls.
    """

    def __init__(self) -> None:
        self._scope: dict[str, Any] = {}
        self._globals: dict[str, Any] = {}
        self._notebook: list[NotebookEntry] = []
        self._lock = threading.Lock()
        self._total_executions: int = 0
        self._storage_dir: Path | None = None
        self._bound_df: pd.DataFrame | None = None
        self._bound_db_config: Any = None
        self._bound_source_label: str = ""
        self.schema_registry = DataFrameSchemaRegistry()
        self._persisted_entry_count: int = 0
        self._init_scope()

    # ------ bootstrap ------------------------------------------------

    def _init_scope(self) -> None:
        """Populate scope with standard libs and safe builtins.

        We use a single dict for both globals and locals in exec() so that
        list/dict/set comprehensions and nested functions can resolve names like
        ``df`` via LOAD_GLOBAL — they cannot see a separate locals dict.
        """
        import numpy as _np  # pylint: disable=reimported
        import pandas as _pd  # pylint: disable=reimported

        safe = dict(SAFE_BUILTINS)
        safe["__import__"] = _make_safe_import(_ALL_ALLOWED_LIBS)

        df_entry = self._bound_df if self._bound_df is not None else _pd.DataFrame()
        self._scope.update({"__builtins__": safe, "pd": _pd, "np": _np, "df": df_entry})

        # _globals is kept as an alias so callers that reference it still work.
        self._globals = self._scope

    # ------ data binding ---------------------------------------------

    def bind_dataframe(
        self,
        df: pd.DataFrame,
        source_label: str = "",
        db_runtime_config: Any = None,
    ) -> None:
        """Inject / replace ``df`` in scope and log the change."""
        with self._lock:
            is_first_load = self._bound_df is None
            old_shape = self._bound_df.shape if self._bound_df is not None else None
            normalized_source_label = str(source_label or "").strip()
            source_changed = bool(
                self._bound_source_label
                and normalized_source_label
                and self._bound_source_label != normalized_source_label
            )

            self._bound_df = df
            self._bound_db_config = db_runtime_config
            if source_changed:
                self._scope = {}
                self._globals = self._scope
                self._total_executions = 0
                self.schema_registry = DataFrameSchemaRegistry()
                self._init_scope()
            else:
                self._scope["df"] = df
            if normalized_source_label:
                self._bound_source_label = normalized_source_label
            self.schema_registry.register_dataframe(
                variable_name="df",
                df=df,
                source_kind="session_dataframe",
                source_name=source_label or "session_dataframe",
            )

            new_shape = df.shape
            if not is_first_load and old_shape != new_shape:
                self._notebook.append(
                    NotebookEntry(
                        timestamp=_now_iso(),
                        entry_type="data_source_change",
                        result_summary=(
                            f"Источник данных изменён: {source_label or 'новый датасет'}. "
                            f"Было {old_shape[0]}x{old_shape[1]}, "
                            f"стало {new_shape[0]}x{new_shape[1]}. "
                            f"Столбцы: {list(df.columns[:20])}"
                        ),
                    )
                )
            elif is_first_load:
                self._notebook.append(
                    NotebookEntry(
                        timestamp=_now_iso(),
                        entry_type="data_source_change",
                        result_summary=(
                            f"Загружен датасет: {source_label or 'датасет'}. "
                            f"Размер {new_shape[0]}x{new_shape[1]}. "
                            f"Столбцы: {list(df.columns[:20])}"
                        ),
                    )
                )

        self._persist_notebook()

    def log_code_entry(
        self,
        *,
        tool_name: str,
        code: str,
        result_summary: str,
        language: str = "python",
        question: str = "",
    ) -> None:
        """Append a code entry to the notebook from an external tool (e.g. SQLTool)."""
        with self._lock:
            self._notebook.append(
                NotebookEntry(
                    timestamp=_now_iso(),
                    entry_type="code",
                    tool_name=tool_name,
                    language=language,
                    question=question[:300],
                    code=code[:500],
                    result_summary=result_summary[:200],
                )
            )
        self._persist_notebook()

    # ------ execution ------------------------------------------------

    @staticmethod
    def _last_assigned_name(node: ast.stmt) -> str | None:
        target: ast.expr | None = None
        if isinstance(node, ast.Assign) and node.targets:
            target = node.targets[-1]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.AugAssign):
            target = node.target

        if not isinstance(target, ast.Name):
            return None
        name = target.id
        if name.startswith("_") or name in _INFRA_KEYS:
            return None
        return name

    def execute(
        self,
        code: str,
        tool_name: str = "",
        include_plotly: bool = False,
        timeout_sec: float = 25.0,
        extra_scope: dict[str, Any] | None = None,
        isolated: bool = False,
        return_stdout: bool = False,
    ) -> Any:
        """Execute *code* in the persistent scope.

        Returns the tool result extracted from the scope, optionally with
        captured stdout.
        Raises ``TimeoutError`` or ``RuntimeError`` on failure.
        """
        code = normalize_code(code)
        if not code:
            raise RuntimeError("Пустой код")

        execution_scope = (
            {name: self._copy_for_execution(name, value) for name, value in self._scope.items()}
            if isolated
            else self._scope
        )

        # Ensure plotly is available when needed.
        if include_plotly and "px" not in execution_scope:
            import plotly.express as _px
            import plotly.graph_objects as _go
            from plotly.subplots import make_subplots as _make_subplots

            from backend.tools.plotly_express_compat import wrap_plotly_express

            execution_scope.update(
                {
                    "px": wrap_plotly_express(_px),
                    "go": _go,
                    "make_subplots": _make_subplots,
                }
            )

        # Merge extra_scope (e.g. tool-specific helpers).
        if extra_scope:
            execution_scope.update(
                {name: self._copy_for_execution(name, value) for name, value in extra_scope.items()}
            )

        # Compile AST — capture last expression as __tool_last_expr__.
        tree = ast.parse(code, filename="<tool_code>", mode="exec")
        captures_last_expression = False
        if tree.body:
            if isinstance(tree.body[-1], ast.Expr):
                tree.body[-1] = ast.Assign(
                    targets=[ast.Name(id="__tool_last_expr__", ctx=ast.Store())],
                    value=tree.body[-1].value,
                )
                ast.fix_missing_locations(tree)
                captures_last_expression = True
            else:
                last_name = self._last_assigned_name(tree.body[-1])
                if last_name:
                    tree.body.append(
                        ast.Assign(
                            targets=[ast.Name(id="__tool_last_expr__", ctx=ast.Store())],
                            value=ast.Dict(
                                keys=[ast.Constant(value=last_name)],
                                values=[ast.Name(id=last_name, ctx=ast.Load())],
                            ),
                        )
                    )
                    ast.fix_missing_locations(tree)
                    captures_last_expression = True
        assigned_names = {
            symbol.get_name()
            for symbol in symtable.symtable(code, "<tool_code>", "exec").get_symbols()
            if symbol.is_assigned() or symbol.is_imported()
        }
        if captures_last_expression:
            assigned_names.add("__tool_last_expr__")
        compiled = compile(tree, filename="<tool_code>", mode="exec")

        with self._lock:
            input_values, input_modules = _picklable_scope(
                execution_scope,
                set(execution_scope),
            )
            persistent_names = set() if isolated else set(self._scope) | assigned_names
            context_name = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
            context = multiprocessing.get_context(context_name)
            parent_connection, child_connection = context.Pipe(duplex=False)
            process = context.Process(
                target=_execute_worker,
                args=(
                    child_connection,
                    marshal.dumps(compiled),
                    input_values,
                    input_modules,
                    assigned_names,
                    persistent_names,
                    include_plotly,
                ),
            )
            process.start()
            child_connection.close()
            ready = wait(
                [parent_connection, process.sentinel],
                timeout=max(0.0, float(timeout_sec)),
            )
            if parent_connection not in ready:
                if process.sentinel in ready:
                    process.join()
                    parent_connection.close()
                    raise RuntimeError(
                        f"Sandbox worker завершился без результата (exit code {process.exitcode})"
                    )
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join()
                parent_connection.close()
                raise TimeoutError(f"Превышен лимит выполнения ({timeout_sec} сек)")

            status, payload, stdout_text, updated_values, updated_modules = parent_connection.recv()
            parent_connection.close()
            process.join()
            if status == "error":
                error = payload
                try:
                    error.sandbox_stdout = stdout_text.strip()
                except Exception:
                    pass
                raise error
            result = payload
            if not isolated:
                for name, value in updated_values.items():
                    self._scope[name] = value
                for name, module_name in updated_modules.items():
                    self._scope[name] = builtins.__import__(module_name)
            self._total_executions += 1

            # Log to notebook.
            new_vars = [] if isolated else self._detect_new_user_vars()
            if not isolated:
                self._register_dataframe_vars(new_vars, source_kind="pandas_result")
            self._notebook.append(
                NotebookEntry(
                    timestamp=_now_iso(),
                    entry_type="code",
                    tool_name=tool_name,
                    code=code[:500],
                    result_summary=_describe_value(result)[:200],
                    variables_created=new_vars,
                )
            )

        self._persist_notebook()
        return (result, stdout_text.strip()) if return_stdout else result

    @staticmethod
    def _copy_for_execution(name: str, value: Any) -> Any:
        if name == "__builtins__" and isinstance(value, dict):
            return dict(value)
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return value.copy(deep=True)
        if name in _INFRA_KEYS:
            return value
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    def _detect_new_user_vars(self) -> list[str]:
        """Return names of user-created variables (non-infra, non-dunder)."""
        return [
            k
            for k in self._scope
            if k not in _INFRA_KEYS
            and not k.startswith("_")
            and not callable(self._scope[k])
            and not isinstance(self._scope[k], type)
        ]

    def _register_dataframe_vars(self, names: list[str], *, source_kind: str) -> None:
        for name in names:
            value = self._scope.get(name)
            if not isinstance(value, pd.DataFrame):
                continue
            if name == "df":
                continue
            self.schema_registry.register_dataframe(
                variable_name=name,
                df=value,
                source_kind=source_kind,  # type: ignore[arg-type]
                source_name=name,
            )

    # ------ prompt context -------------------------------------------

    def describe_for_prompt(self) -> str:
        """Markdown summary of scope variables + recent notebook entries.

        Injected into the LLM system prompt so it knows what's available.
        """
        with self._lock:
            # Build descriptions under lock without copying full scope dict.
            var_descriptions: list[tuple[str, str]] = sorted(
                (name, _describe_value(val))
                for name, val in self._scope.items()
                if name not in _INFRA_KEYS
                and not name.startswith("_")
                and not callable(val)
                and not isinstance(val, type)
            )
            recent = list(self._notebook[-8:])

        parts: list[str] = []

        # ── Variables ──
        var_lines: list[str] = []
        for name, description in var_descriptions:
            var_lines.append(f"  - `{name}`: {description}")

        if var_lines:
            parts.append(
                "Доступные переменные в sandbox (из предыдущих tool-вызовов):\n"
                + "\n".join(var_lines[:30])
                + "\nИспользуй их напрямую вместо повторного вычисления."
            )

        # ── Notebook (last N) ──
        if recent:
            nb_lines: list[str] = []
            for entry in recent:
                if entry.entry_type == "data_source_change":
                    nb_lines.append(f"  📊 {entry.result_summary}")
                else:
                    nb_lines.append(f"  🔧 [{entry.tool_name}] → {entry.result_summary}")
            parts.append("Лог сессии (последние действия):\n" + "\n".join(nb_lines))

        return "\n\n".join(parts)

    # ------ persistence -----------------------------------------------

    def set_storage_dir(self, path: Path) -> None:
        """Set the directory where notebook.md will be persisted."""
        self._storage_dir = path

    def ensure_storage_dir(self, path: Path) -> None:
        """Set storage directory if not already configured (idempotent)."""
        if self._storage_dir is None:
            self.set_storage_dir(path)

    def get_notebook_cells(self) -> list[dict]:
        """Return notebook entries as a list of dicts for the JSON API."""
        with self._lock:
            entries = list(self._notebook)
        return [{"index": i + 1, **entry.to_dict()} for i, entry in enumerate(entries)]

    def render_notebook_md(self) -> str:
        """Render the notebook as a human-readable Markdown string."""
        if not self._notebook:
            return ""
        lines: list[str] = ["# Notebook сессии", ""]
        for entry in self._notebook:
            ts = entry.timestamp[:19].replace("T", " ")
            if entry.entry_type == "data_source_change":
                lines.append(f"### {ts} — Источник данных")
                lines.append(entry.result_summary)
            else:
                lines.append(f"### {ts} — {entry.tool_name or 'code'}")
                if entry.question:
                    lines.append(f"*Q: {entry.question}*")
                if entry.code:
                    lang = entry.language or "python"
                    lines.append(f"```{lang}")
                    lines.append(entry.code)
                    lines.append("```")
                if entry.result_summary:
                    lines.append(f"**Результат:** {entry.result_summary}")
                if entry.variables_created:
                    lines.append(f"**Переменные:** {', '.join(entry.variables_created)}")
            lines.append("")
        return "\n".join(lines)

    def _persist_notebook(self) -> None:
        """Save notebook.md to storage directory if configured. No-op if nothing changed."""
        if self._storage_dir is None:
            return
        current_count = len(self._notebook)
        if current_count <= self._persisted_entry_count:
            return
        try:
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            md = self.render_notebook_md()
            (self._storage_dir / "notebook.md").write_text(md, encoding="utf-8")
            self._persisted_entry_count = current_count
        except Exception:
            logger.warning("Failed to persist notebook.md", exc_info=True)

    # ------ housekeeping ---------------------------------------------

    def get_user_scope(self) -> dict[str, Any]:
        """Snapshot of user-visible sandbox variables (for code preflight)."""
        with self._lock:
            return {key: val for key, val in self._scope.items() if key not in _INFRA_KEYS}

    def put(
        self,
        name: str,
        value: Any,
        *,
        schema_entry: DataFrameSchemaEntry | None = None,
    ) -> None:
        """Inject a named variable into scope without executing code.

        Used by non-exec tools (e.g. sql_tool) to make their results
        available to subsequent tools (e.g. plotly_tool, pandas_tool).
        """
        if not is_user_variable_name(name):
            return
        if isinstance(value, pd.DataFrame):
            from backend.data_access.dataframe_utils import deduplicate_dataframe_columns

            value = deduplicate_dataframe_columns(value)
        with self._lock:
            self._scope[name] = value
            if isinstance(value, pd.DataFrame):
                if schema_entry is not None:
                    self.schema_registry.register(schema_entry)
                else:
                    self.schema_registry.register_dataframe(
                        variable_name=name,
                        df=value,
                        source_kind="unknown",
                        source_name=name,
                    )

    def clear(self) -> None:
        """Full reset — wipe scope and notebook."""
        with self._lock:
            self._scope.clear()
            self._notebook.clear()
            self._total_executions = 0
            self._persisted_entry_count = 0
            self.schema_registry = DataFrameSchemaRegistry()
            self._init_scope()

    @property
    def execution_count(self) -> int:
        return self._total_executions

    def __bool__(self) -> bool:
        return self._total_executions > 0


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def normalize_code(code: str) -> str:
    import re

    text = str(code or "").strip()
    if not text:
        return ""
    fenced = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        parts = [b.strip() for b in fenced if b.strip()]
        text = "\n\n".join(parts).strip()
    lower = text.lower()
    if lower.startswith("python\n"):
        text = text.split("\n", 1)[1].strip()
    return text.strip()


def _describe_value(val: Any) -> str:
    """Short human-readable description of a value."""
    if val is None:
        return "None"
    if isinstance(val, pd.DataFrame):
        from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

        col_parts: list[str] = []
        for col in list(val.columns)[:40]:
            name = str(col)
            series = val[col]
            if is_datetime64_any_dtype(series):
                dtype = "datetime"
            elif is_numeric_dtype(series):
                dtype = str(series.dtype)
            else:
                dtype = "string"
            col_parts.append(f"{name} ({dtype})")
        if len(val.columns) > 40:
            col_parts.append(f"... +{len(val.columns) - 40} cols")
        return f"DataFrame {val.shape[0]}×{val.shape[1]}: " + ", ".join(col_parts)
    if isinstance(val, pd.Series):
        return f"Series {val.shape}, name={val.name}"
    if isinstance(val, np.ndarray):
        return f"ndarray {val.shape}"
    if isinstance(val, dict):
        return f"dict ({len(val)} keys)"
    if isinstance(val, list | tuple):
        return f"{type(val).__name__} (len={len(val)})"
    if isinstance(val, int | float | bool | str):
        r = repr(val)
        return r[:80] if len(r) > 80 else r
    return type(val).__name__
