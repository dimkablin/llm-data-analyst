import ast
import copy
import hashlib
import keyword
import logging
import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from backend.agent.callbacks import strip_thinking
from backend.artifacts.artifact_meta import extract_artifact_hints
from backend.core.redaction import sanitize_error_text
from backend.tools.artifact_references import (
    EXECUTION_ARTIFACT_ATTR,
    QUERY_META_ATTR,
    materialize_artifact_inputs,
)
from backend.tools.observations import (
    DataFrameSchemaSummary,
    ToolExecutionObservation,
    ToolObservationStatus,
    exception_metadata,
)

if TYPE_CHECKING:
    from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
    from backend.tools.sandbox import SessionSandbox

from backend.tools.code_preflight import list_sandbox_user_var_names, preflight_sandbox_code
from backend.tools.sandbox import normalize_code

logger = logging.getLogger(__name__)


class ToolResultEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0")
    artifact_type: str
    items: dict[str, object]


class _CodeInput(BaseModel):
    code: str = Field(description="Валидный Python-код для выполнения в sandbox-окружении.")
    input_artifacts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Только для durable-артефактов из истории: Python alias -> стабильный artifact_id "
            "из контекста истории. Для текущей sandbox-переменной не передавай input_artifacts; "
            "значение справа никогда не является именем артефакта."
        ),
    )


class BaseExecTool(BaseTool):
    """
    Базовый инструмент для анализа данных.
    Выполнение делегируется SessionSandbox — единому namespace на сессию.
    """

    name: str = "base_tool"
    args_schema: type[BaseModel] = _CodeInput
    artifact_name: str = "base"
    human_name: str = "артефактов"
    description: str = ""
    allowed_libs: set[str] = {"pandas", "numpy"}
    allowed_artifact_types: tuple = ()
    sandbox_tool_names: ClassVar[frozenset[str]] = frozenset(
        {"sql_tool", "pandas_tool", "plotly_tool", "database_tool"}
    )
    nested_tool_call_message: ClassVar[str] = (
        "An executable tool call is one action over existing named sandbox artifacts. "
        "Complete this action with those artifacts; request another capability as the "
        "next top-level tool call."
    )
    matplotlib_message: ClassVar[str] = (
        "Использование matplotlib запрещено. Для графиков используй plotly_tool."
    )
    pandas_plot_message: ClassVar[str] = (
        "Использование pandas.plot запрещено. Для графиков используй plotly_tool."
    )
    response_format: str = "content_and_artifact"
    parallel_safe: ClassVar[bool] = False
    execution_timeout_sec: float = 25.0
    tool_result_schema_version: str = "1.0"
    artifact_name_max_len: int = 48
    tool_cache_size: int = 48

    _df: pd.DataFrame = PrivateAttr()
    _include_plotly: bool = PrivateAttr(default=False)
    _tool_cache: OrderedDict[str, tuple[str, dict[str, object]]] = PrivateAttr(default_factory=OrderedDict)
    _dataset_signature: str = PrivateAttr(default="")
    _db_runtime_config: "RuntimeDBConnectionConfig | None" = PrivateAttr(default=None)
    _sandbox: "SessionSandbox | None" = PrivateAttr(default=None)
    _captured_stdout: str = PrivateAttr(default="")
    _session_id: str = PrivateAttr(default="")
    _session_store: Any | None = PrivateAttr(default=None)
    _execution_store: Any | None = PrivateAttr(default=None)

    def __init__(
        self,
        df: pd.DataFrame,
        execution_timeout_sec: float = 25.0,
        include_plotly: bool = False,
        tool_cache_size: int = 48,
        db_runtime_config: "RuntimeDBConnectionConfig | None" = None,
        sandbox: "SessionSandbox | None" = None,
        session_id: str = "",
        session_store: Any | None = None,
        execution_store: Any | None = None,
    ) -> None:
        super().__init__()
        self._df = df
        self._include_plotly = include_plotly
        self.execution_timeout_sec = execution_timeout_sec
        self.tool_cache_size = max(0, int(tool_cache_size))
        self._dataset_signature = self._build_dataset_signature(df)
        self._db_runtime_config = db_runtime_config
        self._sandbox = sandbox
        self._session_id = str(session_id or "")
        self._session_store = session_store
        self._execution_store = execution_store

    @staticmethod
    def _build_dataset_signature(df: pd.DataFrame) -> str:
        head = df.head(8).to_csv(index=False)
        tail = df.iloc[-8:].to_csv(index=False)
        columns = ",".join(str(col) for col in df.columns[:64])
        payload = f"{df.shape}|{columns}|{head}|{tail}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _cache_key(self, code: str) -> str:
        sandbox_state = str(self._sandbox.execution_count) if self._sandbox else ""
        payload = f"{self.name}|{self._dataset_signature}|{self.execution_timeout_sec}|{sandbox_state}|{code}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _cache_get(self, cache_key: str) -> tuple[str, dict[str, object]] | None:
        if self.tool_cache_size <= 0:
            return None
        cached = self._tool_cache.get(cache_key)
        if cached is None:
            return None
        self._tool_cache.move_to_end(cache_key)
        return copy.deepcopy(cached)

    def _cache_set(self, cache_key: str, value: tuple[str, dict[str, object]]) -> None:
        if self.tool_cache_size <= 0:
            return
        self._tool_cache[cache_key] = copy.deepcopy(value)
        self._tool_cache.move_to_end(cache_key)
        while len(self._tool_cache) > self.tool_cache_size:
            self._tool_cache.popitem(last=False)

    def syntax_error(self, code: str, error: SyntaxError) -> tuple[str, dict[str, object]]:
        artifact_name = getattr(self, "artifact_name", "base")
        human_name = getattr(self, "human_name", "артефактов")
        code_lines = code.splitlines()
        error_line = code_lines[error.lineno - 1] if error.lineno and error.lineno <= len(code_lines) else ""
        pointer = " " * (error.offset - 1) + "^" if error.offset and error.offset > 0 else ""
        text = f"❌ Ошибка при создании {human_name}:\n{error_line}\n{pointer}\nSyntaxError: {error.msg}"
        return text, {
            artifact_name: None,
            "text": text,
        }

    def other_error(self, error: Exception) -> tuple[str, dict[str, object]]:
        artifact_name = getattr(self, "artifact_name", "base")
        message = sanitize_error_text(str(error) or error.__class__.__name__)
        text = f"❌ Ошибка при создании {self.human_name}: {message}"
        return text, {artifact_name: None, "text": text}

    @staticmethod
    def get_imports_from_code(code: str) -> set[str]:
        tree = ast.parse(code)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
        return imports

    def validate_libraries(self, code: str) -> tuple[bool, str]:
        if not self.allowed_libs:
            return True, ""
        try:
            imports = self.get_imports_from_code(code)
            forbidden = [lib for lib in imports if lib not in self.allowed_libs]
            if forbidden:
                return (
                    False,
                    f"В {self.name} разрешено использовать только библиотеки: "
                    f"{', '.join(sorted(self.allowed_libs))}. "
                    f"Обнаружены запрещенные библиотеки: {', '.join(forbidden)}",
                )
            return True, ""
        except Exception as e:
            return False, f"Ошибка при анализе импортов в {self.name}: {e}"

    def validate_code_patterns(self, code: str) -> tuple[bool, str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return True, ""

        system_modules = {"os", "pathlib", "shutil", "subprocess", "sys"}
        unsafe_names = {"__import__", "compile", "eval", "exec", "globals", "locals", "open"}
        file_readers = {
            "read_csv",
            "read_excel",
            "read_feather",
            "read_json",
            "read_parquet",
            "read_table",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
                if roots & self.sandbox_tool_names:
                    return False, self.nested_tool_call_message
                if "matplotlib" in roots:
                    return False, self.matplotlib_message
                if roots & system_modules:
                    return False, "Системные библиотеки недоступны в инструменте."
            if isinstance(node, ast.ImportFrom):
                root = str(node.module or "").split(".", 1)[0]
                if root in self.sandbox_tool_names:
                    return False, self.nested_tool_call_message
                if root == "matplotlib":
                    return False, self.matplotlib_message
                if root in system_modules:
                    return False, "Системные библиотеки недоступны в инструменте."
            if isinstance(node, ast.Name) and node.id in {"matplotlib", "plt"}:
                return False, self.matplotlib_message
            if isinstance(node, ast.Name) and node.id in unsafe_names:
                return False, "Доступ к системному окружению Python запрещен."
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.sandbox_tool_names:
                    return False, self.nested_tool_call_message
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "plot" or (
                    isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "plot"
                ):
                    return False, self.pandas_plot_message
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id in {"pandas", "pd"}
                    and node.func.attr in file_readers
                ):
                    return False, (
                        "Загрузка файлов запрещена. DataFrame `df` уже доступен в области видимости — "
                        "используй его напрямую без pd.read_csv/read_excel."
                    )
        return True, ""

    def validate_tool_result(self, tool_result: dict[str, object]) -> tuple[bool, str]:
        invalid_keys = []
        for name, data in tool_result.items():
            if not isinstance(data, self.allowed_artifact_types):
                invalid_keys.append(name)
        if invalid_keys:
            used_types_str = ", ".join([type(tool_result[key]).__name__ for key in invalid_keys])
            allowed_types_str = ", ".join([t.__name__ for t in self.allowed_artifact_types])
            return (
                False,
                f"Неверный тип для ключей: {', '.join(invalid_keys)}. "
                f"Тип данных: {used_types_str}. Разрешенные типы: {allowed_types_str}",
            )
        return True, ""

    def _normalize_item_name(self, raw_name: object, fallback_idx: int) -> str:
        name = str(raw_name).strip()
        if not name:
            name = f"{self.artifact_name}_{fallback_idx}"
        name = re.sub(r"\s+", " ", name)
        if self.artifact_name_max_len > 0 and len(name) > self.artifact_name_max_len:
            name = name[: self.artifact_name_max_len - 3].rstrip() + "..."
        return name

    def post_process_tool_result(self, tool_result: dict[str, object]) -> dict[str, object]:
        """
        Нормализует output перед валидацией/сериализацией.
        Можно переопределять в конкретных инструментах.
        """
        normalized: dict[str, object] = {}
        used: set[str] = set()
        for idx, (name, value) in enumerate(tool_result.items(), start=1):
            candidate = self._normalize_item_name(name, idx)
            if candidate in used:
                suffix = 2
                while f"{candidate}_{suffix}" in used:
                    suffix += 1
                candidate = f"{candidate}_{suffix}"
            used.add(candidate)
            normalized[candidate] = value
        return normalized

    def _validate_tool_contract(self, tool_result: object) -> tuple[dict[str, object] | None, str]:
        if not isinstance(tool_result, dict):
            if isinstance(tool_result, self.allowed_artifact_types):
                return {self.artifact_name: tool_result}, ""
            return None, "`tool_result` должен быть объектом JSON (dict)."

        raw_result = dict(tool_result)
        if (
            "tool_result" in raw_result
            and isinstance(raw_result.get("tool_result"), dict)
            and len(raw_result) == 1
        ):
            raw_result = dict(raw_result["tool_result"])

        reserved_keys = {
            "schema_version",
            "schemaVersion",
            "artifact_type",
            "artifactType",
            "type",
            "items",
        }
        raw_schema_version = raw_result.get("schema_version", raw_result.get("schemaVersion"))
        raw_artifact_type = raw_result.get(
            "artifact_type",
            raw_result.get("artifactType", raw_result.get("type")),
        )
        raw_items = raw_result.get("items")

        if raw_items is None:
            if len(raw_result) == 1 and self.artifact_name in raw_result:
                raw_items = raw_result.get(self.artifact_name)
            elif reserved_keys.intersection(raw_result.keys()):
                return (
                    None,
                    "Нарушен контракт `tool_result`: поле `items` отсутствует или имеет неверный тип.",
                )
            else:
                raw_items = raw_result

        if not isinstance(raw_items, dict):
            # Be tolerant to common LLM envelope mistakes:
            # - items returned as a single payload instead of {"name": payload}
            # - items returned as list for table/value output
            raw_items = {self.artifact_name: raw_items}

        artifact_aliases = {
            "plotly": "plot",
            "graph": "plot",
            "figure": "plot",
            "chart": "plot",
            "dataframe": "table",
            "df": "table",
            "metric": "value",
            "metrics": "value",
            "values": "value",
            "json_data": "json",
            "structured": "json",
            "search_result": "json",
        }
        normalized_artifact_type = str(raw_artifact_type or self.artifact_name).strip().lower()
        normalized_artifact_type = artifact_aliases.get(normalized_artifact_type, normalized_artifact_type)

        normalized_schema_version = str(raw_schema_version or self.tool_result_schema_version).strip()
        if normalized_schema_version == "1":
            normalized_schema_version = "1.0"

        envelope_candidate = {
            "schema_version": normalized_schema_version,
            "artifact_type": normalized_artifact_type,
            "items": raw_items,
        }

        try:
            envelope = ToolResultEnvelope.model_validate(envelope_candidate)
        except ValidationError as exc:
            return (
                None,
                f"Нарушен контракт `tool_result` JSON schema: {exc.errors(include_input=False)}",
            )

        if envelope.schema_version != self.tool_result_schema_version:
            return (
                None,
                "Неверная версия контракта `tool_result`: "
                f"{envelope.schema_version}. Ожидается {self.tool_result_schema_version}.",
            )
        if envelope.artifact_type != self.artifact_name:
            return (
                None,
                "Неверный `artifact_type` в `tool_result`: "
                f"{envelope.artifact_type}. Ожидается {self.artifact_name}.",
            )
        if not envelope.items:
            return None, "Поле `items` в `tool_result` не должно быть пустым."
        return dict(envelope.items), ""

    def get_execution_scope(self) -> dict[str, Any]:
        return {}

    def get_preflight_extra_allowed(self) -> set[str]:
        """Names that will exist at runtime but may not yet be in sandbox scope."""
        from backend.tools.code_preflight import _PLOTLY_SCOPE_NAMES

        names = set(self.get_execution_scope().keys())
        if self._include_plotly:
            names.update(_PLOTLY_SCOPE_NAMES)
        return names

    @staticmethod
    def _extract_payload_hints(tool_result: object) -> dict[str, Any]:
        return extract_artifact_hints(tool_result)

    def _publish_result_items_to_sandbox(self, items: dict[str, object]) -> None:
        """Expose successful artifact items as named sandbox variables.

        Tool orchestration relies on successful outputs being addressable by the
        artifact names mentioned in tool observations. Most pandas/sql flows get
        this for free because user code assigns intermediate variables or the SQL
        tool injects its dataframe explicitly. Helper-backed tools such as
        forecast_tool return an artifact envelope directly, so without this step
        a later pandas/plotly tool can see `forecast_result` in the observation
        but cannot actually reference it in sandbox code.
        """
        if self._sandbox is None:
            return
        for raw_name, value in items.items():
            name = str(raw_name or "").strip()
            if not name or not name.isidentifier() or keyword.iskeyword(name):
                continue
            self._sandbox.put(name, value)

    @staticmethod
    def _assigns_tool_result(code: str) -> bool:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        return any(
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "tool_result"
            for node in ast.walk(tree)
        )

    def _inferred_tool_result_note(
        self,
        *,
        code: str,
        raw_tool_result: object,
        normalized_result: dict[str, object],
    ) -> str:
        if self._assigns_tool_result(code):
            return ""
        if isinstance(raw_tool_result, str) and raw_tool_result.strip():
            source = "printed stdout" if re.search(r"\bprint\s*\(", code) else "string output"
        else:
            names = ", ".join(f"`{name}`" for name in normalized_result)
            source = f"result variable(s): {names}" if names else "last expression"
        return (
            f"# {self.name} inferred `tool_result` from {source} because code did not assign `tool_result`."
        )

    def _merge_inferred_artifact_hints(
        self,
        artifact_hints: dict[str, Any],
        *,
        code: str,
        normalized_result: dict[str, object],
    ) -> dict[str, Any]:
        """Add deterministic provenance hints that can be inferred from code/scope."""
        if self._sandbox is None:
            return artifact_hints
        source_artifact_names = self._referenced_dataframe_names(code)
        merged_hints = copy.deepcopy(artifact_hints)
        meta = dict(merged_hints.get("meta") or {})
        lineage = dict(meta.get("lineage") or {})
        existing_artifacts = self._normalize_name_list(lineage.get("source_artifact_names"))
        if source_artifact_names:
            lineage["source_artifact_names"] = self._unique_names(
                [*existing_artifacts, *source_artifact_names]
            )
            source_artifact_ids = [
                str(metadata.get("artifact_id") or "").strip()
                for name in source_artifact_names
                if isinstance(
                    metadata := self._sandbox.get_user_scope()[name].attrs.get(EXECUTION_ARTIFACT_ATTR),
                    dict,
                )
                and str(metadata.get("artifact_id") or "").strip()
            ]
            if source_artifact_ids:
                lineage["source_artifact_ids"] = self._unique_names(
                    [
                        *self._normalize_name_list(lineage.get("source_artifact_ids")),
                        *source_artifact_ids,
                    ]
                )
            meta["lineage"] = lineage
            merged_hints["meta"] = meta

        if self.artifact_name != "table":
            return merged_hints
        if not any(isinstance(value, (pd.DataFrame, pd.Series)) for value in normalized_result.values()):
            return merged_hints
        if not self._code_combines_tabular_inputs(code):
            return merged_hints

        source_table_names = source_artifact_names
        if len(source_table_names) < 2:
            return merged_hints

        existing_names = self._normalize_name_list(lineage.get("source_table_names"))
        combined_names = self._unique_names([*existing_names, *source_table_names])
        if len(combined_names) < 2:
            return merged_hints

        lineage["source_table_names"] = combined_names
        existing_tables = lineage.get("source_tables")
        if not isinstance(existing_tables, list) or not existing_tables:
            lineage["source_tables"] = [
                {"qualified_name": name, "table_name": name} for name in combined_names
            ]
        meta["lineage"] = lineage
        merged_hints["meta"] = meta
        return merged_hints

    @staticmethod
    def _normalize_name_list(raw: object) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, (list, tuple, set)):
            values = list(raw)
        else:
            return []
        return [str(item).strip() for item in values if str(item).strip()]

    @staticmethod
    def _unique_names(values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            name = str(value).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            result.append(name)
        return result

    @staticmethod
    def _code_combines_tabular_inputs(code: str) -> bool:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr in {"merge", "join"}:
                    return True
                if (
                    func.attr in {"merge", "concat"}
                    and isinstance(func.value, ast.Name)
                    and func.value.id in {"pd", "pandas"}
                ):
                    return True
            if isinstance(func, ast.Name) and func.id in {"merge", "concat"}:
                return True
        return False

    def _referenced_dataframe_names(self, code: str) -> list[str]:
        if self._sandbox is None:
            return []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        referenced_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assigned_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        scope = self._sandbox.get_user_scope()
        return [
            name
            for name in sorted(referenced_names)
            if name not in assigned_names and isinstance(scope.get(name), pd.DataFrame)
        ]

    def _referenced_truncated_dataframe_names(self, code: str) -> list[str]:
        if self._sandbox is None:
            return []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        referenced_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assigned_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        scope = self._sandbox.get_user_scope()
        return [
            name
            for name in sorted(referenced_names)
            if name not in assigned_names
            and isinstance(scope.get(name), (pd.DataFrame, pd.Series))
            and isinstance(scope[name].attrs.get(QUERY_META_ATTR), dict)
            and (
                scope[name].attrs[QUERY_META_ATTR].get("truncated") is True
                or scope[name].attrs[QUERY_META_ATTR].get("has_more_rows") is True
            )
        ]

    def _available_variable_names(self) -> list[str]:
        if self._sandbox is None:
            return []
        return list_sandbox_user_var_names(self._sandbox.get_user_scope())

    def _referenced_dataframe_schemas(self, code: str) -> list[DataFrameSchemaSummary]:
        if self._sandbox is None:
            return []
        scope = self._sandbox.get_user_scope()
        return [
            DataFrameSchemaSummary.from_value(name, scope.get(name))
            for name in self._referenced_dataframe_names(code)
        ]

    def _contract_hint(self) -> str:
        if self.artifact_name == "plot":
            return (
                "This tool must assign the final chart artifact to `tool_result`.\n"
                "Debug prints are not considered tool output.\n"
                "For plot output:\n"
                'tool_result = chart.result(fig, artifact_name="chart_name")'
            )
        return (
            "This tool must assign the final artifact to `tool_result`.\n"
            "Debug prints are not considered tool output.\n"
            "For table output:\n"
            'tool_result = {"schema_version": "1.0", "artifact_type": "table", '
            '"items": {"result": result_df}}'
        )

    def _error_result(
        self,
        *,
        input_code: str,
        error: str,
        executed_code: str | None = None,
        error_type: str | None = None,
        missing_symbol: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        clean_error = sanitize_error_text(str(error or "").strip())
        observation = ToolExecutionObservation(
            status=ToolObservationStatus.ERROR,
            tool_name=self.name,
            input_code=input_code,
            executed_code=executed_code if executed_code != input_code else None,
            error=clean_error,
            contract_hint=self._contract_hint(),
            available_variables=self._available_variable_names(),
            referenced_dataframe_schemas=self._referenced_dataframe_schemas(input_code),
        )
        text = observation.to_message_text()
        payload: dict[str, object] = {
            self.artifact_name: None,
            "text": text,
            "status": observation.status.value,
            "error": clean_error,
            "input_code": input_code,
            "contract_hint": observation.contract_hint,
            "available_variables": list(observation.available_variables),
            "referenced_dataframe_schemas": [
                schema.model_dump(mode="json") for schema in observation.referenced_dataframe_schemas
            ],
            "observation": observation.model_dump(mode="json"),
        }
        if observation.executed_code:
            payload["executed_code"] = observation.executed_code
        if error_type:
            payload["error_type"] = error_type
        if missing_symbol:
            payload["missing_symbol"] = missing_symbol
        return text, payload

    def _execute_in_sandbox(self, code: str) -> object:
        """Delegate execution to the session sandbox."""
        if self._sandbox is None:
            raise RuntimeError("SessionSandbox не инициализирован для инструмента")

        extra_scope = self.get_execution_scope()
        try:
            tool_result, stdout = self._sandbox.execute(
                code=code,
                tool_name=self.name,
                include_plotly=self._include_plotly,
                timeout_sec=self.execution_timeout_sec,
                extra_scope=extra_scope or None,
                isolated=True,
                return_stdout=True,
            )
        except Exception as exc:
            self._captured_stdout = str(getattr(exc, "sandbox_stdout", "") or "")
            raise
        self._captured_stdout = stdout
        return tool_result

    def _try_run_once(self, code: str) -> tuple[bool, str, dict[str, object]]:
        """Execute code once. Returns (success, message, payload).

        On success: success=True, message=✅ text, payload=full artifact dict.
        On failure: success=False, message=clean error string, payload={}.
        """
        try:
            truncated_inputs = self._referenced_truncated_dataframe_names(code)
            tool_result = self._execute_in_sandbox(code)
            artifact_hints = self._extract_payload_hints(tool_result)

            if tool_result is None:
                return False, "Не найдена переменная `tool_result`", {}

            normalized_result, contract_message = self._validate_tool_contract(tool_result)
            if normalized_result is None:
                return False, contract_message, {}

            normalized_result = self.post_process_tool_result(normalized_result)
            if not isinstance(normalized_result, dict) or not normalized_result:
                return False, "post_process_tool_result вернул пустой или неверный результат.", {}

            valid, validate_message = self.validate_tool_result(normalized_result)
            if not valid:
                return False, validate_message, {}

            if truncated_inputs:
                for value in normalized_result.values():
                    if not isinstance(value, (pd.DataFrame, pd.Series)):
                        continue
                    query_meta = dict(value.attrs.get(QUERY_META_ATTR) or {})
                    query_meta.update(
                        truncated=True,
                        upstream_artifacts=truncated_inputs,
                    )
                    value.attrs[QUERY_META_ATTR] = query_meta

            self._publish_result_items_to_sandbox(normalized_result)

            artifact_hints = self._merge_inferred_artifact_hints(
                artifact_hints,
                code=code,
                normalized_result=normalized_result,
            )
            if truncated_inputs:
                artifact_hints = copy.deepcopy(artifact_hints)
                meta = dict(artifact_hints.get("meta") or {})
                meta["upstream_completeness"] = {
                    "truncated": True,
                    "source_artifacts": truncated_inputs,
                }
                artifact_hints["meta"] = meta

            if truncated_inputs:
                sources = ", ".join(f"`{name}`" for name in truncated_inputs)
                text = (
                    f"TRUNCATED_RESULT: {self.name} output derives from incomplete "
                    f"input artifact(s): {sources}. It is not analysis-ready. "
                    "Return to the source tool and obtain complete final-grain data, "
                    "using complete non-overlapping partitions if the final grain exceeds "
                    "the source cap. Do not calculate, plot, or answer from this artifact."
                )
            else:
                text = (
                    f"✅ Создано через {self.name} - {len(normalized_result)} {self.human_name}: "
                    f"{', '.join(normalized_result.keys())}"
                )
            inferred_note = self._inferred_tool_result_note(
                code=code,
                raw_tool_result=tool_result,
                normalized_result=normalized_result,
            )
            tool_result_note = ""
            if isinstance(tool_result, dict):
                tool_result_note = str(tool_result.get("tool_result_note") or "").strip()
            result_notes = [note for note in (inferred_note, tool_result_note) if note]
            if result_notes:
                text = f"{text}\n" + "\n".join(result_notes)

            payload: dict[str, object] = {
                "schema_version": self.tool_result_schema_version,
                "text": text,
                "code": code,
                "artifact_type": self.artifact_name,
                "items": normalized_result,
            }
            if result_notes:
                payload["tool_result_note"] = "\n".join(result_notes)

            if artifact_hints:
                payload.update(artifact_hints)
            payload[self.artifact_name] = normalized_result
            return True, text, payload

        except SyntaxError as e:
            code_lines = code.splitlines()
            error_line = code_lines[e.lineno - 1] if e.lineno and e.lineno <= len(code_lines) else ""
            return False, f"SyntaxError: {e.msg}\n{error_line}", exception_metadata(e)
        except KeyError as e:
            missing = str(e.args[0]) if e.args else str(e)
            return False, f"KeyError: {missing}", exception_metadata(e)
        except Exception as e:
            message = str(e) or e.__class__.__name__
            logger.exception("Tool %s failed while executing code", self.name)
            return False, message, exception_metadata(e)

    def _run(
        self,
        code: str,
        input_artifacts: dict[str, str] | None = None,
    ) -> tuple[str, dict[str, object]]:
        self._captured_stdout = ""
        code = normalize_code(strip_thinking(code))
        if not code:
            return self._error_result(input_code=code, error=f"Empty code for {self.name}.")

        if input_artifacts:
            if self._sandbox is None:
                return self._error_result(
                    input_code=code,
                    error="SessionSandbox is unavailable for artifact materialization.",
                )
            try:
                materialize_artifact_inputs(
                    input_artifacts,
                    session_id=self._session_id,
                    session_store=self._session_store,
                    execution_store=self._execution_store,
                    sandbox=self._sandbox,
                )
            except ValueError as exc:
                return self._error_result(input_code=code, error=str(exc))

        cache_key = self._cache_key(code)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        valid, validate_message = self.validate_libraries(code)
        if not valid:
            return self._error_result(input_code=code, error=validate_message)

        valid, validate_message = self.validate_code_patterns(code)
        if not valid:
            return self._error_result(input_code=code, error=validate_message)

        run_code = code
        if self._sandbox is not None:
            run_code, preflight_err = preflight_sandbox_code(
                run_code,
                self._sandbox.get_user_scope(),
                extra_allowed=self.get_preflight_extra_allowed(),
            )
            if preflight_err:
                return self._error_result(
                    input_code=code,
                    executed_code=run_code,
                    error=preflight_err,
                )

        ok, msg, payload = self._try_run_once(run_code)
        if self._captured_stdout:
            msg = f"{msg}\n\nSTDOUT_FOR_LLM_CONTEXT:\n{self._captured_stdout}"
            if ok:
                payload["text"] = msg

        if ok:
            if run_code != code:
                payload["input_code"] = code
                payload["executed_code"] = run_code
            result = (msg, payload)
            self._cache_set(cache_key, result)
            return result

        return self._error_result(
            input_code=code,
            executed_code=run_code,
            error=msg,
            error_type=str(payload.get("error_type") or "") or None,
            missing_symbol=str(payload.get("missing_symbol") or "") or None,
        )
