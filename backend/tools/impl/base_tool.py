import ast
import copy
import hashlib
import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from backend.artifacts.artifact_meta import extract_artifact_hints
from backend.core.redaction import sanitize_error_text

if TYPE_CHECKING:
    from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
    from backend.tools.sandbox import SessionSandbox

from backend.tools.sandbox import normalize_code


class ToolResultEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0")
    artifact_type: str
    items: dict[str, object]


class BaseExecTool(BaseTool):
    """
    Базовый инструмент для анализа данных.
    Выполнение делегируется SessionSandbox — единому namespace на сессию.
    """

    name: str = "base_tool"
    artifact_name: str = "base"
    human_name: str = "артефактов"
    description: str = ""
    allowed_libs: set[str] = {"pandas", "numpy"}
    allowed_artifact_types: tuple = ()
    forbidden_code_patterns: tuple[tuple[str, str], ...] = (
        (
            r"\.plot\s*\(",
            "Использование pandas.plot запрещено. Для графиков используй plotly_tool.",
        ),
        (
            r"\bmatplotlib\b|\bplt\.",
            "Использование matplotlib запрещено. Для графиков используй plotly_tool.",
        ),
        (
            r"\bopen\s*\(|\beval\s*\(|\bexec\s*\(|\bcompile\s*\(",
            "Опасные вызовы (open/eval/exec/compile) запрещены.",
        ),
        (
            r"\b__import__\b|\bglobals\s*\(|\blocals\s*\(",
            "Доступ к системному окружению Python запрещен.",
        ),
        (
            r"\bos\b|\bsys\b|\bpathlib\b|\bsubprocess\b|\bshutil\b",
            "Системные библиотеки недоступны в инструменте.",
        ),
        (
            r"\bpd\.read_csv\b|\bpd\.read_excel\b|\bpd\.read_parquet\b"
            r"|\bpd\.read_json\b|\bpd\.read_table\b|\bpd\.read_feather\b"
            r"|\bpandas\.read_csv\b|\bpandas\.read_excel\b",
            "Загрузка файлов запрещена. DataFrame `df` уже доступен в области видимости — "
            "используй его напрямую без pd.read_csv/read_excel.",
        ),
    )
    response_format: str = "content_and_artifact"
    execution_timeout_sec: float = 25.0
    tool_result_schema_version: str = "1.0"
    artifact_name_max_len: int = 48
    tool_cache_size: int = 48
    _df: pd.DataFrame = PrivateAttr()
    _include_plotly: bool = PrivateAttr(default=False)
    _tool_cache: OrderedDict[str, tuple[str, dict[str, object]]] = PrivateAttr(
        default_factory=OrderedDict
    )
    _dataset_signature: str = PrivateAttr(default="")
    _db_runtime_config: "RuntimeDBConnectionConfig | None" = PrivateAttr(default=None)
    _sandbox: "SessionSandbox | None" = PrivateAttr(default=None)

    def __init__(
        self,
        df: pd.DataFrame,
        execution_timeout_sec: float = 25.0,
        include_plotly: bool = False,
        tool_cache_size: int = 48,
        db_runtime_config: "RuntimeDBConnectionConfig | None" = None,
        sandbox: "SessionSandbox | None" = None,
    ) -> None:
        super().__init__()
        self._df = df
        self._include_plotly = include_plotly
        self.execution_timeout_sec = execution_timeout_sec
        self.tool_cache_size = max(0, int(tool_cache_size))
        self._dataset_signature = self._build_dataset_signature(df)
        self._db_runtime_config = db_runtime_config
        self._sandbox = sandbox

    @staticmethod
    def _build_dataset_signature(df: pd.DataFrame) -> str:
        head = df.head(8).to_csv(index=False)
        tail = df.iloc[-8:].to_csv(index=False)
        columns = ",".join(str(col) for col in df.columns[:64])
        payload = f"{df.shape}|{columns}|{head}|{tail}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _cache_key(self, code: str) -> str:
        sandbox_state = str(self._sandbox.execution_count) if self._sandbox else ""
        payload = (
            f"{self.name}|{self._dataset_signature}|{self.execution_timeout_sec}|{sandbox_state}|{code}"
        )
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

    def syntax_error(
        self, code: str, error: SyntaxError
    ) -> tuple[str, dict[str, object]]:
        artifact_name = getattr(self, "artifact_name", "base")
        human_name = getattr(self, "human_name", "артефактов")
        code_lines = code.splitlines()
        error_line = (
            code_lines[error.lineno - 1]
            if error.lineno and error.lineno <= len(code_lines)
            else ""
        )
        pointer = (
            " " * (error.offset - 1) + "^" if error.offset and error.offset > 0 else ""
        )
        text = (
            f"❌ Ошибка при создании {human_name}:\n"
            f"{error_line}\n"
            f"{pointer}\n"
            f"SyntaxError: {error.msg}"
        )
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
        for pattern, message in self.forbidden_code_patterns:
            if re.search(pattern, code, flags=re.IGNORECASE):
                return False, message
        return True, ""

    def validate_tool_result(self, tool_result: dict[str, object]) -> tuple[bool, str]:
        invalid_keys = []
        for name, data in tool_result.items():
            if not isinstance(data, self.allowed_artifact_types):
                invalid_keys.append(name)
        if invalid_keys:
            used_types_str = ", ".join(
                [type(tool_result[key]).__name__ for key in invalid_keys]
            )
            allowed_types_str = ", ".join(
                [t.__name__ for t in self.allowed_artifact_types]
            )
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

    def _validate_tool_contract(
        self, tool_result: object
    ) -> tuple[dict[str, object] | None, str]:
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
                candidate_items = raw_result.get(self.artifact_name)
                if isinstance(candidate_items, dict):
                    raw_items = candidate_items
            elif reserved_keys.intersection(raw_result.keys()):
                return (
                    None,
                    "Нарушен контракт `tool_result`: поле `items` отсутствует или имеет неверный тип.",
                )
            else:
                raw_items = raw_result

        if not isinstance(raw_items, dict):
            return None, "Поле `items` в `tool_result` должно быть объектом JSON (dict)."

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
        }
        normalized_artifact_type = str(raw_artifact_type or self.artifact_name).strip().lower()
        normalized_artifact_type = artifact_aliases.get(
            normalized_artifact_type, normalized_artifact_type
        )

        normalized_schema_version = str(
            raw_schema_version or self.tool_result_schema_version
        ).strip()
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
                "Нарушен контракт `tool_result` JSON schema: "
                f"{exc.errors(include_input=False)}",
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

    @staticmethod
    def _extract_payload_hints(tool_result: object) -> dict[str, Any]:
        return extract_artifact_hints(tool_result)

    def _execute_in_sandbox(self, code: str) -> object:
        """Delegate execution to the session sandbox."""
        if self._sandbox is None:
            raise RuntimeError("SessionSandbox не инициализирован для инструмента")

        extra_scope = self.get_execution_scope()
        return self._sandbox.execute(
            code=code,
            tool_name=self.name,
            include_plotly=self._include_plotly,
            timeout_sec=self.execution_timeout_sec,
            extra_scope=extra_scope or None,
        )

    def _run(self, code: str) -> tuple[str, dict[str, object]]:
        code = normalize_code(code)
        if not code:
            text = f"❌ Ошибка при создании {self.human_name}: пустой код инструмента"
            return text, {self.artifact_name: None, "text": text}

        cache_key = self._cache_key(code)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        valid, validate_message = self.validate_libraries(code)
        if not valid:
            text = f"❌ Ошибка при создании {self.human_name}: {validate_message}"
            return text, {self.artifact_name: None, "text": text}

        valid, validate_message = self.validate_code_patterns(code)
        if not valid:
            text = f"❌ Ошибка при создании {self.human_name}: {validate_message}"
            return text, {self.artifact_name: None, "text": text}

        try:
            tool_result = self._execute_in_sandbox(code)
            artifact_hints = self._extract_payload_hints(tool_result)

            if tool_result is None:
                text = (
                    f"❌ Ошибка при создании {self.human_name}: "
                    "Не найдена переменная `tool_result`"
                )
                return text, {self.artifact_name: None, "text": text}

            normalized_result, contract_message = self._validate_tool_contract(
                tool_result
            )
            if normalized_result is None:
                text = (
                    f"❌ Ошибка при создании {self.human_name}: "
                    f"{contract_message}"
                )
                return text, {self.artifact_name: None, "text": text}

            normalized_result = self.post_process_tool_result(normalized_result)
            if not isinstance(normalized_result, dict) or not normalized_result:
                text = (
                    f"❌ Ошибка валидации результатов {self.human_name}: "
                    "post_process_tool_result вернул пустой или неверный результат."
                )
                return text, {self.artifact_name: None, "text": text}

            valid, validate_message = self.validate_tool_result(normalized_result)
            if not valid:
                text = (
                    f"❌ Ошибка валидации результатов {self.human_name}: "
                    f"{validate_message}"
                )
                return text, {self.artifact_name: None, "text": text}

            text = (
                f"✅ Создано через {self.name} - {len(normalized_result)} {self.human_name}: "
                f"{', '.join(normalized_result.keys())}"
            )
            payload = {
                "text": text,
                "code": code,
            }
            if artifact_hints:
                payload.update(artifact_hints)
            payload[self.artifact_name] = normalized_result
            result = (text, payload)
            self._cache_set(cache_key, result)
            return result
        except SyntaxError as e:
            return self.syntax_error(code, e)
        except Exception as e:
            return self.other_error(e)
