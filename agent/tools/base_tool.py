import ast
import builtins
import copy
import hashlib
import multiprocessing
import queue
import re
import traceback
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from backend.artifact_meta import extract_artifact_hints
from backend.redaction import sanitize_error_text

if TYPE_CHECKING:
    from backend.db_runtime_service import RuntimeDBConnectionConfig


SAFE_BUILTINS = {
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
    "Exception": builtins.Exception,
    "ValueError": builtins.ValueError,
    "TypeError": builtins.TypeError,
}


class ToolResultEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0")
    artifact_type: str
    items: dict[str, object]


def _normalize_tool_code(code: str) -> str:
    text = str(code or "").strip()
    if not text:
        return ""

    fenced_blocks = re.findall(
        r"```(?:python|py)?\s*([\s\S]*?)```",
        text,
        flags=re.IGNORECASE,
    )
    if fenced_blocks:
        parts = [block.strip() for block in fenced_blocks if block.strip()]
        text = "\n\n".join(parts).strip()

    lower = text.lower()
    if lower.startswith("python\n"):
        text = text.split("\n", 1)[1].strip()

    return text.strip()


def _execute_tool_code(
    code: str,
    df: pd.DataFrame,
    include_plotly: bool,
    allowed_libs: tuple[str, ...],
    db_runtime_config: "RuntimeDBConnectionConfig | None" = None,
    extra_scope: dict[str, Any] | None = None,
) -> object:
    code = _normalize_tool_code(code)
    allowed = set(allowed_libs)

    def _safe_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in allowed:
            raise ImportError(f"Импорт библиотеки '{root}' запрещен в инструменте")
        return builtins.__import__(name, globals_, locals_, fromlist, level)

    safe_builtins = dict(SAFE_BUILTINS)
    safe_builtins["__import__"] = _safe_import

    local_scope: dict[str, Any] = {
        "df": df,
        "db_connection": db_runtime_config,
        "db_runtime": db_runtime_config,
    }
    if extra_scope:
        local_scope.update(extra_scope)
    import pandas as _pd
    import numpy as _np

    local_scope.update({"pd": _pd, "np": _np})
    if include_plotly:
        import plotly.express as _px
        import plotly.graph_objects as _go

        local_scope.update({"px": _px, "go": _go})

    tree = ast.parse(code, filename="<tool_code>", mode="exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id="__tool_last_expr__", ctx=ast.Store())],
            value=tree.body[-1].value,
        )
        ast.fix_missing_locations(tree)

    compiled = compile(tree, filename="<tool_code>", mode="exec")
    exec(compiled, {"__builtins__": safe_builtins}, local_scope)
    if "tool_result" in local_scope:
        return local_scope.get("tool_result")

    alias_candidates = (
        "result",
        "output",
        "final_result",
        "artifact",
        "artifacts",
        "value",
        "values",
        "table",
        "plot",
        "payload",
        "data",
        "__tool_last_expr__",
    )
    for candidate in alias_candidates:
        if candidate in local_scope:
            return local_scope.get(candidate)

    for key, value in local_scope.items():
        key_text = str(key).strip().lower()
        if key_text.endswith("_result") or key_text == "result":
            return value

    return None


def _tool_worker(
    result_queue: multiprocessing.Queue,
    code: str,
    df: pd.DataFrame,
    include_plotly: bool,
    allowed_libs: tuple[str, ...],
    db_runtime_config: "RuntimeDBConnectionConfig | None" = None,
    extra_scope: dict[str, Any] | None = None,
) -> None:
    try:
        result = _execute_tool_code(
            code,
            df,
            include_plotly,
            allowed_libs,
            db_runtime_config,
            extra_scope,
        )
        result_queue.put({"ok": True, "result": result})
    except Exception:
        result_queue.put(
            {"ok": False, "error": sanitize_error_text(traceback.format_exc())}
        )


class BaseExecTool(BaseTool):
    """
    Базовый инструмент для анализа данных с помощью безопасного изолированного выполнения.
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

    def __init__(
        self,
        df: pd.DataFrame,
        execution_timeout_sec: float = 25.0,
        include_plotly: bool = False,
        tool_cache_size: int = 48,
        db_runtime_config: "RuntimeDBConnectionConfig | None" = None,
    ) -> None:
        super().__init__()
        self._df = df
        self._include_plotly = include_plotly
        self.execution_timeout_sec = execution_timeout_sec
        self.tool_cache_size = max(0, int(tool_cache_size))
        self._dataset_signature = self._build_dataset_signature(df)
        self._db_runtime_config = db_runtime_config

    @staticmethod
    def _build_dataset_signature(df: pd.DataFrame) -> str:
        head = df.head(8).to_csv(index=False)
        tail = df.tail(8).to_csv(index=False)
        columns = ",".join(str(col) for col in df.columns[:64])
        payload = f"{df.shape}|{columns}|{head}|{tail}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _cache_key(self, code: str) -> str:
        payload = (
            f"{self.name}|{self._dataset_signature}|{self.execution_timeout_sec}|{code}"
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
                # Совместимость с legacy-ответами, где tool_result = {"name": payload}
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

    @staticmethod
    def _pick_start_method() -> str:
        methods = multiprocessing.get_all_start_methods()
        if "forkserver" in methods:
            return "forkserver"
        if "spawn" in methods:
            return "spawn"
        return "fork"

    def get_execution_scope(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def _extract_payload_hints(tool_result: object) -> dict[str, Any]:
        return extract_artifact_hints(tool_result)

    def _execute_in_sandbox(self, code: str) -> object:
        ctx = multiprocessing.get_context(self._pick_start_method())
        result_queue = ctx.Queue(maxsize=1)
        extra_scope = self.get_execution_scope()
        process = ctx.Process(
            target=_tool_worker,
            args=(
                result_queue,
                code,
                self._df,
                self._include_plotly,
                tuple(sorted(self.allowed_libs)),
                self._db_runtime_config,
                extra_scope,
            ),
            daemon=True,
        )
        process.start()
        process.join(timeout=self.execution_timeout_sec)

        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            raise TimeoutError(
                f"Превышен лимит выполнения инструмента ({self.execution_timeout_sec} сек)"
            )

        try:
            payload = result_queue.get_nowait()
        except queue.Empty:
            raise RuntimeError("Инструмент завершился без результата")

        if not payload.get("ok", False):
            raise RuntimeError(payload.get("error", "Неизвестная ошибка выполнения"))

        return payload.get("result")

    def _run(self, code: str) -> tuple[str, dict[str, object]]:
        code = _normalize_tool_code(code)
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
