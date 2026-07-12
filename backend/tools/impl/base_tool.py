import ast
import copy
import hashlib
import logging
import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from backend.agent.callbacks import strip_thinking
from backend.agent.llm_client import make_reasoning_llm
from backend.artifacts.artifact_meta import extract_artifact_hints
from backend.core.redaction import sanitize_error_text

if TYPE_CHECKING:
    from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
    from backend.tools.sandbox import SessionSandbox

from backend.tools.sandbox import normalize_code

logger = logging.getLogger(__name__)


class ToolResultEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0")
    artifact_type: str
    items: dict[str, object]


class _CodeInput(BaseModel):
    code: str = Field(description="Валидный Python-код для выполнения в sandbox-окружении.")


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
    code_fix_max_retries: int = 3

    # Subclass thinking default for internal LLM calls.
    # Effective thinking = settings.llm_enable_thinking AND TOOL_ENABLE_THINKING.
    # _fix_with_llm() always uses enable_thinking=False regardless of this flag.
    TOOL_ENABLE_THINKING: ClassVar[bool] = False
    _df: pd.DataFrame = PrivateAttr()
    _include_plotly: bool = PrivateAttr(default=False)
    _tool_cache: OrderedDict[str, tuple[str, dict[str, object]]] = PrivateAttr(
        default_factory=OrderedDict
    )
    _dataset_signature: str = PrivateAttr(default="")
    _db_runtime_config: "RuntimeDBConnectionConfig | None" = PrivateAttr(default=None)
    _sandbox: "SessionSandbox | None" = PrivateAttr(default=None)
    _llm_base_url: str | None = PrivateAttr(default=None)
    _llm_model: str | None = PrivateAttr(default=None)
    _llm_api_key: str | None = PrivateAttr(default=None)
    _llm_enable_thinking: bool = PrivateAttr(default=False)
    _llm_chat_template_kwargs_enabled: bool = PrivateAttr(default=True)
    _llm_provider: str = PrivateAttr(default="")

    def __init__(
        self,
        df: pd.DataFrame,
        execution_timeout_sec: float = 25.0,
        include_plotly: bool = False,
        tool_cache_size: int = 48,
        db_runtime_config: "RuntimeDBConnectionConfig | None" = None,
        sandbox: "SessionSandbox | None" = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_enable_thinking: bool = False,
        llm_chat_template_kwargs_enabled: bool = True,
        llm_provider: str = "",
        code_fix_max_retries: int = 3,
    ) -> None:
        super().__init__()
        self._df = df
        self._include_plotly = include_plotly
        self.execution_timeout_sec = execution_timeout_sec
        self.tool_cache_size = max(0, int(tool_cache_size))
        self.code_fix_max_retries = max(0, int(code_fix_max_retries))
        self._dataset_signature = self._build_dataset_signature(df)
        self._db_runtime_config = db_runtime_config
        self._sandbox = sandbox
        self._llm_base_url = llm_base_url
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key
        # effective = global setting AND this tool class's default
        self._llm_enable_thinking = llm_enable_thinking and type(self).TOOL_ENABLE_THINKING
        self._llm_chat_template_kwargs_enabled = llm_chat_template_kwargs_enabled
        self._llm_provider = str(llm_provider or "").strip().lower()

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
            "json_data": "json",
            "structured": "json",
            "search_result": "json",
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

    def _try_run_once(self, code: str) -> tuple[bool, str, dict[str, object]]:
        """Execute code once. Returns (success, message, payload).

        On success: success=True, message=✅ text, payload=full artifact dict.
        On failure: success=False, message=clean error string, payload={}.
        """
        try:
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

            text = (
                f"✅ Создано через {self.name} - {len(normalized_result)} {self.human_name}: "
                f"{', '.join(normalized_result.keys())}"
            )

            payload: dict[str, object] = {
                "text": text,
                "code": code,
                "artifact_type": self.artifact_name,
                "items": normalized_result,
            }

            if artifact_hints:
                payload.update(artifact_hints)
            payload[self.artifact_name] = normalized_result
            return True, text, payload

        except SyntaxError as e:
            code_lines = code.splitlines()
            error_line = (
                code_lines[e.lineno - 1]
                if e.lineno and e.lineno <= len(code_lines)
                else ""
            )
            return False, f"SyntaxError: {e.msg}\n{error_line}", {}
        except Exception as e:
            logger.exception("Tool %s failed while executing code", self.name)
            return False, str(e) or e.__class__.__name__, {}

    def _fix_with_llm(self, code: str, error: str, attempt: int) -> str | None:
        """Ask LLM to fix broken code given the error message. Returns fixed code or None."""
        if not self._llm_base_url or not self._llm_model:
            return None

        llm = make_reasoning_llm(
            provider=self._llm_provider,
            model=self._llm_model,
            base_url=self._llm_base_url,
            api_key=self._llm_api_key or "no-key",
            enable_thinking=False,
            temperature=0.0,
            max_tokens=2048,
            streaming=False,
            chat_template_kwargs_enabled=self._llm_chat_template_kwargs_enabled,
        )

        # Extract the first ~400 chars of the tool description to give the LLM scope context.
        scope_hint = (self.description or "")[:400].strip()

        prompt = f"""Fix the Python code for {self.name} (attempt {attempt}).

Tool scope (available variables):
{scope_hint}

Failed code:
{code}

Error:
{error}

Return ONLY the corrected Python code. No markdown, no explanations, no code fences."""

        try:
            resp = llm.invoke([
                SystemMessage(content="Fix code. Return Python only."),
                HumanMessage(content=prompt),
            ])
            fixed = strip_thinking(str(resp.content or "")).strip()
            # Strip markdown code fences if LLM adds them despite instructions.
            fixed = re.sub(r"^```(?:python)?\s*", "", fixed, flags=re.IGNORECASE)
            fixed = re.sub(r"\s*```$", "", fixed)
            return normalize_code(fixed.strip()) or None
        except Exception as exc:
            logger.debug("code_fix_with_llm failed on attempt %d: %s", attempt, exc)
            return None

    def _run(self, code: str) -> tuple[str, dict[str, object]]:
        # Strip any leaked <think>...</think> blocks (e.g. when the upstream LLM
        # embeds reasoning inside the generated code argument).
        code = normalize_code(strip_thinking(code))
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

        current_code = code
        last_error = ""

        for attempt in range(1 + self.code_fix_max_retries):
            ok, msg, payload = self._try_run_once(current_code)
            if ok:
                result = (msg, payload)
                self._cache_set(cache_key, result)
                return result

            last_error = msg

            if attempt < self.code_fix_max_retries:
                fixed = self._fix_with_llm(current_code, last_error, attempt + 1)
                if fixed and fixed != current_code:
                    logger.debug(
                        "%s: code fix attempt %d/%d — retrying after error: %s",
                        self.name, attempt + 1, self.code_fix_max_retries, last_error[:120],
                    )
                    current_code = fixed
                    continue
                # LLM unavailable or returned identical code — no point retrying.
                break

        text = f"❌ Ошибка при создании {self.human_name}: {last_error}"
        return text, {self.artifact_name: None, "text": text}
