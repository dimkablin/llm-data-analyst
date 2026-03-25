"""
Local deep research service.

Runs an iterative search-and-analyse loop entirely inside the llm-data-analyst
process — no external deep-research backend required.

Flow per research job
─────────────────────
1. Iteration 0   : use the original user query as-is.
2. Iteration N>0 : LLM generates the next search query based on all findings
                   collected so far; may also decide to FINISH early.
3. After every iteration ≥ 2 the supervisor LLM can emit FINISH.
4. Final step    : LLM writes a structured Markdown report from all findings.

The result is returned as a DeepResearchQueryResult (same dataclass as the
HTTP-based DeepResearchIntegrationService), so the tool layer is unchanged.
"""
from __future__ import annotations

import copy
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.artifact_meta import build_source_query_recipe_step
from backend.config import Settings, settings as _global_settings
from backend.deep_research_integration import DeepResearchIntegrationError
from backend.integration_contract import build_operation_meta, build_source_descriptor
from backend.search_integration import SearchIntegrationService, SearchIntegrationError


# ─── Prompts ──────────────────────────────────────────────────────────────────

_SYSTEM_DEEP_RESEARCH = """\
Ты — эксперт по глубокому веб-исследованию. Твоя задача — максимально \
полно и достоверно исследовать тему, используя несколько итераций поиска.

Правила:
• Каждую итерацию формулируй новый, более конкретный или дополнительный \
  поисковый запрос — не повторяй уже использованные.
• Перекрёстно проверяй факты из разных источников.
• Приоритизируй свежие и авторитетные источники.
• НЕ останавливайся на первой итерации, если информации недостаточно \
  или она противоречива.
• Итоговый отчёт должен быть структурированным, с разделами и \
  цитированием конкретных источников (URL).
"""

_ITERATION_PROMPT = """\
Исследование запроса: "{query}"

Итерация {iteration} из {max_iterations}.

Уже собранные данные:
{findings}

Задача: сформулируй следующий поисковый запрос для получения новой информации.
Также реши, нужно ли продолжать исследование.

Правила:
• Если итерация ≥ 2 и данных достаточно — можно указать FINISH.
• Если данных мало или они противоречивы — укажи CONTINUE.
• Новый запрос должен отличаться от предыдущих и уточнять пробелы.

Ответь ТОЛЬКО валидным JSON (без markdown-кода, без комментариев):
{{
  "next_query": "следующий поисковый запрос на русском",
  "decision": "CONTINUE",
  "reason": "краткое объяснение"
}}
"""

_REPORT_PROMPT = """\
Исследование завершено.

Оригинальный запрос: "{query}"

Все собранные данные по итерациям:
{findings}

Создай итоговый структурированный отчёт на русском языке в формате Markdown.
Структура отчёта:
1. **Краткое резюме** — 2–3 предложения с главным выводом.
2. **Основные разделы** — детальный анализ по ключевым аспектам темы.
3. **Источники** — список URL, использованных при исследовании.

Требования:
• Факты из разных источников сопоставляй и объясняй противоречия.
• Указывай источники в тексте в формате [название](url).
• Пиши конкретно, без воды.
"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_bool_env(name: str, *, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce_positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _llm_text(content: Any) -> str:
    """Extract plain text from an LLM AIMessage.content, stripping thinking blocks."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text") or item.get("content") or ""
                if isinstance(t, str) and t:
                    parts.append(t)
            elif isinstance(item, str):
                parts.append(item)
        text = " ".join(parts)
    else:
        text = str(content)

    # Strip <think>...</think> blocks (Qwen3 / other reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _parse_json_block(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a text string."""
    # Try the whole string first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Find the outermost {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def _format_findings(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "(нет данных)"
    lines: list[str] = []
    prev_iter: int | None = None
    for f in findings:
        it = f.get("iteration", 0)
        if it != prev_iter:
            lines.append(f"\n### Итерация {it} — запрос: «{f.get('query', '')}»")
            prev_iter = it
        title = f.get("title", "")
        url = f.get("url", "")
        snippet = (f.get("snippet") or "").strip()
        entry = f"- [{title}]({url})"
        if snippet:
            preview = snippet[:300].replace("\n", " ")
            entry += f"\n  {preview}"
        lines.append(entry)
    return "\n".join(lines)


# ─── Dataclass (reused from deep_research_integration) ────────────────────────

@dataclass(frozen=True)
class DeepResearchQueryResult:
    query: str
    research_id: str
    status: str | None
    summary: str | None
    report_text: str | None
    rows: list[dict[str, Any]]
    sources: list[str]
    warnings: list[str]
    request_params: dict[str, Any]

    @property
    def result_count(self) -> int:
        return len(self.rows)


# ─── Service ──────────────────────────────────────────────────────────────────

class LocalDeepResearchService:
    """
    Iterative deep research running inside the llm-data-analyst process.

    Uses SearchIntegrationService (→ search_service bridge → SearXNG) for
    each iteration, and the project's own LLM for query generation and
    report writing.
    """

    SOURCE_TYPE = "deep_research"
    SOURCE_REF_ID = "deep_research"
    SOURCE_LABEL = "Deep Research"
    SOURCE_MODE = "local"

    def __init__(
        self,
        search_service: SearchIntegrationService,
        settings: Settings,
        *,
        enabled: bool = True,
        default_max_iterations: int = 4,
        default_language: str = "ru",
    ) -> None:
        self._search = search_service
        self._settings = settings
        self._enabled_flag = enabled
        self._default_max_iterations = default_max_iterations
        self._default_language = default_language
        self._llm: ChatOpenAI | None = None  # lazy init

    @classmethod
    def from_env(
        cls,
        search_service: SearchIntegrationService,
        *,
        app_settings: Settings | None = None,
    ) -> "LocalDeepResearchService":
        s = app_settings or _global_settings
        enabled = _get_bool_env("DEEP_RESEARCH_ENABLED", default=True)
        max_iter = _coerce_positive_int(
            os.getenv("DEEP_RESEARCH_MAX_ITERATIONS_DEFAULT"), default=4
        )
        language = (os.getenv("DEEP_RESEARCH_LANGUAGE_DEFAULT") or "ru").strip()
        return cls(
            search_service=search_service,
            settings=s,
            enabled=enabled,
            default_max_iterations=max_iter,
            default_language=language,
        )

    # ── Public interface (matches DeepResearchIntegrationService) ──────────────

    @property
    def is_enabled(self) -> bool:
        return self._enabled_flag and self._search.is_enabled

    def source_ref(self) -> dict[str, str]:
        return {
            "source_type": self.SOURCE_TYPE,
            "source_ref_id": self.SOURCE_REF_ID,
            "source_label": self.SOURCE_LABEL,
            "source_mode": self.SOURCE_MODE,
        }

    def source_descriptor(self) -> dict[str, Any]:
        return build_source_descriptor(
            source_type=self.SOURCE_TYPE,
            source_ref_id=self.SOURCE_REF_ID,
            source_label=self.SOURCE_LABEL,
            display_name_ru="Глубокое исследование",
            source_mode=self.SOURCE_MODE,
            enabled=self.is_enabled,
            available=self.is_enabled,
            description="Local iterative deep research using LLM + search bridge.",
            description_ru="Итеративный глубокий поиск через LLM и поисковый сервис.",
            capabilities=["deep_research", "research_report", "web_results"],
            requires_session_data=False,
            timeout_hint_sec=300.0,
        )

    def build_artifact_payload(
        self,
        result: DeepResearchQueryResult,
        *,
        artifact_name: str = "deep_research_report",
        tool_name: str = "deep_research_tool",
    ) -> dict[str, Any]:
        clean_name = (artifact_name or "deep_research_report").strip() or "deep_research_report"
        return {
            "artifact_name": clean_name,
            "rows": copy.deepcopy(result.rows),
            "source": self.source_ref(),
            "recipe": [
                build_source_query_recipe_step(
                    query=result.query,
                    source_type=self.SOURCE_TYPE,
                    tool_name=tool_name,
                    title="Deep Research Query",
                    summary=result.summary or result.report_text or f"Deep research: {result.query}",
                    params=result.request_params,
                    result_count=result.result_count,
                )
            ],
            "meta": {
                "deep_research": build_operation_meta(
                    status=result.status,
                    warnings=result.warnings,
                    request_params=result.request_params,
                    timeout_sec=300.0,
                    extra={
                        "query": result.query,
                        "research_id": result.research_id,
                        "summary": result.summary,
                        "report_text": result.report_text,
                        "result_count": result.result_count,
                        "sources": list(result.sources),
                    },
                )
            },
        }

    def run_research(
        self,
        query: str,
        *,
        max_iterations: int | None = None,
        language: str | None = None,
    ) -> DeepResearchQueryResult:
        if not self.is_enabled:
            raise DeepResearchIntegrationError(
                "LocalDeepResearchService is disabled. "
                "Set DEEP_RESEARCH_ENABLED=true and configure SEARCH_BACKEND_URL."
            )

        clean_query = query.strip()
        if not clean_query:
            raise DeepResearchIntegrationError("Research query must not be empty.")

        n_iter = _coerce_positive_int(max_iterations, default=self._default_max_iterations)
        lang = (language or self._default_language or "ru").strip()
        research_id = uuid.uuid4().hex[:12]

        request_params: dict[str, Any] = {
            "query": clean_query,
            "max_iterations": n_iter,
            "language": lang,
        }

        findings: list[dict[str, Any]] = []
        all_sources: list[str] = []
        warnings: list[str] = []
        current_query = clean_query

        llm = self._get_llm()

        for iteration in range(n_iter):
            # ── Search ────────────────────────────────────────────────────────
            try:
                search_result = self._search.search(
                    current_query,
                    language=lang if lang not in ("ru", "all") else None,
                )
                for item in search_result.results:
                    findings.append({
                        "iteration": iteration + 1,
                        "query": current_query,
                        "title": item.title,
                        "url": item.url,
                        "snippet": item.snippet or "",
                    })
                    if item.url:
                        all_sources.append(item.url)
            except SearchIntegrationError as exc:
                warnings.append(f"Iteration {iteration + 1} search failed: {exc}")

            # ── Decide next query or finish ───────────────────────────────────
            if iteration >= n_iter - 1:
                # Last iteration — no point asking LLM
                break

            decision, next_query = self._supervisor_step(
                llm=llm,
                query=clean_query,
                findings=findings,
                iteration=iteration + 1,
                max_iterations=n_iter,
            )

            if decision == "FINISH" and iteration >= 1:
                break

            current_query = next_query or clean_query

        # ── Write final report ────────────────────────────────────────────────
        report_text = self._write_report(llm, clean_query, findings)
        summary = self._extract_summary(report_text)

        rows = self._build_rows(findings)
        deduped_sources = list(dict.fromkeys(all_sources))

        return DeepResearchQueryResult(
            query=clean_query,
            research_id=research_id,
            status="completed",
            summary=summary,
            report_text=report_text,
            rows=rows,
            sources=deduped_sources,
            warnings=warnings,
            request_params=copy.deepcopy(request_params),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self._settings.llm_model,
                base_url=self._settings.llm_base_url,
                api_key=self._settings.llm_api_key,
                temperature=0.7,
                max_tokens=self._settings.llm_max_tokens_reasoning,
                timeout=120,
                streaming=False,
            )
        return self._llm

    def _supervisor_step(
        self,
        *,
        llm: ChatOpenAI,
        query: str,
        findings: list[dict[str, Any]],
        iteration: int,
        max_iterations: int,
    ) -> tuple[str, str]:
        """Ask LLM for next query and whether to continue. Returns (decision, next_query)."""
        findings_text = _format_findings(findings)
        prompt = _ITERATION_PROMPT.format(
            query=query,
            iteration=iteration,
            max_iterations=max_iterations,
            findings=findings_text,
        )
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=_SYSTEM_DEEP_RESEARCH),
                    HumanMessage(content=prompt),
                ]
            )
            text = _llm_text(response.content)
            parsed = _parse_json_block(text)
            if parsed:
                decision = str(parsed.get("decision") or "CONTINUE").upper().strip()
                next_query = str(parsed.get("next_query") or query).strip()
                if decision not in {"CONTINUE", "FINISH"}:
                    decision = "CONTINUE"
                return decision, next_query or query
        except Exception:
            pass

        return "CONTINUE", query

    def _write_report(
        self,
        llm: ChatOpenAI,
        query: str,
        findings: list[dict[str, Any]],
    ) -> str:
        findings_text = _format_findings(findings)
        prompt = _REPORT_PROMPT.format(query=query, findings=findings_text)
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=_SYSTEM_DEEP_RESEARCH),
                    HumanMessage(content=prompt),
                ]
            )
            return _llm_text(response.content)
        except Exception as exc:
            fallback = findings_text or f"Ошибка генерации отчёта: {exc}"
            return fallback

    @staticmethod
    def _extract_summary(report_text: str | None) -> str | None:
        if not report_text:
            return None
        lines = [ln.strip() for ln in report_text.splitlines() if ln.strip()]
        # First non-header line gives a good summary
        for line in lines:
            if line.startswith("#"):
                continue
            if len(line) > 20:
                return line[:400]
        return lines[0][:400] if lines else None

    @staticmethod
    def _build_rows(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, f in enumerate(findings, start=1):
            rows.append(
                {
                    "rank": idx,
                    "kind": "finding",
                    "title": f.get("title") or f.get("query") or "",
                    "content": f.get("snippet") or "",
                    "url": f.get("url") or "",
                    "source_name": "",
                }
            )
        return rows
