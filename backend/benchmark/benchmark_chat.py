#!/usr/bin/env python3
"""Benchmark runner for LLM Data Analyst.

Runs TXT/MD questions or YAML/JSON benchmark cases in one chat session, captures
tool events, measures latency, optionally runs LLM-as-a-judge, and writes JSONL,
CSV, Markdown, and a readable HTML chat report.

See backend/benchmark/README_benchmark.md for Russian usage notes and examples.
"""
from __future__ import annotations

import argparse
import atexit
import csv
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    import markdown2
except Exception:  # noqa: BLE001 - markdown rendering is a presentation enhancement.
    markdown2 = None  # type: ignore[assignment]


DEFAULT_BASE_URL = "http://localhost:8605"
MARKDOWN_EXTRAS = [
    "fenced-code-blocks",
    "tables",
    "strike",
    "cuddled-lists",
    "break-on-newline",
    "code-friendly",
    "task_list",
]
LATENCY_PENALTY_STEPS: list[tuple[int, int, str]] = [
    (120_000, 35, "slow_gt_120s"),
    (90_000, 25, "slow_gt_90s"),
    (60_000, 18, "slow_gt_60s"),
    (40_000, 10, "slow_gt_40s"),
    (30_000, 5, "slow_gt_30s"),
]


@dataclass
class JudgeConfig:
    enabled: bool = False
    base_url: str = ""
    model: str = ""
    api_key: str = "EMPTY"
    timeout: int = 120
    temperature: float = 0.0


@dataclass
class ToolRun:
    tool_name: str
    status: str = "started"
    started_at: float = 0.0
    finished_at: float | None = None
    duration_ms: int | None = None
    input_summary: str = ""
    input_preview: str = ""
    output_preview: str = ""
    result_summary: str = ""
    artifact_keys: list[str] = field(default_factory=list)
    error: str = ""
    start_payload: dict[str, Any] = field(default_factory=dict)
    end_payload: dict[str, Any] = field(default_factory=dict)


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def _env_value(name: str, env_file: dict[str, str], default: str = "") -> str:
    return (os.getenv(name) or env_file.get(name) or default).strip()


def _chat_completions_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    if clean.endswith("/v1"):
        return f"{clean}/chat/completions"
    return f"{clean}/v1/chat/completions"


def _request_json(
    method: str,
    url: str,
    *,
    token: str | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            return json.loads(payload) if payload.strip() else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} -> HTTP {exc.code}: {detail}") from exc


def _get_auth_settings(base_url: str, token: str) -> dict[str, Any]:
    return _request_json("GET", f"{base_url}/auth/settings", token=token, timeout=60)


def _patch_auth_settings(base_url: str, token: str, patch: dict[str, Any]) -> dict[str, Any]:
    return _request_json("PATCH", f"{base_url}/auth/settings", token=token, body=patch, timeout=60)


def _upload_csv(url: str, token: str, csv_path: Path, timeout: int = 120) -> dict[str, Any]:
    boundary = "----BenchmarkFormBoundary7MA4YWxkTrZu0gW"
    file_bytes = csv_path.read_bytes()
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{csv_path.name}"\r\n'
        "Content-Type: text/csv\r\n\r\n"
    ).encode("utf-8")
    suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    req = urllib.request.Request(url, data=prefix + file_bytes + suffix, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} -> HTTP {exc.code}: {detail}") from exc


def _trim(text: Any, limit: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def _strip_markdown_numbering(line: str) -> str:
    clean = line.strip()
    clean = re.sub(r"^\s*[-*]\s+", "", clean)
    clean = re.sub(r"^\s*\d+[.)]\s+", "", clean)
    return clean.strip()


def _load_yaml_or_json(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return None
    if path.suffix.lower() == ".json":
        return json.loads(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-untyped]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{path} is not JSON and PyYAML is unavailable") from exc
        return yaml.safe_load(raw)


def _case_from_mapping(item: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    question = str(item.get("question") or item.get("prompt") or item.get("query") or "").strip()
    if not question:
        raise RuntimeError(f"YAML case #{fallback_index} does not include question/prompt/query")
    case_id = str(item.get("id") or item.get("case_id") or fallback_index)
    expectation = {
        key: value
        for key, value in item.items()
        if key not in {"id", "case_id", "question", "prompt", "query", "expectation", "expectations", "expected"}
    }
    nested = item.get("expectation") or item.get("expectations") or item.get("expected")
    if isinstance(nested, dict):
        expectation.update(nested)
    return {"id": case_id, "question": question, "expectation": expectation}


def load_cases(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".yaml", ".yml", ".json"}:
        data = _load_yaml_or_json(path)
        defaults: dict[str, Any] = {}
        raw_cases: Any = data
        if isinstance(data, dict):
            defaults = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}
            raw_cases = data.get("cases") or data.get("questions") or data.get("prompts")
        if not isinstance(raw_cases, list):
            raise RuntimeError("YAML/JSON questions file must be a list or contain cases/questions list")
        cases: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_cases, start=1):
            if isinstance(item, str):
                case = {"id": str(idx), "question": item.strip(), "expectation": dict(defaults)}
            elif isinstance(item, dict):
                case = _case_from_mapping(item, idx)
                expectation = dict(defaults)
                expectation.update(case.get("expectation") or {})
                case["expectation"] = expectation
            else:
                raise RuntimeError(f"YAML case #{idx} must be a string or object")
            if case["question"]:
                cases.append(case)
        return cases

    return [
        {"id": str(idx), "question": question, "expectation": {}}
        for idx, question in enumerate(load_questions(path), start=1)
    ]


def load_questions(path: Path) -> list[str]:
    questions: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("---"):
            continue
        clean = _strip_markdown_numbering(line)
        if not clean:
            continue
        # Skip explanatory prose in bundled prompt files; keep actual prompts.
        if not clean.endswith(("?", ".", "!", ":", ")", "»")) and len(clean.split()) < 4:
            continue
        if clean.lower().startswith(("примеры ", "эти ", "они ", "работают ")):
            continue
        questions.append(clean)
    return questions


def _answer_has_fallback(text: str) -> bool:
    lower = text.lower()
    markers = (
        "не смог завершить полноценный анализ",
        "fallback response",
        "модель сейчас недоступна",
        "не смог сформировать содержательный ответ",
        "не вернул финальный ответ",
    )
    return any(marker in lower for marker in markers)


def _answer_has_required_tool_unavailable(text: str) -> bool:
    lower = text.lower()
    return (
        "необходимый tool" in lower
        and ("выключен" in lower or "недоступен" in lower)
    ) or ("required tool" in lower and "disabled" in lower)


def _answer_has_iteration_limit(text: str) -> bool:
    lower = text.lower()
    return "огранич" in lower and ("итерац" in lower or "шаг" in lower)


def _question_needs_plot(question: str) -> bool:
    lower = question.lower()
    return any(
        marker in lower
        for marker in (
            "график",
            "диаграм",
            "визуал",
            "гистограмм",
            "boxplot",
            "sankey",
            "heatmap",
            "теплов",
            "кругов",
            "столбчат",
            "линейн",
        )
    )


def _question_needs_artifact(question: str) -> bool:
    lower = question.lower()
    return _question_needs_plot(question) or any(
        marker in lower
        for marker in (
            "таблиц",
            "топ",
            "посчитай",
            "сколько",
            "сравни",
            "найди",
            "покажи",
            "построй",
            "анализ",
        )
    )


def _extract_artifact_stats(final_payload: dict[str, Any] | None) -> dict[str, Any]:
    artifacts = (final_payload or {}).get("artifacts") if isinstance(final_payload, dict) else []
    if not isinstance(artifacts, list):
        artifacts = []

    counts: Counter[str] = Counter()
    summaries: list[dict[str, Any]] = []
    plot_trace_count = 0
    table_rows_previewed = 0
    value_keys: list[str] = []

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        artifact_type = str(artifact.get("type") or "unknown").strip().lower() or "unknown"
        counts[artifact_type] += 1
        title = str(artifact.get("text") or artifact_type).strip()
        data = artifact.get("data") if isinstance(artifact.get("data"), dict) else {}
        fmt = str(data.get("format") or "").strip()
        raw_data = data.get("data")

        summary: dict[str, Any] = {
            "id": artifact.get("id"),
            "type": artifact_type,
            "title": title,
            "format": fmt,
        }
        if artifact_type == "plot" and isinstance(raw_data, dict):
            traces = raw_data.get("data")
            traces_count = len(traces) if isinstance(traces, list) else 0
            plot_trace_count += traces_count
            summary["traces"] = traces_count
        elif artifact_type == "table" and isinstance(raw_data, dict):
            rows = raw_data.get("data")
            columns = raw_data.get("columns")
            row_count = len(rows) if isinstance(rows, list) else 0
            col_count = len(columns) if isinstance(columns, list) else 0
            table_rows_previewed += row_count
            summary["rows_previewed"] = row_count
            summary["columns"] = col_count
        elif artifact_type == "value" and isinstance(raw_data, dict):
            keys = [str(key) for key in raw_data]
            value_keys.extend(keys)
            summary["keys"] = keys[:20]
        summaries.append(summary)

    return {
        "artifact_summaries": summaries,
        "artifact_count_from_payload": len(artifacts),
        "table_count_from_payload": int(counts.get("table", 0)),
        "plot_count_from_payload": int(counts.get("plot", 0)),
        "value_count_from_payload": int(counts.get("value", 0)),
        "json_count_from_payload": int(counts.get("json", 0)),
        "plot_trace_count": plot_trace_count,
        "has_valid_plot": plot_trace_count > 0,
        "table_rows_previewed": table_rows_previewed,
        "value_keys": value_keys[:50],
    }


def _artifact_evidence(final_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    artifacts = (final_payload or {}).get("artifacts") if isinstance(final_payload, dict) else []
    if not isinstance(artifacts, list):
        return []
    evidence: list[dict[str, Any]] = []
    for artifact in artifacts[:20]:
        if not isinstance(artifact, dict):
            continue
        artifact_type = str(artifact.get("type") or "unknown").strip().lower() or "unknown"
        data = artifact.get("data") if isinstance(artifact.get("data"), dict) else {}
        raw_data = data.get("data")
        item: dict[str, Any] = {
            "id": artifact.get("id"),
            "type": artifact_type,
            "title": artifact.get("text") or artifact_type,
            "format": data.get("format"),
        }
        if artifact_type == "table" and isinstance(raw_data, dict):
            rows = raw_data.get("data") if isinstance(raw_data.get("data"), list) else []
            columns = raw_data.get("columns") if isinstance(raw_data.get("columns"), list) else []
            item["columns"] = columns[:20]
            item["rows_preview"] = rows[:8]
        elif artifact_type == "value" and isinstance(raw_data, dict):
            item["values"] = dict(list(raw_data.items())[:40])
        elif artifact_type == "plot" and isinstance(raw_data, dict):
            traces = raw_data.get("data") if isinstance(raw_data.get("data"), list) else []
            layout = raw_data.get("layout") if isinstance(raw_data.get("layout"), dict) else {}
            item["layout_title"] = layout.get("title")
            item["traces"] = [
                {
                    "type": trace.get("type"),
                    "name": trace.get("name"),
                    "x_preview": trace.get("x", [])[:8] if isinstance(trace.get("x"), list) else trace.get("x"),
                    "y_preview": trace.get("y", [])[:8] if isinstance(trace.get("y"), list) else trace.get("y"),
                    "labels_preview": trace.get("labels", [])[:8]
                    if isinstance(trace.get("labels"), list)
                    else trace.get("labels"),
                    "values_preview": trace.get("values", [])[:8]
                    if isinstance(trace.get("values"), list)
                    else trace.get("values"),
                }
                for trace in traces[:6]
                if isinstance(trace, dict)
            ]
        else:
            item["raw_preview"] = _trim(json.dumps(raw_data, ensure_ascii=False, default=str), 3000)
        evidence.append(item)
    return evidence


def _latency_penalty(duration_ms: int) -> tuple[int, str]:
    for threshold_ms, penalty, label in LATENCY_PENALTY_STEPS:
        if duration_ms > threshold_ms:
            return penalty, label
    return 0, ""


def _score_case(question: str, result: dict[str, Any]) -> tuple[int, list[str], int, str]:
    issues: list[str] = []
    score = 100
    answer = str(result.get("answer_text") or "")
    if not result.get("ok"):
        score -= 45
        issues.append("request_failed")
    if not answer.strip():
        score -= 25
        issues.append("empty_answer")
    if result.get("fallback_used"):
        score -= 35
        issues.append("fallback_used")
    if _answer_has_required_tool_unavailable(answer):
        score -= 45
        issues.append("required_tool_unavailable")
    if _answer_has_iteration_limit(answer):
        score -= 25
        issues.append("iteration_limit")
    if int(result.get("tool_errors") or 0) > 0:
        score -= min(35, 12 * int(result.get("tool_errors") or 0))
        issues.append("tool_errors")
    has_plot = int(result.get("plot_count") or 0) > 0 or bool(result.get("has_valid_plot"))
    if _question_needs_plot(question) and not has_plot:
        score -= 25
        issues.append("missing_plot")
    elif _question_needs_artifact(question) and int(result.get("artifact_count") or 0) == 0:
        score -= 15
        issues.append("missing_artifact")
    if _question_needs_artifact(question) and int(result.get("tool_calls") or 0) == 0:
        score -= 30
        issues.append("no_current_turn_evidence")
    if int(result.get("tool_calls") or 0) > 8:
        score -= min(20, (int(result.get("tool_calls") or 0) - 8) * 3)
        issues.append("many_tool_calls")
    latency_penalty, latency_bucket = _latency_penalty(int(result.get("duration_ms") or 0))
    if latency_penalty:
        score -= latency_penalty
        issues.append(latency_bucket)
    return max(0, min(100, score)), issues, latency_penalty, latency_bucket


def _safe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _extract_json_object(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise RuntimeError("Judge returned JSON, but not an object")
    return data


def _judge_messages(question: str, row: dict[str, Any], expectation: dict[str, Any]) -> list[dict[str, str]]:
    tool_errors = [
        {
            "tool_name": tool.get("tool_name"),
            "error": tool.get("error") or tool.get("output_preview"),
        }
        for tool in row.get("tool_runs", [])
        if isinstance(tool, dict) and tool.get("status") == "error"
    ]
    tool_observations = [
        {
            "tool_name": tool.get("tool_name"),
            "input": _trim(tool.get("input_preview"), 1000),
            "output": _trim(tool.get("output_preview") or tool.get("result_summary"), 4000),
        }
        for tool in row.get("tool_runs", [])
        if isinstance(tool, dict) and tool.get("status") != "error"
    ]
    payload = {
        "current_date": date.today().isoformat(),
        "question": question,
        "answer": _trim(row.get("answer_text"), 12_000),
        "expectations": expectation or {},
        "metrics": {
            "duration_ms": row.get("duration_ms"),
            "tool_calls": row.get("tool_calls"),
            "tool_errors": row.get("tool_errors"),
            "artifact_count": row.get("artifact_count"),
            "plot_count": row.get("plot_count"),
            "table_count": row.get("table_count"),
            "has_valid_plot": row.get("has_valid_plot"),
            "quality_issues": row.get("quality_issues"),
        },
        "artifact_summaries": row.get("artifact_summaries") or [],
        "artifact_evidence": _artifact_evidence(row.get("final_payload") if isinstance(row.get("final_payload"), dict) else None),
        "successful_tool_observations": tool_observations,
        "tool_errors": tool_errors,
    }
    system = (
        "You are a strict QA judge for an LLM data analyst benchmark. "
        "Always write every JSON string value in Russian. "
        "The benchmark answers are expected to be in Russian. "
        "Evaluate whether the assistant answered the user's analytics question correctly. "
        "Use provided expectations when they are present. Expectations may describe required "
        "numbers, facts, conclusions, or insights. If expectations are empty, judge general "
        "answer quality, grounding in artifacts, relevance, completeness, and whether requested "
        "charts/tables were produced. Do not penalize missing charts/tables when the user did not "
        "request them and expectations do not require them. "
        "Be tolerant of rough multiplicative comparisons and rounded ratios when the direction of "
        "the insight is correct and the underlying absolute values are present. Do not treat phrases "
        "like 'several times higher' or approximate 'x times' comparisons as major hallucinations "
        "unless they reverse the conclusion or replace required exact facts. "
        "Tool-call errors are already penalized by the benchmark runner: treat them as context, "
        "and apply only a small penalty when the final answer is still correct and grounded. "
        "For mutable current facts, use the supplied current_date and successful current-run "
        "observations instead of relying on older model memory. "
        "Apply a large penalty for hallucinations: invented numbers not present in answer/artifact "
        "evidence, conclusions unsupported by tables/values/plots, claimed tables or charts that "
        "were not produced, requested plots/tables that are missing, or references to analysis "
        "that is absent from the artifacts. "
        "A correct answer with some failed tool calls should score high; a polished answer with "
        "unsupported data should score low. Return only JSON with this schema: "
        '{"pass": boolean, "score": integer 0-100, "issues": string[], '
        '"missing": string[], "comment": string}.'
    )
    user = (
        "Benchmark case data follows as JSON. Do not require exact wording, but be strict "
        "about hallucinations: missing numbers, wrong conclusions, missing requested plots/tables, "
        "claimed-but-absent artifacts, and unsupported claims. Be less strict about approximate "
        "ratio wording when absolute values and the main conclusion are correct. Mention tool-call errors only as "
        "secondary issues unless they caused an ungrounded or incomplete final answer.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def run_judge(
    config: JudgeConfig,
    *,
    question: str,
    row: dict[str, Any],
    expectation: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "judge_enabled": config.enabled,
        "judge_score": None,
        "judge_pass": None,
        "judge_issues": [],
        "judge_missing": [],
        "judge_comment": "",
        "judge_error": "",
        "judge_duration_ms": None,
        "judge_raw": None,
        "expectation": expectation or {},
    }
    if not config.enabled:
        return result
    if not config.base_url or not config.model:
        result["judge_error"] = "Judge is enabled, but judge base URL or model is empty."
        return result

    body = {
        "model": config.model,
        "messages": _judge_messages(question, row, expectation),
        "temperature": config.temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    started = time.perf_counter()
    req = urllib.request.Request(
        _chat_completions_url(config.base_url),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        result["judge_error"] = f"HTTP {exc.code}: {_trim(detail, 1000)}"
        result["judge_duration_ms"] = int((time.perf_counter() - started) * 1000)
        return result
    except Exception as exc:  # noqa: BLE001
        result["judge_error"] = str(exc)
        result["judge_duration_ms"] = int((time.perf_counter() - started) * 1000)
        return result

    result["judge_duration_ms"] = int((time.perf_counter() - started) * 1000)
    result["judge_raw"] = payload
    content = ""
    try:
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") if isinstance(choice, dict) else {}
        if isinstance(message, dict):
            content = str(message.get("content") or "")
        if not content and isinstance(choice, dict):
            content = str(choice.get("text") or "")
    except Exception:
        content = ""

    try:
        parsed = _extract_json_object(content)
    except Exception as exc:  # noqa: BLE001
        result["judge_error"] = f"Could not parse judge JSON: {exc}; raw={_trim(content, 1000)}"
        return result

    score_raw = parsed.get("score")
    try:
        judge_score = max(0, min(100, int(score_raw)))
    except Exception:
        judge_score = None
    judge_pass = parsed.get("pass")
    if judge_pass is None and judge_score is not None:
        judge_pass = judge_score >= 75

    result.update(
        {
            "judge_score": judge_score,
            "judge_pass": bool(judge_pass) if judge_pass is not None else None,
            "judge_issues": _safe_list(parsed.get("issues")),
            "judge_missing": _safe_list(parsed.get("missing")),
            "judge_comment": str(parsed.get("comment") or ""),
            "judge_parsed": parsed,
        }
    )
    return result


def _summarize_rows_for_llm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for row in rows:
        summary.append(
            {
                "turn_index": row.get("turn_index"),
                "case_id": row.get("case_id"),
                "question": _trim(row.get("question"), 500),
                "quality_score": row.get("quality_score"),
                "heuristic_score": row.get("heuristic_score"),
                "judge_score": row.get("judge_score"),
                "judge_pass": row.get("judge_pass"),
                "duration_sec": round(int(row.get("duration_ms") or 0) / 1000, 1),
                "latency_bucket": row.get("latency_bucket"),
                "tool_calls": row.get("tool_calls"),
                "tool_errors": row.get("tool_errors"),
                "quality_issues": row.get("quality_issues") or [],
                "judge_issues": _safe_list(row.get("judge_issues"))[:8],
                "judge_missing": _safe_list(row.get("judge_missing"))[:8],
                "judge_comment": _trim(row.get("judge_comment"), 1200),
                "judge_error": _trim(row.get("judge_error"), 600),
                "answer_preview": _trim(row.get("answer_text"), 1000),
            }
        )
    return summary


def run_llm_summary(config: JudgeConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "summary_enabled": config.enabled,
        "summary_text": "",
        "summary_error": "",
        "summary_duration_ms": None,
        "summary_raw": None,
    }
    if not config.enabled:
        result["summary_error"] = "LLM summary disabled because judge/LLM config is disabled."
        return result
    if not rows:
        result["summary_error"] = "No rows to summarize."
        return result
    if not config.base_url or not config.model:
        result["summary_error"] = "LLM summary enabled, but base URL or model is empty."
        return result

    payload = {
        "overall": {
            "questions": len(rows),
            "avg_score": round(_avg([int(row.get("quality_score") or 0) for row in rows]), 1),
            "avg_judge_score": round(
                _avg([int(row.get("judge_score")) for row in rows if row.get("judge_score") is not None]),
                1,
            ),
            "total_tool_errors": sum(int(row.get("tool_errors") or 0) for row in rows),
            "slow_turns_gt_30s": sum(int(row.get("duration_ms") or 0) > 30_000 for row in rows),
        },
        "turns": _summarize_rows_for_llm(rows),
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Ты QA-аналитик для benchmark-прогона LLM data analyst. "
                "Пиши только на русском. Составь короткое executive summary для начала HTML-отчета. "
                "Нужно быстро подсветить главные проблемы: галлюцинации, пропущенные ожидаемые инсайты, "
                "неподтвержденные цифры, отсутствующие графики/таблицы, tool errors и медленные вопросы. "
                "Не пересказывай каждый ответ целиком. Верни обычный текст с 3-7 короткими пунктами."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        },
    ]
    body = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    started = time.perf_counter()
    req = urllib.request.Request(
        _chat_completions_url(config.base_url),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            payload_raw = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        result["summary_error"] = f"HTTP {exc.code}: {_trim(detail, 1000)}"
        result["summary_duration_ms"] = int((time.perf_counter() - started) * 1000)
        return result
    except Exception as exc:  # noqa: BLE001
        result["summary_error"] = str(exc)
        result["summary_duration_ms"] = int((time.perf_counter() - started) * 1000)
        return result

    result["summary_duration_ms"] = int((time.perf_counter() - started) * 1000)
    result["summary_raw"] = payload_raw
    try:
        choice = (payload_raw.get("choices") or [{}])[0]
        message = choice.get("message") if isinstance(choice, dict) else {}
        if isinstance(message, dict):
            result["summary_text"] = str(message.get("content") or "").strip()
        if not result["summary_text"] and isinstance(choice, dict):
            result["summary_text"] = str(choice.get("text") or "").strip()
    except Exception as exc:  # noqa: BLE001
        result["summary_error"] = f"Could not extract summary text: {exc}"
    return result


def _apply_judge_score(row: dict[str, Any]) -> None:
    judge_score = row.get("judge_score")
    if judge_score is None:
        return
    base_score = int(row.get("quality_score") or 0)
    combined = round(base_score * 0.6 + int(judge_score) * 0.4)
    issues = list(row.get("quality_issues") or [])
    if row.get("judge_pass") is False:
        combined = min(combined, 70)
        issues.append("judge_failed")
    row["quality_score"] = max(0, min(100, combined))
    row["quality_issues"] = issues


def _match_tool_start(active: deque[ToolRun], tool_name: str) -> ToolRun | None:
    for item in reversed(active):
        if item.tool_name == tool_name and item.finished_at is None:
            return item
    for item in reversed(active):
        if item.finished_at is None:
            return item
    return None


def stream_query(
    base_url: str,
    token: str,
    session_id: str,
    *,
    question: str,
    use_history: bool,
    include_reasoning: bool,
    analysis_depth: str | None,
    selected_skill_ids: list[str] | None,
    timeout: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": question,
        "use_history": use_history,
        "include_reasoning": include_reasoning,
    }
    if analysis_depth:
        body["analysis_depth"] = analysis_depth
    if selected_skill_ids:
        body["selected_skill_ids"] = selected_skill_ids

    req = urllib.request.Request(
        f"{base_url}/sessions/{session_id}/query/stream",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    started = time.perf_counter()
    token_chunks: list[str] = []
    reasoning_chars = 0
    phase_events: list[dict[str, Any]] = []
    tool_runs: list[ToolRun] = []
    active_tools: deque[ToolRun] = deque()
    final_payload: dict[str, Any] | None = None
    api_events: list[dict[str, Any]] = []
    stream_error = ""
    event_name: str | None = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    event_name = None
                    continue
                if line.startswith("event: "):
                    event_name = line[7:].strip()
                    continue
                if not line.startswith("data: "):
                    continue

                payload_raw = line[6:]
                try:
                    payload: Any = json.loads(payload_raw)
                except json.JSONDecodeError:
                    payload = payload_raw

                current_event = event_name
                event_name = None
                api_events.append({"event": current_event or "message", "data": payload})

                if current_event == "token" and isinstance(payload, str):
                    token_chunks.append(payload)
                elif current_event == "reasoning_token" and isinstance(payload, str):
                    reasoning_chars += len(payload)
                elif current_event == "thinking_end" and isinstance(payload, str):
                    reasoning_chars += len(payload)
                elif current_event == "phase" and isinstance(payload, dict):
                    phase_events.append(payload)
                elif current_event == "tool_start" and isinstance(payload, dict):
                    tool = ToolRun(
                        tool_name=str(payload.get("tool_name") or "unknown"),
                        started_at=time.perf_counter(),
                        input_summary=str(payload.get("input_summary") or ""),
                        input_preview=str(payload.get("input_preview") or ""),
                        start_payload=dict(payload),
                    )
                    tool_runs.append(tool)
                    active_tools.append(tool)
                elif current_event == "tool_end" and isinstance(payload, dict):
                    tool_name = str(payload.get("tool_name") or "unknown")
                    tool = _match_tool_start(active_tools, tool_name)
                    if tool is None:
                        tool = ToolRun(tool_name=tool_name, started_at=started)
                        tool_runs.append(tool)
                    tool.status = str(payload.get("status") or "done")
                    tool.finished_at = time.perf_counter()
                    tool.duration_ms = max(0, int((tool.finished_at - tool.started_at) * 1000))
                    tool.output_preview = str(payload.get("output_preview") or "")
                    tool.result_summary = str(payload.get("result_summary") or "")
                    tool.artifact_keys = [
                        str(item) for item in (payload.get("artifact_keys") or []) if str(item).strip()
                    ]
                    tool.end_payload = dict(payload)
                    if tool.status == "error":
                        tool.error = str(payload.get("error") or payload.get("output_preview") or "")
                elif current_event == "final" and isinstance(payload, dict):
                    final_payload = payload
                elif current_event == "error":
                    stream_error = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        stream_error = f"HTTP {exc.code}: {detail}"
    except Exception as exc:  # noqa: BLE001 - CLI should record failures, not crash mid-run.
        stream_error = str(exc)

    wall_duration_ms = int((time.perf_counter() - started) * 1000)
    metrics = (final_payload or {}).get("metrics") if isinstance(final_payload, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    artifacts = (final_payload or {}).get("artifacts") if isinstance(final_payload, dict) else []
    if not isinstance(artifacts, list):
        artifacts = []
    artifact_stats = _extract_artifact_stats(final_payload)

    answer_text = str((final_payload or {}).get("text") or "".join(token_chunks)).strip()
    tool_errors = sum(1 for tool in tool_runs if tool.status == "error")
    tool_names = [tool.tool_name for tool in tool_runs]
    counts = Counter(tool_names)

    result: dict[str, Any] = {
        "ok": bool(final_payload) and not stream_error,
        "error": stream_error,
        "answer_text": answer_text,
        "answer_preview": _trim(answer_text, 600),
        "answer_chars": len(answer_text),
        "fallback_used": _answer_has_fallback(answer_text),
        "wall_duration_ms": wall_duration_ms,
        "duration_ms": int(metrics.get("duration_ms") or wall_duration_ms),
        "llm_duration_ms": int(metrics.get("llm_duration_ms") or 0),
        "non_llm_duration_ms": int(metrics.get("non_llm_duration_ms") or 0),
        "llm_calls": int(metrics.get("llm_calls") or 0),
        "artifact_count": int(metrics.get("artifact_count") or artifact_stats["artifact_count_from_payload"]),
        "table_count": int(metrics.get("table_count") or artifact_stats["table_count_from_payload"]),
        "plot_count": int(metrics.get("plot_count") or artifact_stats["plot_count_from_payload"]),
        "value_count": int(metrics.get("value_count") or artifact_stats["value_count_from_payload"]),
        "json_count": int(metrics.get("json_count") or artifact_stats["json_count_from_payload"]),
        "model": metrics.get("model"),
        "tool_calls": len(tool_runs),
        "tool_errors": tool_errors,
        "contract_valid": bool((final_payload or {}).get("contract_valid", False)),
        "terminal_status": str(
            (final_payload or {}).get("terminal_status") or "failed"
        ),
        "capability_outcomes": list(
            (final_payload or {}).get("capability_outcomes") or []
        ),
        "error_fingerprints": list(
            (final_payload or {}).get("error_fingerprints") or []
        ),
        "retry_count": int((final_payload or {}).get("retry_count") or 0),
        "reported_tool_error_count": int(
            (final_payload or {}).get("tool_error_count") or 0
        ),
        "tool_names": tool_names,
        "tool_name_counts": dict(counts),
        "repeated_tool_calls": sum(count - 1 for count in counts.values() if count > 1),
        "tool_runs": [
            {
                "tool_name": tool.tool_name,
                "status": tool.status,
                "duration_ms": tool.duration_ms,
                "input_summary": tool.input_summary,
                "input_preview": tool.input_preview,
                "result_summary": tool.result_summary,
                "output_preview": tool.output_preview,
                "artifact_keys": tool.artifact_keys,
                "error": tool.error,
                "start_payload": tool.start_payload,
                "end_payload": tool.end_payload,
            }
            for tool in tool_runs
        ],
        "phase_count": len(phase_events),
        "reasoning_chars": reasoning_chars,
        "final_payload": final_payload,
        "api_events": api_events,
        **artifact_stats,
    }
    score, issues, latency_penalty, latency_bucket = _score_case(question, result)
    result["heuristic_score"] = score
    result["quality_score"] = score
    result["quality_issues"] = issues
    result["latency_penalty"] = latency_penalty
    result["latency_bucket"] = latency_bucket
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "turn_index",
        "case_id",
        "question",
        "ok",
        "quality_score",
        "heuristic_score",
        "quality_issues",
        "latency_penalty",
        "latency_bucket",
        "judge_enabled",
        "judge_score",
        "judge_pass",
        "judge_issues",
        "judge_missing",
        "judge_comment",
        "judge_error",
        "judge_duration_ms",
        "duration_ms",
        "llm_duration_ms",
        "non_llm_duration_ms",
        "llm_calls",
        "wall_duration_ms",
        "tool_calls",
        "tool_errors",
        "reported_tool_error_count",
        "contract_valid",
        "terminal_status",
        "retry_count",
        "error_fingerprints",
        "tool_names",
        "repeated_tool_calls",
        "artifact_count",
        "table_count",
        "plot_count",
        "plot_trace_count",
        "has_valid_plot",
        "value_count",
        "answer_chars",
        "fallback_used",
        "history_question_count",
        "history_answer_chars_before",
        "approx_context_chars_before",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    ", ".join(str(item) for item in row.get(key, []))
                    if key in {"quality_issues", "tool_names", "judge_issues", "judge_missing"}
                    and isinstance(row.get(key), list)
                    else row.get(key)
                )
                for key in columns
            })


def _avg(values: list[int | float]) -> float:
    clean = [float(v) for v in values if v is not None]
    return sum(clean) / len(clean) if clean else 0.0


def _write_markdown(path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    scores = [int(row.get("quality_score") or 0) for row in rows]
    heuristic_scores = [int(row.get("heuristic_score") or row.get("quality_score") or 0) for row in rows]
    durations = [int(row.get("duration_ms") or 0) for row in rows]
    llm_durations = [int(row.get("llm_duration_ms") or 0) for row in rows]
    non_llm_durations = [int(row.get("non_llm_duration_ms") or 0) for row in rows]
    tool_errors = [int(row.get("tool_errors") or 0) for row in rows]
    latency_penalties = [int(row.get("latency_penalty") or 0) for row in rows]
    judge_scores = [int(row.get("judge_score")) for row in rows if row.get("judge_score") is not None]
    n = len(rows)
    first = rows[: max(1, n // 4)] if rows else []
    last = rows[-max(1, n // 4):] if rows else []
    first_score = _avg([int(row.get("quality_score") or 0) for row in first])
    last_score = _avg([int(row.get("quality_score") or 0) for row in last])
    quality_drop = first_score - last_score
    first_duration = _avg([int(row.get("duration_ms") or 0) for row in first])
    last_duration = _avg([int(row.get("duration_ms") or 0) for row in last])
    slow_counts = {
        "30s": sum(1 for value in durations if value > 30_000),
        "40s": sum(1 for value in durations if value > 40_000),
        "60s": sum(1 for value in durations if value > 60_000),
        "90s": sum(1 for value in durations if value > 90_000),
        "120s": sum(1 for value in durations if value > 120_000),
    }

    lines = [
        "# Benchmark Report",
        "",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Base URL: `{meta.get('base_url')}`",
        f"- Session ID: `{meta.get('session_id')}`",
        f"- Questions: `{n}`",
        f"- Dataset: `{meta.get('csv') or 'not attached'}`",
        f"- Avg score: `{_avg(scores):.1f}`",
        f"- Avg heuristic score: `{_avg(heuristic_scores):.1f}`",
        f"- First quartile score: `{first_score:.1f}`",
        f"- Last quartile score: `{last_score:.1f}`",
        f"- Quality drop: `{quality_drop:.1f}`",
        f"- Avg duration: `{_avg(durations) / 1000:.1f}s`",
        f"- Avg LLM duration: `{_avg(llm_durations) / 1000:.1f}s`",
        f"- Avg non-LLM duration: `{_avg(non_llm_durations) / 1000:.1f}s`",
        f"- First quartile duration: `{first_duration / 1000:.1f}s`",
        f"- Last quartile duration: `{last_duration / 1000:.1f}s`",
        f"- Duration growth: `{(last_duration - first_duration) / 1000:.1f}s`",
        f"- Slow turns: `>30s={slow_counts['30s']}, >40s={slow_counts['40s']}, >60s={slow_counts['60s']}, >90s={slow_counts['90s']}, >120s={slow_counts['120s']}`",
        f"- Total latency penalty: `{sum(latency_penalties)}`",
        f"- Judge enabled: `{bool(meta.get('judge_enabled'))}`",
        f"- Avg judge score: `{_avg(judge_scores):.1f}`" if judge_scores else "- Avg judge score: `n/a`",
        f"- Total tool errors: `{sum(tool_errors)}`",
        "",
        "| Turn | Score | Heuristic | Latency penalty | Judge | Time | Tools | Errors | Artifacts | Issues | Question |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        issues = ", ".join(row.get("quality_issues") or [])
        question = str(row.get("question") or "").replace("|", "\\|")
        judge = "n/a" if row.get("judge_score") is None else str(row.get("judge_score"))
        lines.append(
            "| {turn} | {score} | {heuristic} | {latency} | {judge} | {time:.1f}s | {tools} | {errors} | {arts} | {issues} | {question} |".format(
                turn=row.get("turn_index"),
                score=row.get("quality_score"),
                heuristic=row.get("heuristic_score"),
                latency=row.get("latency_penalty"),
                judge=judge,
                time=int(row.get("duration_ms") or 0) / 1000,
                tools=row.get("tool_calls"),
                errors=row.get("tool_errors"),
                arts=row.get("artifact_count"),
                issues=issues.replace("|", "\\|"),
                question=_trim(question, 120),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _html_text(text: Any) -> str:
    return html.escape(str(text or "")).replace("\n", "<br>")


def _markdown_inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
        escaped,
    )
    return escaped


def _is_markdown_table(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    header = lines[index].strip()
    separator = lines[index + 1].strip()
    return "|" in header and re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", separator) is not None


def _split_table_row(line: str) -> list[str]:
    clean = line.strip().strip("|")
    return [cell.strip() for cell in clean.split("|")]


def _basic_markdown_html(text: str) -> str:
    lines = text.splitlines()
    parts: list[str] = []
    paragraph: list[str] = []
    idx = 0

    def flush_paragraph() -> None:
        if paragraph:
            parts.append("<p>" + "<br>".join(_markdown_inline(line) for line in paragraph) + "</p>")
            paragraph.clear()

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            idx += 1
            continue

        fence_match = re.match(r"^```(\w+)?\s*$", stripped)
        if fence_match:
            flush_paragraph()
            idx += 1
            code_lines: list[str] = []
            while idx < len(lines) and not lines[idx].strip().startswith("```"):
                code_lines.append(lines[idx])
                idx += 1
            if idx < len(lines):
                idx += 1
            parts.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
            continue

        if _is_markdown_table(lines, idx):
            flush_paragraph()
            headers = _split_table_row(lines[idx])
            idx += 2
            rows: list[list[str]] = []
            while idx < len(lines) and "|" in lines[idx].strip() and lines[idx].strip():
                rows.append(_split_table_row(lines[idx]))
                idx += 1
            thead = "".join(f"<th>{_markdown_inline(cell)}</th>" for cell in headers)
            tbody = "".join(
                "<tr>" + "".join(f"<td>{_markdown_inline(cell)}</td>" for cell in row) + "</tr>"
                for row in rows
            )
            parts.append(f"<div class='table-wrap'><table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table></div>")
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            parts.append(f"<h{level}>{_markdown_inline(heading_match.group(2))}</h{level}>")
            idx += 1
            continue

        if re.match(r"^[-*]\s+", stripped):
            flush_paragraph()
            items: list[str] = []
            while idx < len(lines):
                item_match = re.match(r"^[-*]\s+(.+)$", lines[idx].strip())
                if not item_match:
                    break
                items.append("<li>" + _markdown_inline(item_match.group(1)) + "</li>")
                idx += 1
            parts.append("<ul>" + "".join(items) + "</ul>")
            continue

        if re.match(r"^\d+[.)]\s+", stripped):
            flush_paragraph()
            items = []
            while idx < len(lines):
                item_match = re.match(r"^\d+[.)]\s+(.+)$", lines[idx].strip())
                if not item_match:
                    break
                items.append("<li>" + _markdown_inline(item_match.group(1)) + "</li>")
                idx += 1
            parts.append("<ol>" + "".join(items) + "</ol>")
            continue

        paragraph.append(line)
        idx += 1

    flush_paragraph()
    return "\n".join(parts)


def _markdown_html(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if markdown2 is None:
        return _basic_markdown_html(raw)
    try:
        rendered = markdown2.markdown(raw, extras=MARKDOWN_EXTRAS)
    except Exception:
        return _html_text(raw)
    rendered = re.sub(r"<script\b[^>]*>.*?</script>", "", str(rendered), flags=re.IGNORECASE | re.DOTALL)
    rendered = re.sub(r"\s+on[a-zA-Z]+\s*=\s*(['\"]).*?\1", "", rendered, flags=re.DOTALL)
    rendered = re.sub(r"\s+on[a-zA-Z]+\s*=\s*[^\s>]+", "", rendered)
    rendered = re.sub(r"(href|src)\s*=\s*(['\"])\s*javascript:.*?\2", r'\1="#"', rendered, flags=re.IGNORECASE | re.DOTALL)
    return rendered.strip()


def _json_pre(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _artifact_preview_html(artifact: dict[str, Any], index: int) -> str:
    artifact_type = str(artifact.get("type") or "unknown")
    title = html.escape(str(artifact.get("text") or artifact_type))
    data = artifact.get("data") if isinstance(artifact.get("data"), dict) else {}
    fmt = str(data.get("format") or "")
    raw = data.get("data")

    header = f"<div class='artifact-title'>#{index} {html.escape(artifact_type)}: {title}</div>"
    if artifact_type == "table" and isinstance(raw, dict):
        columns = raw.get("columns") if isinstance(raw.get("columns"), list) else []
        rows = raw.get("data") if isinstance(raw.get("data"), list) else []
        thead = "".join(f"<th>{html.escape(str(col))}</th>" for col in columns[:12])
        body_rows = []
        for row in rows[:12]:
            cells = row if isinstance(row, list) else []
            body_rows.append(
                "<tr>"
                + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells[:12])
                + "</tr>"
            )
        note = f"<div class='muted'>format={html.escape(fmt)}, preview rows={len(rows)}</div>"
        return (
            header
            + note
            + f"<div class='table-wrap'><table><thead><tr>{thead}</tr></thead>"
            + f"<tbody>{''.join(body_rows)}</tbody></table></div>"
        )

    if artifact_type == "plot" and isinstance(raw, dict):
        div_id = f"plot_{index}_{abs(hash(json.dumps(raw, sort_keys=True, default=str))) % 10_000_000}"
        payload = json.dumps(raw, ensure_ascii=False, default=str)
        traces = raw.get("data")
        trace_count = len(traces) if isinstance(traces, list) else 0
        return (
            header
            + f"<div class='muted'>plotly traces={trace_count}</div>"
            + f"<div id='{div_id}' class='plot'></div>"
            + "<script>"
            + f"renderPlot('{div_id}', {payload});"
            + "</script>"
        )

    if artifact_type == "value" and isinstance(raw, dict):
        items = "".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in list(raw.items())[:30]
        )
        return header + f"<table class='kv'>{items}</table>"

    preview = json.dumps(raw, ensure_ascii=False, indent=2, default=str) if raw is not None else ""
    return header + f"<pre>{html.escape(_trim(preview, 3000))}</pre>"


def _write_chat_html(path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    score_values = [int(row.get("quality_score") or 0) for row in rows]
    durations = [int(row.get("duration_ms") or 0) for row in rows]
    llm_durations = [int(row.get("llm_duration_ms") or 0) for row in rows]
    non_llm_durations = [int(row.get("non_llm_duration_ms") or 0) for row in rows]
    first = rows[: max(1, len(rows) // 4)] if rows else []
    last = rows[-max(1, len(rows) // 4):] if rows else []
    first_score = _avg([int(row.get("quality_score") or 0) for row in first])
    last_score = _avg([int(row.get("quality_score") or 0) for row in last])
    first_duration = _avg([int(row.get("duration_ms") or 0) for row in first])
    last_duration = _avg([int(row.get("duration_ms") or 0) for row in last])
    slow_30 = sum(1 for value in durations if value > 30_000)
    slow_40 = sum(1 for value in durations if value > 40_000)

    parts = [
        "<!doctype html>",
        "<html lang='ru'><head><meta charset='utf-8'>",
        "<title>LLM Data Analyst Benchmark Chat</title>",
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>",
        "<script>",
        "function renderPlot(id, fig){",
        "  const el = document.getElementById(id);",
        "  if (!el || !window.Plotly) return;",
        "  const data = Array.isArray(fig.data) ? fig.data : [];",
        "  const layout = Object.assign({height: 420, margin: {t: 48, r: 24, b: 60, l: 60}}, fig.layout || {});",
        "  Plotly.newPlot(el, data, layout, {responsive: true, displaylogo: false});",
        "}",
        "</script>",
        "<style>",
        """
        :root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }
        body { margin: 0; background: #f5f7fb; color: #172033; }
        header { position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid #d9e0ea; padding: 18px 28px; }
        h1 { margin: 0 0 8px; font-size: 22px; }
        .summary { display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px; color: #526071; }
        .pill { background: #eef2f7; border: 1px solid #d9e0ea; border-radius: 999px; padding: 5px 10px; }
        main { max-width: 1180px; margin: 0 auto; padding: 22px; }
        .turn { background: #fff; border: 1px solid #dce3ed; border-radius: 8px; margin: 0 0 18px; overflow: hidden; box-shadow: 0 6px 20px rgba(20,32,52,.06); }
        .turn-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; padding: 12px 16px; background: #fbfcfe; border-bottom: 1px solid #e5ebf3; }
        .score-good { color: #137a3a; } .score-mid { color: #946200; } .score-bad { color: #b42318; }
        .message { padding: 16px; border-bottom: 1px solid #edf1f6; }
        .run-summary { background: #fff; border: 1px solid #dce3ed; border-radius: 8px; margin: 0 0 18px; padding: 16px; box-shadow: 0 6px 20px rgba(20,32,52,.06); }
        .role { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #6b7788; margin-bottom: 6px; font-weight: 700; }
        .question { background: #f7fbff; }
        .answer { line-height: 1.5; }
        .markdown-body { line-height: 1.55; overflow-wrap: anywhere; }
        .markdown-body > :first-child { margin-top: 0; }
        .markdown-body > :last-child { margin-bottom: 0; }
        .markdown-body p, .markdown-body ul, .markdown-body ol, .markdown-body blockquote, .markdown-body pre, .markdown-body table { margin: 0 0 10px; }
        .markdown-body ul, .markdown-body ol { padding-left: 22px; }
        .markdown-body li + li { margin-top: 4px; }
        .markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 { margin: 12px 0 8px; line-height: 1.25; }
        .markdown-body h1 { font-size: 22px; }
        .markdown-body h2 { font-size: 19px; }
        .markdown-body h3 { font-size: 16px; }
        .markdown-body h4 { font-size: 14px; }
        .markdown-body code { border: 1px solid #d9e0ea; border-radius: 5px; background: #eef2f7; padding: 1px 4px; font-family: Consolas, Menlo, monospace; font-size: .92em; }
        .markdown-body pre code { border: 0; background: transparent; padding: 0; }
        .markdown-body blockquote { border-left: 3px solid #98a7c2; margin-left: 0; padding: 7px 10px; background: #f6f8fb; color: #475467; }
        .tools, .artifacts { padding: 14px 16px; border-bottom: 1px solid #edf1f6; }
        .judge { padding: 14px 16px; border-bottom: 1px solid #edf1f6; background: #fffdf7; }
        .tool { display: grid; grid-template-columns: 170px 90px 100px 1fr; gap: 10px; align-items: start; padding: 8px 0; border-bottom: 1px dashed #e2e8f0; font-size: 13px; }
        .tool:last-child { border-bottom: 0; }
        .tool-error { color: #b42318; font-weight: 700; }
        details { margin-top: 8px; }
        summary { cursor: pointer; color: #344054; font-weight: 700; }
        .muted { color: #667085; font-size: 12px; }
        .issues { color: #b42318; font-size: 13px; }
        .artifact { border: 1px solid #e1e7f0; border-radius: 8px; padding: 12px; margin: 10px 0; background: #fcfdff; }
        .artifact-title { font-weight: 700; margin-bottom: 6px; }
        .table-wrap { overflow: auto; max-height: 420px; border: 1px solid #e2e8f0; border-radius: 6px; }
        table { border-collapse: collapse; width: 100%; font-size: 12px; }
        th, td { border-bottom: 1px solid #e8edf4; padding: 6px 8px; text-align: left; vertical-align: top; }
        th { background: #f3f6fa; position: sticky; top: 0; }
        .kv th { width: 240px; position: static; }
        pre { white-space: pre-wrap; word-break: break-word; background: #f6f8fb; padding: 10px; border-radius: 6px; overflow: auto; max-height: 520px; }
        .raw-api { padding: 14px 16px; background: #fbfcfe; }
        .tool-full { grid-column: 1 / -1; }
        .plot { width: 100%; min-height: 420px; }
        """,
        "</style></head><body>",
        "<header><h1>LLM Data Analyst Benchmark Chat</h1><div class='summary'>",
        f"<span class='pill'>session: {html.escape(str(meta.get('session_id')))}</span>",
        f"<span class='pill'>questions: {len(rows)}</span>",
        f"<span class='pill'>avg score: {_avg(score_values):.1f}</span>",
        f"<span class='pill'>first quartile: {first_score:.1f}</span>",
        f"<span class='pill'>last quartile: {last_score:.1f}</span>",
        f"<span class='pill'>quality drop: {first_score - last_score:.1f}</span>",
        f"<span class='pill'>avg time: {_avg(durations) / 1000:.1f}s</span>",
        f"<span class='pill'>avg LLM: {_avg(llm_durations) / 1000:.1f}s</span>",
        f"<span class='pill'>avg other: {_avg(non_llm_durations) / 1000:.1f}s</span>",
        f"<span class='pill'>time growth: {(last_duration - first_duration) / 1000:.1f}s</span>",
        f"<span class='pill'>slow: &gt;30s={slow_30}, &gt;40s={slow_40}</span>",
        f"<span class='pill'>judge: {html.escape(str(bool(meta.get('judge_enabled'))))}</span>",
        f"<span class='pill'>dataset: {html.escape(str(meta.get('csv') or 'not attached'))}</span>",
        "</div></header><main>",
    ]

    summary_text = str(meta.get("llm_summary") or "").strip()
    summary_error = str(meta.get("llm_summary_error") or "").strip()
    if summary_text or summary_error:
        parts.append("<section class='run-summary'>")
        parts.append("<div class='role'>llm run summary</div>")
        if summary_text:
            parts.append(f"<div class='markdown-body'>{_markdown_html(summary_text)}</div>")
        if summary_error:
            parts.append(f"<div class='tool-error'>{_html_text(summary_error)}</div>")
        parts.append("</section>")

    for row in rows:
        score = int(row.get("quality_score") or 0)
        score_class = "score-good" if score >= 80 else ("score-mid" if score >= 55 else "score-bad")
        issues = ", ".join(row.get("quality_issues") or [])
        parts.extend([
            "<section class='turn'>",
            "<div class='turn-head'>",
            f"<div><strong>Turn {row.get('turn_index')}</strong> <span class='{score_class}'>score={score}</span>"
            f" <span class='muted'>heuristic={row.get('heuristic_score')}, latency_penalty={row.get('latency_penalty')}</span></div>",
            "<div class='muted'>"
            f"time={int(row.get('duration_ms') or 0) / 1000:.1f}s | "
            f"tools={row.get('tool_calls')} | errors={row.get('tool_errors')} | "
            f"artifacts={row.get('artifact_count')} | context_chars_before={row.get('approx_context_chars_before')}"
            "</div>",
            "</div>",
            f"<div class='message question'><div class='role'>user</div>{_html_text(row.get('question'))}</div>",
            f"<div class='message answer'><div class='role'>assistant</div><div class='markdown-body'>{_markdown_html(row.get('answer_text'))}</div></div>",
        ])
        if issues:
            parts.append(
                f"<div class='message'><div class='role'>quality flags</div>"
                f"<div class='issues'>{html.escape(issues)}</div></div>"
            )

        if row.get("judge_enabled"):
            judge_score = "n/a" if row.get("judge_score") is None else str(row.get("judge_score"))
            judge_pass = "n/a" if row.get("judge_pass") is None else str(row.get("judge_pass"))
            judge_issues = ", ".join(row.get("judge_issues") or [])
            judge_missing = ", ".join(row.get("judge_missing") or [])
            parts.append(
                "<div class='judge'><div class='role'>llm judge</div>"
                f"<div><strong>score={html.escape(judge_score)}</strong> pass={html.escape(judge_pass)} "
                f"time={int(row.get('judge_duration_ms') or 0) / 1000:.1f}s</div>"
                f"<div class='issues'>{html.escape(judge_issues)}</div>"
                f"<div class='issues'>{html.escape(judge_missing)}</div>"
                f"<div class='markdown-body'>{_markdown_html(row.get('judge_comment'))}</div>"
                f"<div class='tool-error'>{_html_text(row.get('judge_error'))}</div>"
                "<details><summary>Expectation</summary>"
                f"<pre>{_json_pre(row.get('expectation') or {})}</pre></details>"
                "<details><summary>Parsed judge result</summary>"
                f"<pre>{_json_pre(row.get('judge_parsed') or {})}</pre></details>"
                "<details><summary>Raw judge response</summary>"
                f"<pre>{_json_pre(row.get('judge_raw') or {})}</pre></details>"
                "</div>"
            )

        tool_runs = row.get("tool_runs") if isinstance(row.get("tool_runs"), list) else []
        parts.append("<div class='tools'><div class='role'>tools</div>")
        if tool_runs:
            for tool in tool_runs:
                if not isinstance(tool, dict):
                    continue
                status = str(tool.get("status") or "")
                status_html = (
                    f"<span class='tool-error'>{html.escape(status)}</span>"
                    if status == "error"
                    else html.escape(status)
                )
                parts.append(
                    "<div class='tool'>"
                    f"<div><strong>{html.escape(str(tool.get('tool_name') or 'unknown'))}</strong></div>"
                    f"<div>{status_html}</div>"
                    f"<div>{html.escape(str(tool.get('duration_ms') or ''))} ms</div>"
                    f"<div><div>{_html_text(tool.get('input_preview') or tool.get('input_summary'))}</div>"
                    f"<div class='muted'>{_html_text(tool.get('result_summary') or tool.get('error') or tool.get('output_preview'))}</div></div>"
                    "<div class='tool-full'>"
                    "<details><summary>Tool input_summary</summary>"
                    f"<pre>{html.escape(str(tool.get('input_summary') or ''))}</pre></details>"
                    "<details><summary>Tool input_preview</summary>"
                    f"<pre>{html.escape(str(tool.get('input_preview') or ''))}</pre></details>"
                    "<details><summary>Tool output / error</summary>"
                    f"<pre>{html.escape(str(tool.get('error') or tool.get('output_preview') or tool.get('result_summary') or ''))}</pre></details>"
                    "<details><summary>Raw tool_start payload</summary>"
                    f"<pre>{_json_pre(tool.get('start_payload') or {})}</pre></details>"
                    "<details><summary>Raw tool_end payload</summary>"
                    f"<pre>{_json_pre(tool.get('end_payload') or {})}</pre></details>"
                    "</div>"
                    "</div>"
                )
        else:
            parts.append("<div class='muted'>No tool events captured.</div>")
        parts.append("</div>")

        final_payload = row.get("final_payload") if isinstance(row.get("final_payload"), dict) else {}
        artifacts = final_payload.get("artifacts") if isinstance(final_payload.get("artifacts"), list) else []
        parts.append("<div class='artifacts'><div class='role'>artifacts</div>")
        if artifacts:
            for idx, artifact in enumerate(artifacts, start=1):
                if isinstance(artifact, dict):
                    parts.append("<div class='artifact'>" + _artifact_preview_html(artifact, idx) + "</div>")
        else:
            parts.append("<div class='muted'>No artifacts in final payload.</div>")
        parts.append("</div>")

        api_events = row.get("api_events") if isinstance(row.get("api_events"), list) else []
        parts.append("<div class='raw-api'><div class='role'>raw api</div>")
        parts.append(
            "<details><summary>Final payload</summary>"
            f"<pre>{_json_pre(final_payload)}</pre></details>"
        )
        parts.append(
            f"<details><summary>All SSE events ({len(api_events)})</summary>"
            f"<pre>{_json_pre(api_events)}</pre></details>"
        )
        parts.append("</div></section>")

    parts.append("</main></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _print_progress(row: dict[str, Any]) -> None:
    status = "OK" if row.get("ok") else "FAIL"
    judge = "n/a" if row.get("judge_score") is None else str(row.get("judge_score"))
    print(
        "[{turn:>3}] {status:<4} score={score:>3} heuristic={heuristic:>3} "
        "latency=-{latency:<2} judge={judge:<3} time={time:>6.1f}s "
        "llm={llm:>6.1f}s other={other:>6.1f}s "
        "tools={tools:<2} errors={errors:<2} arts={arts:<2} {question}".format(
            turn=row["turn_index"],
            status=status,
            score=row["quality_score"],
            heuristic=row.get("heuristic_score") or row.get("quality_score"),
            latency=row.get("latency_penalty") or 0,
            judge=judge,
            time=int(row.get("duration_ms") or 0) / 1000,
            llm=int(row.get("llm_duration_ms") or 0) / 1000,
            other=int(row.get("non_llm_duration_ms") or 0) / 1000,
            tools=row.get("tool_calls") or 0,
            errors=row.get("tool_errors") or 0,
            arts=row.get("artifact_count") or 0,
            question=_trim(row.get("question"), 90),
        ),
        flush=True,
    )


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    env_file = _read_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(
        description="Run questions sequentially in one LLM Data Analyst chat and collect metrics.",
    )
    parser.add_argument("--base-url", default=os.getenv("LDA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--user", default=os.getenv("LDA_USER", "admin"))
    parser.add_argument("--password", default=os.getenv("LDA_PASSWORD", "admin"))
    parser.add_argument(
        "--questions",
        required=True,
        type=Path,
        help="TXT/MD with one question per line, or YAML/JSON with cases and expected answers.",
    )
    parser.add_argument("--csv", type=Path, help="Optional CSV dataset to upload before the run.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("backend") / "benchmark" / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--depth", choices=["light", "medium", "deep"])
    parser.add_argument("--skill", action="append", dest="skills", help="Selected skill id; can be repeated.")
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--no-history", action="store_true", help="Still one session, but sends use_history=false.")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N questions.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run the whole suite in this many isolated fresh sessions.",
    )
    parser.add_argument("--timeout", type=int, default=360, help="Per-question HTTP timeout in seconds.")
    parser.add_argument("--chat-temperature", type=float, help="Temporarily set analyst chat temperature.")
    parser.add_argument("--tool-temperature", type=float, help="Temporarily set analyst tool temperature.")
    parser.add_argument(
        "--keep-temperature-settings",
        action="store_true",
        help="Do not restore previous analyst temperatures after the run.",
    )
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-a-judge after each answer.")
    parser.add_argument(
        "--judge-base-url",
        default=(
            _env_value("JUDGE_LLM_BASE_URL", env_file)
            or _env_value("LLM_MODEL_API_URL", env_file)
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=(
            _env_value("JUDGE_LLM_MODEL", env_file)
            or _env_value("LLM_MODEL_NAME", env_file)
        ),
    )
    parser.add_argument(
        "--judge-api-key",
        default=(
            _env_value("JUDGE_LLM_API_KEY", env_file)
            or _env_value("LLM_API_KEY", env_file, "EMPTY")
        ),
    )
    parser.add_argument("--judge-timeout", type=int, default=120, help="Judge HTTP timeout in seconds.")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    args = parser.parse_args()

    if args.repeats > 1:
        exit_code = 0
        for repeat_index in range(1, args.repeats + 1):
            repeat_out = args.out / f"repeat_{repeat_index}"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--repeats",
                "1",
                "--out",
                str(repeat_out),
            ]
            print(f"Starting isolated repeat {repeat_index}/{args.repeats}: {repeat_out}")
            completed = subprocess.run(command, check=False)
            exit_code = max(exit_code, completed.returncode)
        return exit_code
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    base_url = str(args.base_url).rstrip("/")
    if not args.questions.exists():
        print(f"Questions file not found: {args.questions}", file=sys.stderr)
        return 2
    if args.csv and not args.csv.exists():
        print(f"CSV file not found: {args.csv}", file=sys.stderr)
        return 2

    cases = load_cases(args.questions)
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        print(f"No questions found in {args.questions}", file=sys.stderr)
        return 2

    judge_enabled = bool(args.judge or any(case.get("expectation") for case in cases))
    judge_config = JudgeConfig(
        enabled=judge_enabled,
        base_url=str(args.judge_base_url or "").rstrip("/"),
        model=str(args.judge_model or ""),
        api_key=str(args.judge_api_key or "EMPTY"),
        timeout=int(args.judge_timeout),
        temperature=float(args.judge_temperature),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Backend: {base_url}")
    print(f"Questions: {len(cases)} from {args.questions}")
    print(f"Output: {args.out}")
    if judge_enabled:
        print(f"Judge: {judge_config.model or '<missing model>'} @ {judge_config.base_url or '<missing url>'}")

    auth = _request_json(
        "POST",
        f"{base_url}/auth/login",
        body={"username": args.user, "password": args.password},
        timeout=60,
    )
    token = str(auth.get("access_token") or "")
    if not token:
        raise RuntimeError("Login response did not include access_token")

    previous_temperature_settings: dict[str, Any] = {}
    temperature_patch: dict[str, Any] = {}
    if args.chat_temperature is not None:
        temperature_patch["llm_temperature_chat"] = float(args.chat_temperature)
    if args.tool_temperature is not None:
        temperature_patch["llm_temperature_tool"] = float(args.tool_temperature)
    if temperature_patch:
        current_settings = _get_auth_settings(base_url, token)
        previous_temperature_settings = {
            key: current_settings.get(key)
            for key in ("llm_temperature_chat", "llm_temperature_tool")
            if key in temperature_patch
        }
        _patch_auth_settings(base_url, token, temperature_patch)
        print(
            "Patched analyst temperatures: "
            + ", ".join(f"{key}={value}" for key, value in temperature_patch.items())
        )

        def _restore_temperatures() -> None:
            if args.keep_temperature_settings or not previous_temperature_settings:
                return
            try:
                _patch_auth_settings(base_url, token, previous_temperature_settings)
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup on CLI exit.
                print(f"WARNING: failed to restore analyst temperatures: {exc}", file=sys.stderr)

        atexit.register(_restore_temperatures)

    session = _request_json("POST", f"{base_url}/sessions", token=token, timeout=60)
    session_id = str(session.get("session_id") or "")
    if not session_id:
        raise RuntimeError("Create session response did not include session_id")
    print(f"Session: {session_id}")

    upload_info: dict[str, Any] | None = None
    if args.csv:
        upload_info = _upload_csv(f"{base_url}/sessions/{session_id}/data", token, args.csv)
        print(
            "Uploaded CSV: {name} ({rows} rows x {cols} columns)".format(
                name=args.csv.name,
                rows=upload_info.get("rows", "?"),
                cols=upload_info.get("columns", "?"),
            )
        )

    rows: list[dict[str, Any]] = []
    history_answer_chars = 0
    history_question_chars = 0
    started_at = time.perf_counter()

    def meta_snapshot(elapsed_ms: int | None = None) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "base_url": base_url,
            "session_id": session_id,
            "questions": str(args.questions),
            "csv": str(args.csv) if args.csv else None,
            "upload_info": upload_info,
            "depth": args.depth,
            "skills": args.skills or [],
            "use_history": not args.no_history,
            "include_reasoning": args.include_reasoning,
            "judge_enabled": judge_enabled,
            "judge_base_url": judge_config.base_url,
            "judge_model": judge_config.model,
            "temperature_patch": temperature_patch,
            "previous_temperature_settings": previous_temperature_settings,
            "keep_temperature_settings": bool(args.keep_temperature_settings),
        }
        if elapsed_ms is not None:
            meta["elapsed_ms"] = elapsed_ms
        return meta

    for idx, case in enumerate(cases, start=1):
        question = str(case.get("question") or "")
        expectation = case.get("expectation") if isinstance(case.get("expectation"), dict) else {}
        context_before = history_question_chars + history_answer_chars
        result = stream_query(
            base_url,
            token,
            session_id,
            question=question,
            use_history=not args.no_history,
            include_reasoning=args.include_reasoning,
            analysis_depth=args.depth,
            selected_skill_ids=args.skills,
            timeout=args.timeout,
        )
        row = {
            "run_started_at": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "turn_index": idx,
            "case_id": case.get("id"),
            "question": question,
            "history_question_count": idx - 1,
            "history_question_chars_before": history_question_chars,
            "history_answer_chars_before": history_answer_chars,
            "approx_context_chars_before": context_before,
            **result,
        }
        if judge_enabled:
            judge_result = run_judge(judge_config, question=question, row=row, expectation=expectation)
            row.update(judge_result)
            _apply_judge_score(row)
        else:
            row.update(run_judge(JudgeConfig(enabled=False), question=question, row=row, expectation={}))
        rows.append(row)
        history_question_chars += len(question)
        history_answer_chars += int(row.get("answer_chars") or 0)
        _print_progress(row)

        _write_jsonl(args.out / "results.jsonl", rows)
        _write_csv(args.out / "summary.csv", rows)
        _write_chat_html(
            args.out / "chat.html",
            rows,
            meta_snapshot(),
        )

    elapsed = int((time.perf_counter() - started_at) * 1000)
    meta = meta_snapshot(elapsed)
    if judge_enabled:
        summary_result = run_llm_summary(judge_config, rows)
        meta["llm_summary"] = summary_result.get("summary_text") or ""
        meta["llm_summary_error"] = summary_result.get("summary_error") or ""
        meta["llm_summary_duration_ms"] = summary_result.get("summary_duration_ms")
        meta["llm_summary_raw"] = summary_result.get("summary_raw")
    else:
        meta["llm_summary"] = ""
        meta["llm_summary_error"] = "LLM summary skipped because judge is disabled."
    (args.out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(args.out / "report.md", rows, meta)
    _write_chat_html(args.out / "chat.html", rows, meta)

    print("")
    print(f"Done in {elapsed / 1000:.1f}s")
    print(f"JSONL: {args.out / 'results.jsonl'}")
    print(f"CSV:   {args.out / 'summary.csv'}")
    print(f"Report:{args.out / 'report.md'}")
    print(f"HTML:  {args.out / 'chat.html'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001 - CLI top-level should print a compact error.
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
