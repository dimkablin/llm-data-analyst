from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any, ClassVar
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import urlopen

from backend.core.config import Settings
from backend.observability.models import (
    PhoenixLatencyPoint,
    PhoenixOverviewResponse,
    PhoenixOverviewStats,
    PhoenixTokenUsageRow,
    PhoenixTraceRow,
)


@dataclass(frozen=True)
class PhoenixSpanSnapshot:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    span_kind: str
    start_time: datetime
    end_time: datetime
    status_code: str
    status_message: str
    attributes: dict[str, Any]

    @property
    def duration_ms(self) -> int:
        return max(0, int((self.end_time - self.start_time).total_seconds() * 1000))

    @property
    def session_id(self) -> str | None:
        value = self.attributes.get("session.id") or self.attributes.get("metadata.session_id")
        if value is None:
            return None
        clean = str(value).strip()
        return clean or None

    @property
    def request_kind(self) -> str:
        value = str(self.attributes.get("metadata.request_kind", "")).strip().lower()
        return value or "unknown"

    @property
    def query_preview(self) -> str:
        value = str(self.attributes.get("metadata.query_preview", "")).strip()
        if value:
            return value
        input_value = self._load_json_blob(self.attributes.get("input.value"))
        if isinstance(input_value, dict):
            prompt = str(input_value.get("prompt", "")).strip()
            if prompt:
                return prompt
        return "Запрос без preview"

    @property
    def user(self) -> str | None:
        value = self.attributes.get("metadata.username")
        if value is None:
            return None
        clean = str(value).strip()
        return clean or None

    @property
    def model(self) -> str | None:
        candidates = (
            self.attributes.get("llm.model_name"),
            self.attributes.get("metadata.ls_model_name"),
            self.attributes.get("metadata.model"),
        )
        for candidate in candidates:
            clean = str(candidate or "").strip()
            if clean:
                return clean
        output_value = self._load_json_blob(self.attributes.get("output.value"))
        if isinstance(output_value, dict):
            generations = output_value.get("generations")
            if isinstance(generations, list):
                for group in generations:
                    if not isinstance(group, list):
                        continue
                    for item in group:
                        if not isinstance(item, dict):
                            continue
                        info = item.get("generation_info")
                        if isinstance(info, dict):
                            clean = str(info.get("model_name", "")).strip()
                            if clean:
                                return clean
        return None

    @property
    def token_usage(self) -> tuple[int | None, int | None, int | None, str]:
        input_tokens = self._extract_token_value(
            "llm.token_count.prompt",
            "llm.usage.prompt_tokens",
            "usage.prompt_tokens",
            "prompt_tokens",
        )
        output_tokens = self._extract_token_value(
            "llm.token_count.completion",
            "llm.usage.completion_tokens",
            "usage.completion_tokens",
            "completion_tokens",
        )
        total_tokens = self._extract_token_value(
            "llm.token_count.total",
            "llm.usage.total_tokens",
            "usage.total_tokens",
            "total_tokens",
        )
        source = "unavailable"
        if any(value is not None for value in (input_tokens, output_tokens, total_tokens)):
            source = "provider"
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return input_tokens, output_tokens, total_tokens, source

    def _extract_token_value(self, *keys: str) -> int | None:
        for key in keys:
            direct = self.attributes.get(key)
            parsed = self._to_int(direct)
            if parsed is not None:
                return parsed

        for blob_key in ("output.value", "input.value"):
            blob = self._load_json_blob(self.attributes.get(blob_key))
            parsed = self._find_nested_token_value(blob, keys)
            if parsed is not None:
                return parsed
        return None

    @classmethod
    def _load_json_blob(cls, raw: Any) -> Any:
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        if not text or text[0] not in "[{":
            return None
        try:
            return json.loads(text)
        except Exception:
            return None

    @classmethod
    def _find_nested_token_value(cls, node: Any, keys: tuple[str, ...]) -> int | None:
        wanted = {key.split(".")[-1] for key in keys}
        if isinstance(node, dict):
            for key, value in node.items():
                if key in wanted:
                    parsed = cls._to_int(value)
                    if parsed is not None:
                        return parsed
                nested = cls._find_nested_token_value(value, keys)
                if nested is not None:
                    return nested
        elif isinstance(node, list):
            for item in node:
                nested = cls._find_nested_token_value(item, keys)
                if nested is not None:
                    return nested
        return None

    @staticmethod
    def _to_int(raw: Any) -> int | None:
        if raw is None or raw == "":
            return None
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            return int(raw)
        text = str(raw).strip()
        if not text:
            return None
        try:
            return int(float(text))
        except Exception:
            return None


@dataclass(frozen=True)
class PhoenixRunSnapshot:
    trace_id: str
    session_id: str | None
    request_kind: str
    query_preview: str
    user: str | None
    status: str
    duration_ms: int
    tool_calls: int
    span_count: int
    started_at: datetime
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    token_source: str


class PhoenixObservabilityService:
    REQUEST_KINDS: ClassVar[set[str]] = {"query", "stream", "evaluate"}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        parsed = urlparse(settings.phoenix_collector_endpoint)
        collector_origin = urlunparse(
            parsed._replace(path="", params="", query="", fragment="")
        ).rstrip("/")
        public_origin = f"http://{settings.phoenix_host}:{settings.phoenix_ui_port}"
        origins = [collector_origin, public_origin]
        self._api_origins: list[str] = []
        for origin in origins:
            clean = str(origin).strip().rstrip("/")
            if clean and clean not in self._api_origins:
                self._api_origins.append(clean)

    def build_overview(self) -> PhoenixOverviewResponse:
        generated_at = datetime.now(UTC).isoformat()
        dashboard_url = self._build_public_dashboard_url()
        empty = PhoenixOverviewResponse(
            available=False,
            project_name=self.settings.phoenix_project_name,
            generated_at=generated_at,
            dashboard_url=dashboard_url,
            embed_url=dashboard_url,
            stats=PhoenixOverviewStats(
                total_traces=0,
                success_rate=0.0,
                p50_latency_ms=0,
                unique_sessions=0,
            ),
            latency=[],
            token_usage=[],
            traces=[],
            warnings=[],
        )

        try:
            project = self._resolve_project()
            if project is None:
                return empty.model_copy(
                    update={
                        "warnings": [
                            f"Phoenix project '{self.settings.phoenix_project_name}' not found.",
                        ]
                    }
                )

            spans = self._fetch_spans(project["id"])
            runs = self._build_runs(spans)
            warnings: list[str] = []
            if runs and not any(run.total_tokens is not None for run in runs):
                warnings.append(
                    "LLM provider does not expose token usage in current Phoenix spans yet."
                )

            return PhoenixOverviewResponse(
                available=True,
                project_name=self.settings.phoenix_project_name,
                project_id=project["id"],
                generated_at=generated_at,
                dashboard_url=dashboard_url,
                embed_url=dashboard_url,
                stats=self._build_stats(runs),
                latency=self._build_latency(runs),
                token_usage=self._build_token_rows(runs),
                traces=self._build_trace_rows(runs),
                warnings=warnings,
            )
        except Exception as exc:
            return empty.model_copy(
                update={"warnings": [f"Phoenix API unavailable: {exc}"]}
            )

    def _resolve_project(self) -> dict[str, str] | None:
        payload = self._api_get_json("/v1/projects")
        data = payload.get("data")
        if not isinstance(data, list):
            return None
        wanted = self.settings.phoenix_project_name.strip().lower()
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            identifier = str(item.get("id", "")).strip()
            if name.lower() == wanted and identifier:
                return {"id": identifier, "name": name}
        return None

    def _fetch_spans(self, project_id: str) -> list[PhoenixSpanSnapshot]:
        today = datetime.now(UTC)
        all_items: list[Any] = []
        for day_offset in range(7):
            day_start = (today - timedelta(days=day_offset + 1)).isoformat()
            day_end = (today - timedelta(days=day_offset)).isoformat()
            try:
                payload = self._api_get_json(
                    f"/v1/projects/{project_id}/spans",
                    {"limit": 500, "start_time": day_start, "end_time": day_end},
                )
                data = payload.get("data")
                if isinstance(data, list):
                    all_items.extend(data)
            except Exception:
                pass

        spans: list[PhoenixSpanSnapshot] = []
        for item in all_items:
            if not isinstance(item, dict):
                continue
            context = item.get("context") if isinstance(item.get("context"), dict) else {}
            trace_id = str(context.get("trace_id", "")).strip()
            span_id = str(context.get("span_id", "")).strip()
            start_raw = str(item.get("start_time", "")).strip()
            end_raw = str(item.get("end_time", "")).strip()
            if not (trace_id and span_id and start_raw and end_raw):
                continue
            try:
                start_time = datetime.fromisoformat(start_raw)
                end_time = datetime.fromisoformat(end_raw)
            except Exception:
                continue
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=UTC)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=UTC)
            spans.append(
                PhoenixSpanSnapshot(
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_id=self._normalize_optional_text(item.get("parent_id")),
                    name=str(item.get("name", "")).strip() or "span",
                    span_kind=str(item.get("span_kind", "")).strip().upper(),
                    start_time=start_time,
                    end_time=end_time,
                    status_code=str(item.get("status_code", "")).strip().upper() or "UNSET",
                    status_message=str(item.get("status_message", "")).strip(),
                    attributes=item.get("attributes")
                    if isinstance(item.get("attributes"), dict)
                    else {},
                )
            )
        spans.sort(key=lambda span: span.start_time, reverse=True)
        return spans

    @staticmethod
    def _normalize_optional_text(raw: Any) -> str | None:
        if raw is None:
            return None
        clean = str(raw).strip()
        if not clean or clean.lower() in {"none", "null"}:
            return None
        return clean

    def _build_runs(self, spans: list[PhoenixSpanSnapshot]) -> list[PhoenixRunSnapshot]:
        trace_groups: dict[str, list[PhoenixSpanSnapshot]] = {}
        llm_spans: list[PhoenixSpanSnapshot] = []
        for span in spans:
            trace_groups.setdefault(span.trace_id, []).append(span)
            if span.span_kind == "LLM":
                llm_spans.append(span)

        runs: list[PhoenixRunSnapshot] = []
        for trace_id, group in trace_groups.items():
            root = self._pick_root_request_span(group)
            if root is None:
                continue

            tool_calls = sum(1 for span in group if span.span_kind == "TOOL")
            matched_llms = self._match_llm_spans(root, llm_spans)
            model = self._pick_model(matched_llms) or self._pick_model(group)
            input_tokens, output_tokens, total_tokens, token_source = self._aggregate_tokens(
                matched_llms
            )
            if token_source == "unavailable":
                input_tokens, output_tokens, total_tokens, token_source = self._aggregate_tokens(
                    [root]
                )
            if token_source == "unavailable":
                input_tokens, output_tokens, total_tokens, token_source = self._aggregate_tokens(
                    group
                )

            runs.append(
                PhoenixRunSnapshot(
                    trace_id=trace_id,
                    session_id=root.session_id,
                    request_kind=root.request_kind,
                    query_preview=root.query_preview,
                    user=root.user,
                    status=self._normalize_status(root, group),
                    duration_ms=root.duration_ms,
                    tool_calls=tool_calls,
                    span_count=len(group),
                    started_at=root.start_time,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    token_source=token_source,
                )
            )

        runs.sort(key=lambda run: run.started_at, reverse=True)
        return runs

    def _pick_root_request_span(
        self, spans: list[PhoenixSpanSnapshot]
    ) -> PhoenixSpanSnapshot | None:
        candidates = [
            span
            for span in spans
            if span.parent_id is None
            and span.span_kind != "LLM"
            and span.request_kind in self.REQUEST_KINDS
            and span.session_id
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda span: span.start_time)
        return candidates[0]

    def _match_llm_spans(
        self,
        root: PhoenixSpanSnapshot,
        llm_spans: list[PhoenixSpanSnapshot],
    ) -> list[PhoenixSpanSnapshot]:
        matched: list[PhoenixSpanSnapshot] = []
        lower_bound = root.start_time - timedelta(seconds=3)
        upper_bound = root.end_time + timedelta(seconds=3)
        for span in llm_spans:
            if span.session_id != root.session_id:
                continue
            if span.user != root.user:
                continue
            if span.query_preview != root.query_preview:
                continue
            if not (lower_bound <= span.start_time <= upper_bound):
                continue
            matched.append(span)
        matched.sort(key=lambda span: span.start_time)
        return matched

    @staticmethod
    def _pick_model(spans: list[PhoenixSpanSnapshot]) -> str | None:
        counts: dict[str, int] = {}
        for span in spans:
            model = span.model
            if not model:
                continue
            counts[model] = counts.get(model, 0) + 1
        if not counts:
            return None
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    @staticmethod
    def _aggregate_tokens(
        spans: list[PhoenixSpanSnapshot],
    ) -> tuple[int | None, int | None, int | None, str]:
        input_total = 0
        output_total = 0
        total_total = 0
        has_input = False
        has_output = False
        has_total = False
        source = "unavailable"

        for span in spans:
            input_tokens, output_tokens, total_tokens, span_source = span.token_usage
            if span_source != "unavailable":
                source = span_source
            if input_tokens is not None:
                input_total += input_tokens
                has_input = True
            if output_tokens is not None:
                output_total += output_tokens
                has_output = True
            if total_tokens is not None:
                total_total += total_tokens
                has_total = True

        input_value = input_total if has_input else None
        output_value = output_total if has_output else None
        total_value = total_total if has_total else None
        if total_value is None and input_value is not None and output_value is not None:
            total_value = input_value + output_value
        return input_value, output_value, total_value, source

    @staticmethod
    def _normalize_status(
        root: PhoenixSpanSnapshot, group: list[PhoenixSpanSnapshot]
    ) -> str:
        if root.status_code and root.status_code != "OK":
            return "error"
        if any(span.status_code and span.status_code != "OK" for span in group):
            return "error"
        return "success"

    @staticmethod
    def _percentile(values: list[int], percentile: float) -> int:
        if not values:
            return 0
        if len(values) == 1:
            return values[0]
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
        return ordered[index]

    def _build_stats(self, runs: list[PhoenixRunSnapshot]) -> PhoenixOverviewStats:
        durations = [run.duration_ms for run in runs]
        success_count = sum(1 for run in runs if run.status == "success")
        session_ids = {run.session_id for run in runs if run.session_id}
        success_rate = (success_count / len(runs) * 100.0) if runs else 0.0
        return PhoenixOverviewStats(
            total_traces=len(runs),
            success_rate=round(success_rate, 1),
            p50_latency_ms=int(median(durations)) if durations else 0,
            unique_sessions=len(session_ids),
        )

    def _build_latency(self, runs: list[PhoenixRunSnapshot]) -> list[PhoenixLatencyPoint]:
        buckets: dict[str, list[int]] = {}
        for run in runs:
            key = run.started_at.astimezone().strftime("%Y-%m-%d")
            buckets.setdefault(key, []).append(run.duration_ms)

        today = datetime.now(UTC).astimezone()
        points: list[PhoenixLatencyPoint] = []
        for day_offset in range(6, -1, -1):
            day = today - timedelta(days=day_offset)
            key = day.strftime("%Y-%m-%d")
            label = day.strftime("%d %b")
            values = sorted(buckets.get(key, []))
            points.append(
                PhoenixLatencyPoint(
                    label=label,
                    p50_ms=self._percentile(values, 0.50),
                    p95_ms=self._percentile(values, 0.95),
                    p99_ms=self._percentile(values, 0.99),
                    trace_count=len(values),
                )
            )
        return points

    @staticmethod
    def _build_token_rows(runs: list[PhoenixRunSnapshot]) -> list[PhoenixTokenUsageRow]:
        rows: list[PhoenixTokenUsageRow] = []
        for run in runs:
            rows.append(  # noqa: PERF401
                PhoenixTokenUsageRow(
                    trace_id=run.trace_id,
                    session_id=run.session_id,
                    query_preview=run.query_preview,
                    model=run.model,
                    input_tokens=run.input_tokens,
                    output_tokens=run.output_tokens,
                    total_tokens=run.total_tokens,
                    duration_ms=run.duration_ms,
                    started_at=run.started_at.isoformat(),
                    token_source=run.token_source,
                )
            )
        return rows

    @staticmethod
    def _build_trace_rows(runs: list[PhoenixRunSnapshot]) -> list[PhoenixTraceRow]:
        rows: list[PhoenixTraceRow] = []
        for run in runs[:10]:
            rows.append(  # noqa: PERF401
                PhoenixTraceRow(
                    trace_id=run.trace_id,
                    session_id=run.session_id,
                    query_preview=run.query_preview,
                    request_kind=run.request_kind,
                    user=run.user,
                    status=run.status,
                    duration_ms=run.duration_ms,
                    tool_calls=run.tool_calls,
                    span_count=run.span_count,
                    model=run.model,
                    input_tokens=run.input_tokens,
                    output_tokens=run.output_tokens,
                    total_tokens=run.total_tokens,
                    started_at=run.started_at.isoformat(),
                )
            )
        return rows

    def _build_public_dashboard_url(self) -> str:
        root_path = self.settings.phoenix_host_root_path.strip() or "/phoenix"
        root_path = f"/{root_path.lstrip('/')}"
        return root_path

    def _api_get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = urlencode(
            {key: value for key, value in (params or {}).items() if value is not None}
        )
        last_error: Exception | None = None
        for origin in self._api_origins:
            url = f"{origin}{path}"
            if query:
                url = f"{url}?{query}"
            try:
                with urlopen(url, timeout=5) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    payload = response.read().decode(charset)
                data = json.loads(payload)
                if not isinstance(data, dict):
                    raise ValueError("Phoenix API returned a non-object payload")
                return data
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("No Phoenix API origins configured")


