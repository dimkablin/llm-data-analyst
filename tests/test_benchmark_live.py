from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN_BENCHMARK = os.getenv("LIVE_BENCHMARK_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}

BENCHMARK_CASES = [
    (
        "sportmaster",
        ROOT / "backend" / "benchmark" / "data" / "sportmaster" / "dataset.csv",
        ROOT / "backend" / "benchmark" / "data" / "sportmaster" / "questions.yaml",
    ),
    (
        "investment",
        ROOT / "backend" / "benchmark" / "data" / "investment" / "dataset.csv",
        ROOT / "backend" / "benchmark" / "data" / "investment" / "questions.yaml",
    ),
]

pytestmark = [
    pytest.mark.live,
    pytest.mark.e2e,
    pytest.mark.benchmark,
    pytest.mark.skipif(
        not RUN_BENCHMARK,
        reason="Set LIVE_BENCHMARK_TESTS=1 to run live benchmark tests.",
    ),
]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _benchmark_args(name: str, csv_path: Path, questions_path: Path, out_dir: Path) -> list[str]:
    args = [
        sys.executable,
        str(ROOT / "backend" / "benchmark" / "benchmark_chat.py"),
        "--base-url",
        os.getenv("LDA_BASE_URL", "http://localhost:8605"),
        "--user",
        os.getenv("LDA_USER", "admin"),
        "--password",
        os.getenv("LDA_PASSWORD", "admin"),
        "--csv",
        str(csv_path),
        "--questions",
        str(questions_path),
        "--out",
        str(out_dir),
        "--timeout",
        str(_env_int("BENCHMARK_HTTP_TIMEOUT_SEC", 360)),
        "--judge-timeout",
        str(_env_int("BENCHMARK_JUDGE_TIMEOUT_SEC", 120)),
    ]
    limit = _env_int(f"BENCHMARK_{name.upper()}_LIMIT", _env_int("BENCHMARK_LIMIT", 0))
    if limit > 0:
        args.extend(["--limit", str(limit)])
    return args


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _failure_summary(row: dict[str, Any]) -> str:
    issues = ", ".join(str(item) for item in row.get("quality_issues") or [])
    judge_issues = ", ".join(str(item) for item in row.get("judge_issues") or [])
    return (
        f"{row.get('case_id')}: score={row.get('quality_score')} "
        f"judge={row.get('judge_score')} issues=[{issues}] judge_issues=[{judge_issues}]"
    )


@pytest.mark.parametrize(("name", "csv_path", "questions_path"), BENCHMARK_CASES)
def test_live_benchmark_dataset(name: str, csv_path: Path, questions_path: Path) -> None:
    out_dir = ROOT / ".runtime" / "pytest_benchmark" / f"{name}_{int(time.time())}"
    command = _benchmark_args(name, csv_path, questions_path, out_dir)

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=_env_int("BENCHMARK_RUN_TIMEOUT_SEC", 3600),
        check=False,
    )

    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    rows = _read_rows(out_dir / "results.jsonl")
    assert rows, f"Benchmark produced no rows: {out_dir}"

    min_case_score = _env_int("BENCHMARK_MIN_CASE_SCORE", 50)
    min_avg_score = _env_int("BENCHMARK_MIN_AVG_SCORE", 70)
    min_judge_score = _env_int("BENCHMARK_MIN_JUDGE_SCORE", 60)
    fatal_issues = {"request_failed", "fallback_used", "required_tool_unavailable"}

    failed_rows = []
    for row in rows:
        quality_issues = set(row.get("quality_issues") or [])
        judge_score = row.get("judge_score")
        if (
            not row.get("ok")
            or row.get("fallback_used")
            or row.get("judge_error")
            or row.get("judge_pass") is False
            or int(row.get("quality_score") or 0) < min_case_score
            or (judge_score is not None and int(judge_score) < min_judge_score)
            or bool(quality_issues & fatal_issues)
        ):
            failed_rows.append(row)

    avg_score = statistics.mean(int(row.get("quality_score") or 0) for row in rows)
    assert avg_score >= min_avg_score, f"{name} avg_score={avg_score:.1f}, report={out_dir / 'chat.html'}"
    assert not failed_rows, "\n".join([f"report={out_dir / 'chat.html'}", *map(_failure_summary, failed_rows)])
