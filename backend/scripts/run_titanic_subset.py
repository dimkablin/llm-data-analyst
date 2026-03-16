from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import backend.app as backend_app # noqa: E402


NUMBERED_PROMPT_RE = re.compile(r"^(\d+)\.\s+(.*)$")


def load_numbered_prompts(path: Path) -> dict[int, str]:
    prompts: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = NUMBERED_PROMPT_RE.match(line.strip())
        if match:
            prompts[int(match.group(1))] = match.group(2)
    return prompts


def build_queries(prompts: dict[int, str], scenario: str) -> list[dict[str, Any]]:
    if scenario == "full":
        plot_prompt_ids = {6, 7, 8, 9, 10, 14, 19}
        queries: list[dict[str, Any]] = []
        for prompt_id in sorted(prompts):
            if prompt_id > 21:
                continue
            expected = "plot" if prompt_id in plot_prompt_ids else "non-empty result"
            queries.append(
                {
                    "id": prompt_id,
                    "kind": "full",
                    "expected": expected,
                    "query": prompts[prompt_id],
                }
            )
        if len(queries) != 21:
            raise ValueError(
                f"Expected 21 numbered prompts for full scenario, got {len(queries)}"
            )
        return queries

    if scenario != "subset":
        raise ValueError(f"Unknown scenario: {scenario}")

    queries: list[dict[str, Any]] = []

    for idx in range(1, 6):
        if idx not in prompts:
            raise ValueError(f"Prompt #{idx} not found in prompts file")
        queries.append(
            {
                "id": idx,
                "kind": "first5",
                "expected": "non-empty result",
                "query": prompts[idx],
            }
        )

    queries.append(
        {
            "id": 1001,
            "kind": "table",
            "expected": "table",
            "query": "Построй таблицу выживаемости по полу и классу билета (количество выживших и погибших).",
        }
    )

    plot_prompt_id = 6
    if plot_prompt_id not in prompts:
        raise ValueError(f"Prompt #{plot_prompt_id} not found in prompts file")
    queries.append(
        {
            "id": plot_prompt_id,
            "kind": "plot",
            "expected": "plot",
            "query": prompts[plot_prompt_id],
        }
    )

    return queries


def evaluate_expectation(expected: str, body: dict[str, Any]) -> bool:
    metrics = body.get("metrics") or {}
    text = (body.get("text") or "").strip()

    if expected == "table":
        return int(metrics.get("table_count", 0)) > 0
    if expected == "plot":
        return int(metrics.get("plot_count", 0)) > 0

    return int(metrics.get("artifact_count", 0)) > 0 or bool(text)


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_numbered_prompts(Path(args.prompts_path))
    queries = build_queries(prompts, args.scenario)

    client = TestClient(backend_app.app)

    session_id = client.post("/sessions", timeout=15).json()["session_id"]

    with Path(args.dataset_path).open("rb") as dataset_fh:
        upload_response = client.post(
            f"/sessions/{session_id}/data",
            files={"file": ("dataset.csv", dataset_fh, "text/csv")},
            timeout=30,
        )
    upload_response.raise_for_status()

    print(
        f"session={session_id}, uploaded_rows={upload_response.json().get('rows')}, total_queries={len(queries)}"
    )

    results: list[dict[str, Any]] = []
    started_total = time.time()

    with tqdm(total=len(queries), desc="Titanic subset", unit="query") as progress:
        for query_spec in queries:
            started = time.time()
            result: dict[str, Any] = {
                "id": query_spec["id"],
                "kind": query_spec["kind"],
                "expected": query_spec["expected"],
                "query": query_spec["query"],
            }

            try:
                response = client.post(
                    f"/sessions/{session_id}/query",
                    json={"query": query_spec["query"], "use_history": False},
                    timeout=args.query_timeout,
                )

                result["http_status"] = response.status_code
                result["elapsed_client_ms"] = round((time.time() - started) * 1000)

                if response.status_code != 200:
                    try:
                        result["error"] = response.json()
                    except Exception:
                        result["error"] = {"raw": response.text}
                    result["matches_expectation"] = False
                    results.append(result)
                    progress.set_postfix(
                        {
                            "id": query_spec["id"],
                            "status": response.status_code,
                            "ok": False,
                        }
                    )
                    progress.update(1)
                    continue

                body = response.json()
                result["metrics"] = body.get("metrics")
                result["values"] = body.get("values")
                result["artifact_types"] = [
                    artifact.get("type") for artifact in body.get("artifacts", [])
                ]
                result["text_preview"] = (body.get("text") or "")[:260]
                result["matches_expectation"] = evaluate_expectation(
                    query_spec["expected"], body
                )

                results.append(result)
                progress.set_postfix(
                    {
                        "id": query_spec["id"],
                        "status": 200,
                        "ok": result["matches_expectation"],
                    }
                )
                progress.update(1)
            except Exception as exc:
                result["http_status"] = 0
                result["elapsed_client_ms"] = round((time.time() - started) * 1000)
                result["error"] = {"exception": str(exc)}
                result["matches_expectation"] = False
                results.append(result)
                progress.set_postfix(
                    {
                        "id": query_spec["id"],
                        "status": "exception",
                        "ok": False,
                    }
                )
                progress.update(1)

    summary = {
        "session_id": session_id,
        "total_queries": len(results),
        "http_ok": sum(1 for row in results if row.get("http_status") == 200),
        "expectation_ok": sum(1 for row in results if row.get("matches_expectation")),
        "failed_ids": [row["id"] for row in results if row.get("http_status") != 200],
        "mismatch_ids": [
            row["id"]
            for row in results
            if row.get("http_status") == 200 and not row.get("matches_expectation")
        ],
        "total_wall_time_sec": round(time.time() - started_total, 2),
    }

    payload = {
        "summary": summary,
        "results": results,
    }

    output_path = Path(args.output_path)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output_path}")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run short Titanic benchmark against backend API"
    )
    parser.add_argument(
        "--prompts-path",
        default="examples/titanic/prompts.md",
        help="Path to prompts markdown file",
    )
    parser.add_argument(
        "--dataset-path",
        default="examples/titanic/dataset.csv",
        help="Path to Titanic CSV",
    )
    parser.add_argument(
        "--output-path",
        default="backend_titanic_subset_results.json",
        help="Path to JSON report",
    )
    parser.add_argument(
        "--query-timeout",
        type=float,
        default=150.0,
        help="Timeout per query in seconds",
    )
    parser.add_argument(
        "--scenario",
        choices=["subset", "full"],
        default="subset",
        help="subset: first 5 + table + plot, full: all 21 prompts",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
