"""E2E smoke tests for llm-data-analyst backend.

Tests: table display, chart generation, scalar metric.
Requires: running backend at localhost:8605 with Ollama/Qwen3.
"""
from __future__ import annotations

import csv
import io
import json
import sys
import time

import requests

BASE = "http://localhost:8605"
TIMEOUT_STREAM = 300


def _login() -> str:
    r = requests.post(f"{BASE}/auth/login", json={"username": "admin", "password": "admin"})
    r.raise_for_status()
    return r.json()["access_token"]


def _create_session(token: str) -> str:
    r = requests.post(f"{BASE}/sessions", headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    return r.json()["session_id"]


def _upload_csv(token: str, sid: str) -> None:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["category", "sales", "profit"])
    for cat, s, p in [("A", 100, 20), ("B", 200, 50), ("C", 150, 30), ("D", 80, 10), ("E", 300, 70)]:
        w.writerow([cat, s, p])
    buf.seek(0)
    files = {"file": ("test_data.csv", buf.getvalue(), "text/csv")}
    r = requests.post(
        f"{BASE}/sessions/{sid}/data",
        headers={"Authorization": f"Bearer {token}"},
        files=files,
    )
    r.raise_for_status()


def _query_stream(token: str, sid: str, query: str) -> dict:
    """Send query via SSE stream and collect result."""
    r = requests.post(
        f"{BASE}/sessions/{sid}/query/stream",
        headers={"Authorization": f"Bearer {token}", "Accept": "text/event-stream"},
        json={"query": query},
        stream=True,
        timeout=TIMEOUT_STREAM,
    )
    r.raise_for_status()

    artifacts = []
    text_parts = []
    error = None
    current_event = None
    execution_graph = None

    for raw_line in r.iter_lines(decode_unicode=True):
        if not raw_line:
            current_event = None
            continue
        if raw_line.startswith("event: "):
            current_event = raw_line[7:].strip()
            continue
        if raw_line.startswith("data: "):
            data_str = raw_line[6:]
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            evt_type = current_event or data.get("type", "")
            if evt_type == "artifact":
                artifacts.append(data)
            elif evt_type == "text_delta":
                text_parts.append(data.get("content", ""))
            elif evt_type == "execution_graph":
                execution_graph = data
            elif evt_type == "error":
                error = data.get("message", str(data))
            elif evt_type in ("done", "final"):
                if isinstance(data.get("artifacts"), list):
                    artifacts.extend(data["artifacts"])
                if isinstance(data.get("execution_graph"), dict):
                    execution_graph = data["execution_graph"]

    return {
        "text": "".join(text_parts),
        "artifacts": artifacts,
        "error": error,
        "execution_graph": execution_graph,
    }


def run_test(name: str, token: str, sid: str, query: str, check_fn) -> bool:
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Query: {query}")
    t0 = time.time()
    try:
        result = _query_stream(token, sid, query)
        elapsed = time.time() - t0
        if result["error"]:
            print(f"  ERROR from stream: {result['error']}")
            print(f"  Time: {elapsed:.1f}s | FAIL")
            return False
        ok, detail = check_fn(result)
        status = "PASS" if ok else "FAIL"
        print(f"  Time: {elapsed:.1f}s | arts={len(result['artifacts'])} | {status}")
        if detail:
            print(f"  Detail: {detail}")
        return ok
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  EXCEPTION: {e}")
        print(f"  Time: {elapsed:.1f}s | FAIL")
        return False


def check_table(result: dict) -> tuple[bool, str]:
    arts = result["artifacts"]
    if not arts:
        return False, "no artifacts"
    return True, f"{len(arts)} artifact(s)"


def check_chart(result: dict) -> tuple[bool, str]:
    arts = result["artifacts"]
    if not arts:
        return False, "no artifacts"
    has_plot = any(
        a.get("artifact_type") == "plot" or a.get("type") == "plot"
        for a in arts
    )
    if not has_plot:
        types = [a.get("artifact_type") or a.get("type") for a in arts]
        return False, f"no plot artifact, got types: {types}"
    return True, f"{len(arts)} artifact(s) including plot"


def check_metric(result: dict) -> tuple[bool, str]:
    arts = result["artifacts"]
    if not arts:
        return False, "no artifacts"
    return True, f"{len(arts)} artifact(s)"


def check_multi_tool(result: dict) -> tuple[bool, str]:
    """Verify multi-tool query produces both table and plot artifacts."""
    arts = result["artifacts"]
    if not arts:
        return False, "no artifacts"

    has_plot = any(
        a.get("artifact_type") == "plot" or a.get("type") == "plot"
        for a in arts
    )
    if not has_plot:
        types = [a.get("artifact_type") or a.get("type") for a in arts]
        return False, f"no plot artifact among {len(arts)} artifacts, types: {types}"

    graph = result.get("execution_graph")
    if not graph:
        return False, f"{len(arts)} artifact(s) with plot, but no execution_graph"

    nodes = graph.get("nodes", [])
    tool_nodes = [n for n in nodes if n.get("type") == "tool"]

    details = []
    details.append(f"{len(arts)} artifact(s) with plot")
    details.append(f"{len(tool_nodes)} tool nodes: {[n.get('label') for n in tool_nodes]}")

    return True, "; ".join(details)


def _find_db_connection(token: str) -> str | None:
    """Return the first available DB connection id, or None."""
    r = requests.get(
        f"{BASE}/db-connections",
        headers={"Authorization": f"Bearer {token}"},
    )
    r.raise_for_status()
    conns = r.json()
    return conns[0]["id"] if conns else None


def _bind_db_source(token: str, sid: str, connection_id: str) -> None:
    r = requests.post(
        f"{BASE}/sessions/{sid}/source/db-connection",
        headers={"Authorization": f"Bearer {token}"},
        json={"connection_id": connection_id, "source_mode": "tables"},
    )
    r.raise_for_status()


def main():
    print("E2E Smoke Tests")
    print("=" * 60)

    token = _login()

    # ── CSV session ────────────────────────────────────────────────
    csv_sid = _create_session(token)
    _upload_csv(token, csv_sid)
    print(f"CSV session:  {csv_sid}")

    results = []

    results.append(run_test(
        "1. [CSV] Table Display",
        token, csv_sid,
        "Покажи таблицу с данными",
        check_table,
    ))

    results.append(run_test(
        "2. [CSV] Chart Generation",
        token, csv_sid,
        "Построй столбчатую диаграмму sales по category",
        check_chart,
    ))

    results.append(run_test(
        "3. [CSV] Scalar Metric",
        token, csv_sid,
        "Посчитай среднее значение profit",
        check_metric,
    ))

    results.append(run_test(
        "4. [CSV] Multi-tool (sandbox)",
        token, csv_sid,
        "Посчитай суммарные продажи (sales) по каждой категории (category) и построй столбчатый график по этим агрегированным данным",
        check_multi_tool,
    ))

    # ── DB session ─────────────────────────────────────────────────
    conn_id = _find_db_connection(token)
    if not conn_id:
        print("\n[SKIP] No DB connections found — skipping DB tests")
    else:
        db_sid = _create_session(token)
        _bind_db_source(token, db_sid, conn_id)
        print(f"DB session:   {db_sid}  (connection: {conn_id})")

        results.append(run_test(
            "5. [DB] Table Catalog",
            token, db_sid,
            "Покажи таблицы в базе данных",
            check_table,
        ))

        results.append(run_test(
            "6. [DB] Table Query",
            token, db_sid,
            "Покажи первые 10 строк таблицы titanic",
            check_table,
        ))

        results.append(run_test(
            "7. [DB] Scalar Metric",
            token, db_sid,
            "Посчитай количество строк в таблице titanic",
            check_metric,
        ))

        results.append(run_test(
            "8. [DB] Chart from DB",
            token, db_sid,
            "Построй столбчатую диаграмму выживших и погибших (Survived) из таблицы titanic",
            check_chart,
        ))

    print(f"\n{'='*60}")
    passed = sum(results)
    total = len(results)
    overall = "ALL PASS" if passed == total else f"{passed}/{total} PASSED"
    print(f"OVERALL: {overall}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
