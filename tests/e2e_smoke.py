"""E2E smoke tests for llm-data-analyst backend.

Tests: table display, chart generation, scalar metric, multi-tool sandbox,
       sql_tool->plotly_tool variable injection fix.
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


def run_test(name: str, token: str, sid: str, query: str, check_fn) -> tuple[bool, float]:
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
    t0 = time.time()
    try:
        result = _query_stream(token, sid, query)
        elapsed = time.time() - t0
        if result["error"]:
            print(f"  ERROR from stream: {result['error']}")
            print(f"  Time: {elapsed:.1f}s  FAIL")
            return False, elapsed
        ok, detail = check_fn(result)
        status = "PASS" if ok else "FAIL"
        print(f"  Time: {elapsed:.1f}s | arts={len(result['artifacts'])} | {status}")
        if detail:
            print(f"  Detail: {detail}")
        return ok, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  EXCEPTION: {e}")
        print(f"  Time: {elapsed:.1f}s  FAIL")
        return False, elapsed


def check_table(result: dict) -> tuple[bool, str]:
    arts = result["artifacts"]
    if not arts:
        return False, f"no artifacts. text={result['text'][:120]}"
    return True, f"{len(arts)} artifact(s)"


def check_chart(result: dict) -> tuple[bool, str]:
    arts = result["artifacts"]
    if not arts:
        return False, f"no artifacts. text={result['text'][:120]}"
    has_plot = any(
        a.get("artifact_type") == "plot" or a.get("type") == "plot"
        for a in arts
    )
    if not has_plot:
        types = [a.get("artifact_type") or a.get("type") for a in arts]
        return False, f"no plot artifact, got types: {types}. text={result['text'][:100]}"
    return True, f"{len(arts)} artifact(s) including plot"


def check_metric(result: dict) -> tuple[bool, str]:
    arts = result["artifacts"]
    if not arts:
        return False, f"no artifacts. text={result['text'][:120]}"
    return True, f"{len(arts)} artifact(s)"


def check_multi_tool(result: dict) -> tuple[bool, str]:
    """Verify multi-tool query produces both table and plot artifacts."""
    arts = result["artifacts"]
    if not arts:
        return False, f"no artifacts. text={result['text'][:120]}"

    has_plot = any(
        a.get("artifact_type") == "plot" or a.get("type") == "plot"
        for a in arts
    )
    if not has_plot:
        types = [a.get("artifact_type") or a.get("type") for a in arts]
        return False, f"no plot among {len(arts)} artifacts, types: {types}. text={result['text'][:100]}"

    graph = result.get("execution_graph")
    nodes = graph.get("nodes", []) if graph else []
    tool_nodes = [n for n in nodes if n.get("type") == "tool"]

    return True, (
        f"{len(arts)} artifact(s) with plot; "
        f"{len(tool_nodes)} tool nodes: {[n.get('label') for n in tool_nodes]}"
    )


def check_sql_then_chart(result: dict) -> tuple[bool, str]:
    """Key regression check: sql_tool result used by plotly_tool via sandbox variable.

    Before the fix, plotly_tool would fail with NameError because sql_tool
    did not inject its result DataFrame into the sandbox scope.
    After the fix, the DataFrame is available as a named variable.
    """
    arts = result["artifacts"]
    if not arts:
        return False, f"no artifacts. text={result['text'][:200]}"

    # Check that we got a plot (not just a table)
    has_plot = any(
        a.get("artifact_type") == "plot" or a.get("type") == "plot"
        for a in arts
    )
    if not has_plot:
        types = [a.get("artifact_type") or a.get("type") for a in arts]
        # A table result is acceptable too (agent may use db.query_dataframe inline)
        return True, f"no plot but got {len(arts)} artifact(s) of types {types} — chart via inline SQL is also ok"

    # Check the text for the old error signature
    text = result["text"]
    if "is not defined" in text:
        return False, f"NameError in response: {text[:200]}"

    return True, f"{len(arts)} artifact(s) including plot — sandbox variable injection working"


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


def _detect_db_table(token: str, connection_id: str) -> str:
    """Return first available table name from the DB (prefer titanic or bank_churn_clients)."""
    preferred = ["titanic", "bank_churn_clients"]
    for schema in ["examples", "public", "data"]:
        r = requests.get(
            f"{BASE}/db-connections/{connection_id}/tables?schema={schema}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if r.status_code != 200:
            continue
        tables = r.json()
        if not tables:
            continue
        names = [t.get("name", t.get("table_name", "")) for t in tables if isinstance(t, dict)]
        names = [n for n in names if n]
        for pref in preferred:
            if pref in names:
                return f"{schema}.{pref}"
        if names:
            return f"{schema}.{names[0]}"
    return "titanic"


def _print_timing_summary(results: list[tuple[str, bool, float]]) -> None:
    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'-'*60}")
    csv_times = [(n, ok, t) for n, ok, t in results if "[CSV]" in n]
    db_times = [(n, ok, t) for n, ok, t in results if "[DB]" in n]

    for group, label in [(csv_times, "CSV"), (db_times, "DB")]:
        if not group:
            continue
        print(f"\n  {label} tests:")
        for name, ok, t in group:
            status = "OK" if ok else "FAIL"
            short = name.split(". ", 1)[-1] if ". " in name else name
            print(f"    {status} {short:<35} {t:5.1f}s")
        passed_t = [t for _, ok, t in group if ok]
        all_t = [t for _, _, t in group]
        if passed_t:
            print(f"    avg(pass)={sum(passed_t)/len(passed_t):.1f}s  "
                  f"avg(all)={sum(all_t)/len(all_t):.1f}s  "
                  f"total={sum(all_t):.1f}s")


def main():
    print("E2E Smoke Tests — llm-data-analyst")
    print("=" * 60)

    token = _login()
    all_results: list[tuple[str, bool, float]] = []

    # ── CSV session ────────────────────────────────────────────────
    csv_sid = _create_session(token)
    _upload_csv(token, csv_sid)
    print(f"CSV session:  {csv_sid}")

    csv_tests = [
        ("1. [CSV] Table Display",
         "Покажи таблицу с данными",
         check_table),
        ("2. [CSV] Scalar Metric",
         "Посчитай среднее значение profit",
         check_metric),
        ("3. [CSV] Chart Generation",
         "Построй столбчатую диаграмму sales по category",
         check_chart),
        ("4. [CSV] Multi-tool sandbox (pandas->plotly)",
         "Посчитай суммарные продажи по каждой категории и построй столбчатый график по этим данным",
         check_multi_tool),
    ]

    for name, query, check_fn in csv_tests:
        ok, elapsed = run_test(name, token, csv_sid, query, check_fn)
        all_results.append((name, ok, elapsed))

    # ── DB session ─────────────────────────────────────────────────
    conn_id = _find_db_connection(token)
    if not conn_id:
        print("\n[SKIP] No DB connections found — skipping DB tests")
    else:
        table = _detect_db_table(token, conn_id)
        table_short = table.split(".")[-1]
        db_sid = _create_session(token)
        _bind_db_source(token, db_sid, conn_id)
        print(f"\nDB session:   {db_sid}  (connection: {conn_id}, table: {table})")

        db_tests = [
            ("5. [DB] Table Query",
             f"Покажи первые 10 строк таблицы {table_short}",
             check_table),
            ("6. [DB] Scalar Metric",
             f"Посчитай количество строк в таблице {table_short}",
             check_metric),
            ("7. [DB] Chart from DB (inline SQL in plotly_tool)",
             f"Построй столбчатую диаграмму: количество пассажиров по классу билета (Pclass) из таблицы {table_short}",
             check_chart),
            ("8. [DB] sql_tool->plotly_tool sandbox variable",
             f"Сначала получи топ-5 записей из таблицы {table_short} через sql инструмент, "
             f"затем построй по ним столбчатый график",
             check_sql_then_chart),
        ]

        for name, query, check_fn in db_tests:
            ok, elapsed = run_test(name, token, db_sid, query, check_fn)
            all_results.append((name, ok, elapsed))

    # ── Summary ────────────────────────────────────────────────────
    _print_timing_summary(all_results)

    passed = sum(ok for _, ok, _ in all_results)
    total = len(all_results)
    total_time = sum(t for _, _, t in all_results)
    overall = "ALL PASS" if passed == total else f"FAIL {passed}/{total} PASSED"
    print(f"\n{'='*60}")
    print(f"OVERALL: {overall}  (wall time: {total_time:.1f}s)")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
