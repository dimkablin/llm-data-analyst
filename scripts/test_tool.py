#!/usr/bin/env python3
"""
Скрипт для тестирования конкретного тула через API агента.

Примеры:
  # Простой запрос без файла
  python scripts/test_tool.py --query "сделай описательную статистику"

  # Загрузить CSV и протестировать конкретный тул
  python scripts/test_tool.py --file examples/titanic.csv --query "проведи eda"

  # Активировать скил и проверить что он вызывается
  python scripts/test_tool.py --file data.csv --skill auto_eda --query "сделай разведочный анализ"

  # Другой сервер / другие учётные данные
  python scripts/test_tool.py --base-url http://localhost:8605 --user admin --password admin123 --query "..."
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def _request(method: str, url: str, *, token: str | None = None,
             body: dict | None = None, content_type: str = "application/json") -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers: dict[str, str] = {"Content-Type": content_type}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        print(f"[ERROR] {method} {url} → HTTP {exc.code}: {detail}", file=sys.stderr)
        sys.exit(1)


def _upload_file(url: str, token: str, file_path: Path) -> dict:
    """Multipart/form-data upload."""
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    file_bytes = file_path.read_bytes()
    filename = file_path.name

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        print(f"[ERROR] POST {url} → HTTP {exc.code}: {detail}", file=sys.stderr)
        sys.exit(1)


def _stream_query(url: str, token: str, payload: dict) -> None:
    """Читает SSE-поток и печатает события тулов и текст ответа."""
    data = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        print(f"[ERROR] POST {url} → HTTP {exc.code}: {detail}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "═" * 60)
    print("SSE STREAM")
    print("═" * 60)

    buffer = b""
    tool_calls: list[dict] = []

    for raw_line in resp:
        buffer += raw_line
        line = raw_line.decode(errors="replace").rstrip("\n")

        if not line.startswith("data: "):
            continue

        payload_str = line[6:]
        if payload_str.strip() == "[DONE]":
            break

        try:
            event = json.loads(payload_str)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        if etype == "tool_start":
            name = event.get("tool_name", "?")
            preview = event.get("input_preview", "")
            print(f"\n  ▶ TOOL START: {name}")
            if preview:
                print(f"    INPUT:  {_trim(str(preview), 120)}")
            tool_calls.append({"name": name, "status": "started"})

        elif etype == "tool_end":
            name = event.get("tool_name", "?")
            preview = event.get("output_preview", "")
            error = event.get("error")
            status = "✗ ERROR" if error else "✓ OK"
            print(f"  ◀ TOOL END:   {name}  [{status}]")
            if preview:
                print(f"    OUTPUT: {_trim(str(preview), 180)}")
            if error:
                print(f"    ERROR:  {error}")
            for t in tool_calls:
                if t["name"] == name and t["status"] == "started":
                    t["status"] = "error" if error else "done"
                    break

        elif etype == "token":
            text = event.get("text", "")
            print(text, end="", flush=True)

        elif etype == "error":
            print(f"\n[STREAM ERROR] {event.get('message', event)}", file=sys.stderr)
            break

    print("\n\n" + "═" * 60)
    print("ИТОГО ТУЛОВ:")
    if tool_calls:
        for t in tool_calls:
            icon = "✓" if t["status"] == "done" else ("✗" if t["status"] == "error" else "?")
            print(f"  {icon} {t['name']}")
    else:
        print("  (тулы не вызывались — агент ответил напрямую)")
    print("═" * 60)


def _trim(s: str, n: int) -> str:
    return s[:n] + "…" if len(s) > n else s


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Тестирование тулов агента через REST API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base-url", default="http://localhost:8605",
                        help="Базовый URL бэкенда (default: http://localhost:8605)")
    parser.add_argument("--user", default="admin", help="Имя пользователя (default: admin)")
    parser.add_argument("--password", default="admin", help="Пароль (default: admin)")
    parser.add_argument("--query", required=True, help="Запрос к агенту")
    parser.add_argument("--file", metavar="CSV_PATH",
                        help="CSV-файл для загрузки в сессию перед запросом")
    parser.add_argument("--skill", metavar="SKILL_ID", action="append", dest="skills",
                        help="ID скила для активации (можно указать несколько раз)")
    parser.add_argument("--depth", choices=["light", "medium", "deep"],
                        help="Глубина анализа (default: из настроек)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Использовать синхронный endpoint вместо SSE")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    # 1. Login
    print(f"[1/4] Авторизация как '{args.user}'...")
    auth = _request("POST", f"{base}/auth/login",
                    body={"username": args.user, "password": args.password})
    token = auth["access_token"]
    print(f"      ✓ token получен")

    # 2. Create session
    print("[2/4] Создание сессии...")
    session = _request("POST", f"{base}/sessions", token=token)
    session_id = session["session_id"]
    print(f"      ✓ session_id: {session_id}")

    # 3. Upload file (optional)
    if args.file:
        csv_path = Path(args.file)
        if not csv_path.exists():
            print(f"[ERROR] Файл не найден: {csv_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[3/4] Загрузка файла '{csv_path.name}'...")
        upload_resp = _upload_file(f"{base}/sessions/{session_id}/data", token, csv_path)
        rows = upload_resp.get("rows", "?")
        cols = upload_resp.get("columns", "?")
        print(f"      ✓ загружено {rows} строк × {cols} колонок")
    else:
        print("[3/4] Файл не указан — работаем без датасета")

    # 4. Send query
    payload: dict = {
        "query": args.query,
        "use_history": False,
        "include_reasoning": False,
    }
    if args.skills:
        payload["selected_skill_ids"] = args.skills
        print(f"[4/4] Запрос (скилы: {args.skills}): {_trim(args.query, 80)}")
    else:
        print(f"[4/4] Запрос: {_trim(args.query, 80)}")

    if args.depth:
        payload["analysis_depth"] = args.depth

    if args.no_stream:
        print("      → синхронный режим...")
        resp = _request("POST", f"{base}/sessions/{session_id}/query", token=token, body=payload)
        print("\n" + "═" * 60)
        print("ОТВЕТ:")
        print(resp.get("answer", resp))
        print("═" * 60)
        tools_used = resp.get("tools_used") or []
        if tools_used:
            print("ТУЛЫ:", ", ".join(tools_used))
    else:
        _stream_query(f"{base}/sessions/{session_id}/query/stream", token, payload)


if __name__ == "__main__":
    main()
