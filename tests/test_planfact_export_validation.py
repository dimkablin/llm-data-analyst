from pathlib import Path

import pandas as pd

from backend.api.routes import reports
from backend.auth.auth_db import AuthUser
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.notebook.manifest_store import ManifestStore
from backend.services.planfact_export_validation import parse_validation_sql
from backend.sessions.session_store import SessionStore


class _OwnedSessionAuth:
    @staticmethod
    def is_session_owner(session_id: str, user_id: int) -> bool:
        return bool(session_id) and user_id == 1


def test_planfact_export_builds_excel_friendly_validation_tables(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path / "legacy"), ttl_days=7)
    state = store.create_session()
    runtime = CSVSessionRuntime(base_dir=tmp_path / "duckdb", default_ttl_sec=3600)
    info = runtime.register_dataframes(
        session_id=state.session_id,
        tables={
            "planfact_by_cfo_period": pd.DataFrame(
                {
                    "cfo": ["A", "A", "B"],
                    "period": ["2026-01", "2026-02", "2026-01"],
                    "plan_amount": [100.0, 120.0, 500.0],
                    "fact_amount": [110.0, 130.0, 500.0],
                    "variance_amount": [10.0, 10.0, 0.0],
                    "variance_pct": [0.1, 10.0 / 120.0, 0.0],
                    "plan_source_row_ids": ["1", "2", "3"],
                    "fact_source_row_ids": ["1", "2", "3"],
                }
            ),
            "planfact_by_cfo_article_period": pd.DataFrame(
                {
                    "cfo": ["A"],
                    "article": ["Аренда"],
                    "period": ["2026-01"],
                    "plan_amount": [100.0],
                    "fact_amount": [110.0],
                    "variance_amount": [10.0],
                    "variance_pct": [0.1],
                    "plan_source_row_ids": ["1"],
                    "fact_source_row_ids": ["1"],
                }
            ),
            "planfact_plan_raw": pd.DataFrame(
                {
                    "plan_source_row_id": [1, 2, 3],
                    "ЦФО": ["A", "A", "B"],
                    "План": [100.0, 120.0, 500.0],
                }
            ),
            "planfact_fact_raw": pd.DataFrame(
                {
                    "fact_source_row_id": [1, 2, 3],
                    "ЦФО": ["A", "A", "B"],
                    "Сумма": [110.0, 130.0, 500.0],
                }
            ),
        },
    )
    store.set_source(
        state.session_id,
        source_type="planfact",
        source_ref_id="planfact",
        source_label="План-факт",
        source_mode="duckdb",
    )
    store.set_csv_runtime_state(
        state.session_id,
        csv_loaded=True,
        csv_session_id=info.session_id,
        csv_table_names=info.table_names,
        csv_expires_at=info.expires_at,
    )
    reports.setup(
        auth_db=_OwnedSessionAuth(),  # type: ignore[arg-type]
        store=store,
        csv_runtime=runtime,
        manifest_store=ManifestStore(tmp_path),
    )

    artifact = {
        "type": "table",
        "data": {
            "format": "split",
            "data": {
                "columns": ["cfo", "avg_variance"],
                "data": [["A", 10.0]],
            },
        },
        "meta": {
            "recipe": [
                {
                    "kind": "sql",
                    "code": (
                        "SELECT cfo, AVG(variance_amount) AS avg_variance "
                        "FROM planfact_by_cfo_period "
                        "WHERE period BETWEEN '2026-01' AND '2026-02' "
                        "GROUP BY cfo"
                    ),
                }
            ]
        },
    }
    tables = reports._planfact_validation_tables(
        state.session_id,
        AuthUser(id=1, username="user", is_admin=False, created_at="now"),
        [artifact],
    )

    assert tables["Контроль"]["rows"][0][:4] == [1, "A", "avg_variance", 10.0]
    assert tables["Контроль"]["rows"][0][4].startswith("=AVERAGEIF")
    assert tables["Контроль"]["rows"][0][5] == '=IF(OR(E2="",D2=""),"",E2-D2)'
    assert tables["Контроль"]["rows"][0][6] == (
        '=IF(F2="","НЕ ПРОВЕРЕНО",IF(ABS(F2)<=0.01,"OK","РАСХОЖДЕНИЕ"))'
    )
    assert len(tables["Расчетная детализация"]["rows"]) == 2
    assert {row[0] for row in tables["Первичка план"]["rows"]} == {1}
    assert len(tables["Первичка план"]["rows"]) == 2
    assert {row[0] for row in tables["Первичка факт"]["rows"]} == {1}
    assert len(tables["Первичка факт"]["rows"]) == 2

    direct_artifact = {
        "type": "table",
        "data": {
            "format": "split",
            "data": {
                "columns": [
                    "cfo",
                    "period",
                    "plan_amount",
                    "fact_amount",
                    "variance_amount",
                    "variance_pct",
                ],
                "data": [["A", "2026-01", 100.0, 110.0, 10.0, 0.1]],
            },
        },
        "meta": {
            "query": {
                "requested_sql": (
                    "SELECT cfo, period, plan_amount, fact_amount, variance_amount, variance_pct "
                    "FROM planfact_by_cfo_period "
                    "WHERE ABS(variance_amount) >= 10 "
                    "ORDER BY ABS(variance_amount) DESC"
                )
            }
        },
    }
    direct = reports._planfact_validation_tables(
        state.session_id,
        AuthUser(id=1, username="user", is_admin=False, created_at="now"),
        [direct_artifact],
    )

    assert list(direct) == [
        "Контроль",
        "Расчетная детализация",
        "Первичка план",
        "Первичка факт",
    ]
    assert {row[1] for row in direct["Контроль"]["rows"]} == {
        "plan_amount",
        "fact_amount",
        "variance_amount",
        "variance_pct",
    }
    assert all(row[3].startswith("=IFERROR(AVERAGEIF") for row in direct["Контроль"]["rows"])
    assert len(direct["Расчетная детализация"]["rows"]) == 1
    assert direct["Первичка план"]["rows"][0][0] == 1
    assert direct["Первичка факт"]["rows"][0][0] == 1

    article_artifact = {
        "type": "table",
        "data": {
            "format": "split",
            "data": {
                "columns": ["cfo", "article", "plan_amount", "fact_amount"],
                "data": [["A", "Аренда", 100.0, 110.0]],
            },
        },
        "meta": {
            "recipe": [
                {
                    "kind": "sql",
                    "code": (
                        "SELECT cfo, article, plan_amount, fact_amount "
                        "FROM planfact_by_cfo_article_period WHERE period = '2026-01'"
                    ),
                }
            ]
        },
    }
    article = reports._planfact_validation_tables(
        state.session_id,
        AuthUser(id=1, username="user", is_admin=False, created_at="now"),
        [article_artifact],
    )
    assert len(article["Расчетная детализация"]["rows"]) == 1
    assert article["Первичка план"]["rows"][0][0] == 1
    assert article["Первичка факт"]["rows"][0][0] == 1

    raw_aggregate_artifact = {
        "type": "table",
        "data": {
            "format": "split",
            "data": {
                "columns": ["cfo", "total"],
                "data": [["A", 240.0]],
            },
        },
        "meta": {
            "recipe": [
                {
                    "kind": "sql",
                    "code": (
                        'SELECT "ЦФО" AS cfo, SUM("Сумма") AS total '
                        'FROM planfact_fact_raw GROUP BY "ЦФО"'
                    ),
                }
            ]
        },
    }
    raw_aggregate = reports._planfact_validation_tables(
        state.session_id,
        AuthUser(id=1, username="user", is_admin=False, created_at="now"),
        [raw_aggregate_artifact],
    )
    assert raw_aggregate["Контроль"]["rows"][0][3] == 240.0
    assert len(raw_aggregate["Первичка факт"]["rows"]) == 2

    assert parse_validation_sql(
        "SELECT SUM(fact_amount) AS total FROM planfact_focus_by_cfo_period",
        ["total"],
    ) is not None
    assert parse_validation_sql(
        "SELECT DISTINCT cfo FROM planfact_focus_by_cfo_article_period",
        ["cfo"],
    ) is not None

    unsupported = reports._planfact_validation_tables(
        state.session_id,
        AuthUser(id=1, username="user", is_admin=False, created_at="now"),
        [
            {
                **artifact,
                "meta": {
                    "recipe": [
                        {
                            "kind": "sql",
                            "code": (
                                "SELECT a.cfo, SUM(a.plan_amount) "
                                "FROM planfact_plan_long a "
                                "JOIN planfact_fact_monthly b ON a.cfo = b.cfo "
                                "GROUP BY a.cfo"
                            ),
                        }
                    ]
                },
            }
        ],
    )
    assert list(unsupported) == ["Проверка недоступна"]
