from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from backend.auth.blob_store import BlobWrite, StoredBlob
from backend.data_access.csv_runtime_state_service import CSVRuntimeStateService
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.planfact_source_service import PlanfactSourceService, PlanfactUploadFile
from backend.notebook.manifest_store import ManifestStore
from backend.sessions.session_store import SessionStore


def _service(tmp_path: Path) -> tuple[PlanfactSourceService, SessionStore, CSVSessionRuntime, str]:
    store = SessionStore(str(tmp_path / "legacy"), ttl_days=7)
    session = store.create_session()
    csv_runtime = CSVSessionRuntime(base_dir=tmp_path / "duckdb", default_ttl_sec=3600)
    service = PlanfactSourceService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=ManifestStore(tmp_path),
        storage_dir=tmp_path,
    )
    return service, store, csv_runtime, session.session_id


def test_planfact_pending_files_fall_back_to_postgres_blobs(tmp_path: Path) -> None:
    class FakeBlobStore:
        def get_latest_for_session(self, *, session_id: str, kind: str) -> StoredBlob | None:
            del session_id
            return StoredBlob(
                blob_id=kind,
                logical_name=f"{kind}.xlsx",
                media_type="application/octet-stream",
                content=kind.encode(),
            )

    store = SessionStore(str(tmp_path / "legacy"), ttl_days=7)
    session_id = store.create_session().session_id
    service = PlanfactSourceService(
        store=store,
        csv_runtime=CSVSessionRuntime(base_dir=tmp_path / "duckdb", default_ttl_sec=3600),
        manifest_store=ManifestStore(tmp_path),
        storage_dir=tmp_path,
        blob_store=FakeBlobStore(),  # type: ignore[arg-type]
    )

    name, content = service._read_pending_file(session_id, "plan")

    assert name == "planfact_plan.xlsx"
    assert content == b"planfact_plan"


def test_planfact_runtime_restores_from_postgres_snapshots(tmp_path: Path) -> None:
    class FakeBlobStore:
        def __init__(self) -> None:
            self.items: dict[str, tuple[str, str, bytes]] = {}

        def put_many(self, *, kind: str, items: list[BlobWrite], **_kwargs) -> list[str]:
            ids = [f"{kind}-{len(self.items) + index}" for index in range(len(items))]
            for blob_id, item in zip(ids, items, strict=True):
                self.items[blob_id] = (kind, item.logical_name, item.content)
            return ids

        def delete_many(self, *, blob_ids: list[str], **_kwargs) -> None:
            for blob_id in blob_ids:
                self.items.pop(blob_id, None)

        def get_for_session(self, *, blob_id: str, **_kwargs) -> StoredBlob | None:
            item = self.items.get(blob_id)
            if item is None:
                return None
            _kind, name, content = item
            return StoredBlob(
                blob_id=blob_id,
                logical_name=name,
                media_type="application/vnd.apache.parquet",
                content=content,
            )

    store = SessionStore(str(tmp_path / "legacy"), ttl_days=7)
    session_id = store.create_session().session_id
    csv_runtime = CSVSessionRuntime(base_dir=tmp_path / "duckdb", default_ttl_sec=3600)
    manifest_store = ManifestStore(tmp_path)
    blob_store = FakeBlobStore()
    service = PlanfactSourceService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        storage_dir=tmp_path,
        blob_store=blob_store,  # type: ignore[arg-type]
    )
    plan = pd.DataFrame({"CFO": ["A"], "Article": ["Rent"], "Mar": [100]})
    fact = pd.DataFrame(
        {"Date": ["2026-03-01"], "CFO": ["A"], "Article": ["Rent"], "Amount": [110]}
    )
    service._write_pending_file(session_id, "plan", "plan.xlsx", b"plan")
    service._write_pending_file(session_id, "fact", "fact.xlsx", b"fact")
    service._read_dataframe = lambda name, _content: plan if name == "plan.xlsx" else fact  # type: ignore[method-assign]
    service.confirm(
        session_id=session_id,
        user_id=1,
        config={
            "source_type": "planfact",
            "plan": {
                "cfo_column": "CFO",
                "article_column": "Article",
                "monthly_columns": {"2026-03": "Mar"},
            },
            "fact": {
                "date_column": "Date",
                "cfo_column": "CFO",
                "article_column": "Article",
                "amount_column": "Amount",
            },
        },
    )
    csv_runtime.delete_session(session_id)
    shutil.rmtree(tmp_path / "sessions" / session_id / "sources")

    CSVRuntimeStateService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        storage_dir=tmp_path,
        blob_store=blob_store,  # type: ignore[arg-type]
    ).ensure_csv_runtime(session_id=session_id)

    restored = csv_runtime.query_dataframe(
        session_id,
        "SELECT plan_amount, fact_amount FROM planfact_by_cfo_period",
    )
    assert restored.to_dict(orient="records") == [{"plan_amount": 100.0, "fact_amount": 110.0}]


def test_planfact_confirm_builds_derived_tables_with_outer_join(tmp_path: Path) -> None:
    service, store, csv_runtime, session_id = _service(tmp_path)
    plan = pd.DataFrame(
        {
            "ЦФО": ["A", "A", "B"],
            "статья CF": ["CF02080000 Rent", "Travel (11030700)", "Ops"],
            "CF Mar": [100, 50, 70],
            "CF Apr": [200, 60, 80],
        }
    )
    fact = pd.DataFrame(
        {
            "Дата документа": ["2026-03-05", "2026-03-20", "2026-03-21"],
            "ЦФО (Документ)": ["A", "A", "C"],
            "Статья ДДС": ["Rent", "Meals", "Ops"],
            "Сумма": [130, 20, 90],
        }
    )
    service._write_pending_file(session_id, "plan", "plan.xlsx", b"plan")
    service._write_pending_file(session_id, "fact", "fact.xlsx", b"fact")
    service._read_dataframe = lambda file_name, content: plan if file_name == "plan.xlsx" else fact  # type: ignore[method-assign]

    result = service.confirm(
        session_id=session_id,
        config={
            "source_type": "planfact",
            "plan": {
                "cfo_column": "ЦФО",
                "article_column": "статья CF",
                "monthly_metric": "CF",
                "monthly_columns": {"2026-03": "CF Mar", "2026-04": "CF Apr"},
            },
            "fact": {
                "date_column": "Дата документа",
                "cfo_column": "ЦФО (Документ)",
                "article_column": "Статья ДДС",
                "amount_column": "Сумма",
            },
        },
        ttl_seconds=3600,
    )

    assert "planfact_by_cfo_period" in result.table_names
    assert store.load_data_catalog(session_id) is None
    by_cfo = csv_runtime.query_dataframe(
        session_id,
        "SELECT cfo, period, plan_amount, fact_amount, variance_amount FROM planfact_by_cfo_period ORDER BY cfo",
    )
    assert by_cfo.to_dict(orient="records") == [
        {"cfo": "A", "period": "2026-03", "plan_amount": 150.0, "fact_amount": 150.0, "variance_amount": 0.0},
        {"cfo": "B", "period": "2026-03", "plan_amount": 70.0, "fact_amount": 0.0, "variance_amount": -70.0},
        {"cfo": "C", "period": "2026-03", "plan_amount": 0.0, "fact_amount": 90.0, "variance_amount": 90.0},
    ]
    assert set(by_cfo["period"]) == {"2026-03"}
    lineage = csv_runtime.query_dataframe(
        session_id,
        """
        SELECT plan_source_row_ids, fact_source_row_ids
        FROM planfact_by_cfo_period
        WHERE cfo = 'A' AND period = '2026-03'
        """,
    ).iloc[0]
    assert lineage["plan_source_row_ids"] == "1,2"
    assert lineage["fact_source_row_ids"] == "1,2"

    by_article = csv_runtime.query_dataframe(
        session_id,
        """
        SELECT cfo, article_key, plan_amount, fact_amount
        FROM planfact_by_cfo_article_period
        ORDER BY cfo, article_key
        """,
    )
    assert by_article.to_dict(orient="records") == [
        {"cfo": "A", "article_key": "meals", "plan_amount": 0.0, "fact_amount": 20.0},
        {"cfo": "A", "article_key": "rent", "plan_amount": 100.0, "fact_amount": 130.0},
        {"cfo": "A", "article_key": "travel", "plan_amount": 50.0, "fact_amount": 0.0},
        {"cfo": "B", "article_key": "ops", "plan_amount": 70.0, "fact_amount": 0.0},
        {"cfo": "C", "article_key": "ops", "plan_amount": 0.0, "fact_amount": 90.0},
    ]

    fact_monthly = csv_runtime.query_dataframe(
        session_id,
        "SELECT cfo, period, fact_amount, fact_rows FROM planfact_fact_monthly ORDER BY cfo",
    )
    assert fact_monthly.to_dict(orient="records") == [
        {"cfo": "A", "period": "2026-03", "fact_amount": 150.0, "fact_rows": 2},
        {"cfo": "C", "period": "2026-03", "fact_amount": 90.0, "fact_rows": 1},
    ]

    state = store.load_session(session_id)
    assert state is not None
    assert state.source_type == "planfact"
    assert state.csv_loaded is True
    assert state.csv_session_id == session_id
    assert state.chat_history, "expected AI First Look chat message"
    assert state.chat_history[0]["role"] == "ai"
    assert "AI First Look" in state.chat_history[0]["content"]
    assert any(artifact.get("type") == "plot" for artifact in state.artifacts)
    assert any(
        isinstance(artifact, dict) and artifact.get("meta", {}).get("producer_tool") == "planfact_first_look"
        for artifact in state.artifacts
    )


def test_planfact_extra_key_refines_article_matching(tmp_path: Path) -> None:
    service, _store, csv_runtime, session_id = _service(tmp_path)
    plan = pd.DataFrame(
        {
            "CFO": ["A"],
            "Article": ["Rent"],
            "BusinessUnit": ["BU-1"],
            "CF Mar": [100],
        }
    )
    fact = pd.DataFrame(
        {
            "DocDate": ["2026-03-05"],
            "CFO Doc": ["A"],
            "FactArticle": ["Rent"],
            "BU Doc": ["BU-2"],
            "Amount": [130],
        }
    )
    service._write_pending_file(session_id, "plan", "plan.xlsx", b"plan")
    service._write_pending_file(session_id, "fact", "fact.xlsx", b"fact")
    service._read_dataframe = lambda file_name, content: plan if file_name == "plan.xlsx" else fact  # type: ignore[method-assign]

    service.confirm(
        session_id=session_id,
        config={
            "source_type": "planfact",
            "plan": {
                "cfo_column": "CFO",
                "article_column": "Article",
                "extra_key_column": "BusinessUnit",
                "monthly_metric": "CF",
                "monthly_columns": {"2026-03": "CF Mar"},
            },
            "fact": {
                "date_column": "DocDate",
                "cfo_column": "CFO Doc",
                "article_column": "FactArticle",
                "extra_key_column": "BU Doc",
                "amount_column": "Amount",
            },
            "join": {"extra_keys": ["extra_key"]},
        },
        ttl_seconds=3600,
    )

    by_article = csv_runtime.query_dataframe(
        session_id,
        """
        SELECT cfo, article_key, extra_key, plan_amount, fact_amount
        FROM planfact_by_cfo_article_period
        ORDER BY extra_key
        """,
    )
    assert by_article.to_dict(orient="records") == [
        {"cfo": "A", "article_key": "rent", "extra_key": "bu-1", "plan_amount": 100.0, "fact_amount": 0.0},
        {"cfo": "A", "article_key": "rent", "extra_key": "bu-2", "plan_amount": 0.0, "fact_amount": 130.0},
    ]


def test_planfact_service_content_is_carried_into_breakdowns(tmp_path: Path) -> None:
    service, store, csv_runtime, session_id = _service(tmp_path)
    plan = pd.DataFrame(
        {
            "CFO": ["A", "A"],
            "Article": ["Rent", "Travel"],
            "CF Mar": [100, 50],
        }
    )
    fact = pd.DataFrame(
        {
            "DocDate": ["2026-03-05", "2026-03-20"],
            "CFO Doc": ["A", "A"],
            "FactArticle": ["Rent", "Meals"],
            "ServiceContent": ["Office rent March", "Team lunch"],
            "Amount": [130, 20],
        }
    )
    service._write_pending_file(session_id, "plan", "plan.xlsx", b"plan")
    service._write_pending_file(session_id, "fact", "fact.xlsx", b"fact")
    service._read_dataframe = lambda file_name, content: plan if file_name == "plan.xlsx" else fact  # type: ignore[method-assign]

    service.confirm(
        session_id=session_id,
        config={
            "source_type": "planfact",
            "plan": {
                "cfo_column": "CFO",
                "article_column": "Article",
                "monthly_metric": "CF",
                "monthly_columns": {"2026-03": "CF Mar"},
            },
            "fact": {
                "date_column": "DocDate",
                "cfo_column": "CFO Doc",
                "article_column": "FactArticle",
                "service_content_column": "ServiceContent",
                "amount_column": "Amount",
            },
        },
        ttl_seconds=3600,
    )

    by_article = csv_runtime.query_dataframe(
        session_id,
        """
        SELECT cfo, article_key, service_content, plan_amount, fact_amount
        FROM planfact_by_cfo_article_period
        ORDER BY cfo, article_key
        """,
    )
    assert by_article.to_dict(orient="records") == [
        {
            "cfo": "A",
            "article_key": "meals",
            "service_content": "Team lunch",
            "plan_amount": 0.0,
            "fact_amount": 20.0,
        },
        {
            "cfo": "A",
            "article_key": "rent",
            "service_content": "Office rent March",
            "plan_amount": 100.0,
            "fact_amount": 130.0,
        },
        {
            "cfo": "A",
            "article_key": "travel",
            "service_content": None,
            "plan_amount": 50.0,
            "fact_amount": 0.0,
        },
    ]

    state = store.load_session(session_id)
    assert state is not None
    assert state.chat_history
    assert "Содержание услуги: Office rent March" in state.chat_history[0]["content"]


def test_planfact_detect_picks_service_content_column(tmp_path: Path) -> None:
    service, _store, _csv_runtime, session_id = _service(tmp_path)
    plan = pd.DataFrame(
        {
            "CFO": ["A"],
            "Article": ["Rent"],
            "CF Mar": [100],
        }
    )
    fact = pd.DataFrame(
        {
            "DocDate": ["2026-03-05"],
            "CFO Doc": ["A"],
            "FactArticle": ["Rent"],
            "Содержание услуги": ["Office rent March"],
            "Amount": [130],
        }
    )
    service._read_dataframe = lambda file_name, content: plan if file_name == "plan.xlsx" else fact  # type: ignore[method-assign]

    result = service.detect(
        session_id=session_id,
        plan_file=PlanfactUploadFile(file_name="plan.xlsx", content=b"plan"),
        fact_file=PlanfactUploadFile(file_name="fact.xlsx", content=b"fact"),
    )

    assert result.suggested_config["fact"]["service_content_column"] == "Содержание услуги"
def test_planfact_v3_fields_mapping_artifacts_and_duckdb_tables(tmp_path: Path) -> None:
    service, store, csv_runtime, session_id = _service(tmp_path)
    plan_codes = [str(11000000 + index) for index in range(13)]
    fact_codes = [str(22000000 + index) for index in range(13)]
    articles = [f"{code} | Plan Article {index:02d}" for index, code in enumerate(plan_codes)]
    plan = pd.DataFrame(
        {
            "CFO": ["A"] * len(articles),
            "Статья PL": articles,
            "Plan Counterparty": [f"Plan Vendor {index:02d}" for index in range(13)],
            "PL Mar": [100 + index for index in range(13)],
        }
    )
    fact = pd.DataFrame(
        {
            "DocDate": ["2026-03-05"] * len(articles),
            "CFO Doc": ["A"] * len(articles),
            "Статьи затрат": [
                f"Fact Article {index:02d} ({code})" for index, code in enumerate(fact_codes)
            ],
            "ArticleDDS": ["Wrong DDS"] * len(articles),
            "ServiceContent": [f"Service {index:02d}" for index in range(13)],
            "Fact Counterparty": [f"Fact Vendor {index:02d}" for index in range(13)],
            "Fact Contract": [f"Contract {index:02d}" for index in range(13)],
            "Amount": [150 + index for index in range(13)],
        }
    )
    mapping = pd.DataFrame(
        {
            "Код 1С": fact_codes,
            "Line Code PL": plan_codes,
        }
    )

    def read_dataframe(file_name: str, content: bytes, **_: object) -> pd.DataFrame:
        return {"plan.xlsx": plan, "fact.xlsx": fact, "mapping.xlsx": mapping}[file_name]

    service._read_dataframe = read_dataframe  # type: ignore[method-assign]
    detect = service.detect(
        session_id=session_id,
        plan_file=PlanfactUploadFile(file_name="plan.xlsx", content=b"plan"),
        fact_file=PlanfactUploadFile(file_name="fact.xlsx", content=b"fact"),
        mapping_file=PlanfactUploadFile(file_name="mapping.xlsx", content=b"mapping"),
    )

    assert detect.suggested_config["fact"]["article_column"] == "Статьи затрат"
    assert detect.suggested_config["fact"]["contract_column"] == "Fact Contract"
    assert detect.suggested_config["plan"]["counterparty_column"] == "Plan Counterparty"
    assert detect.suggested_config["article_mapping_source"] == {"file_name": "mapping.xlsx", "row_count": 13}

    service._write_pending_file(session_id, "plan", "plan.xlsx", b"plan")
    service._write_pending_file(session_id, "fact", "fact.xlsx", b"fact")
    store.add_serialized_artifacts(
        session_id,
        [{"id": "old", "type": "note", "meta": {"producer_tool": "planfact_first_look"}}],
    )

    result = service.confirm(session_id=session_id, config=detect.suggested_config, ttl_seconds=3600)
    assert "planfact_by_cfo_article_period" in result.table_names

    rows = csv_runtime.query_dataframe(
        session_id,
        """
        SELECT article, service_content, plan_counterparty, fact_counterparty, fact_contract, article_match_type
        FROM planfact_by_cfo_article_period
        ORDER BY article
        LIMIT 1
        """,
    ).to_dict(orient="records")
    assert rows == [
        {
            "article": "11000000 | Plan Article 00",
            "service_content": "Service 00",
            "plan_counterparty": "Plan Vendor 00",
            "fact_counterparty": "Fact Vendor 00",
            "fact_contract": "Contract 00",
            "article_match_type": "manual",
        }
    ]

    state = store.load_session(session_id)
    assert state is not None
    assert all(artifact.get("id") != "old" for artifact in state.artifacts if isinstance(artifact, dict))
    artifacts_by_id = {
        artifact.get("id"): artifact
        for artifact in state.artifacts
        if isinstance(artifact, dict) and artifact.get("meta", {}).get("producer_tool") == "planfact_first_look"
    }
    assert artifacts_by_id["planfact_first_look_variance_donut"]["meta"]["board_width_units"] == 4
    assert artifacts_by_id["planfact_first_look_plan_to_fact_waterfall"]["meta"]["board_width_units"] == 8

    article_table = artifacts_by_id["planfact_first_look_article_summary"]
    assert article_table["data"]["data"]["columns"][:6] == [
        "ЦФО",
        "Статья",
        "Содержание услуги",
        "Контрагент план",
        "Контрагент факт",
        "Договор",
    ]
    assert len(article_table["data"]["data"]["data"]) == 13
    dashboard = artifacts_by_id["planfact_first_look_dashboard"]["data"]["data"]
    assert dashboard["table"][0]["plan_counterparty"].startswith("Plan Vendor")
    assert "Сравни март" not in "\n".join(dashboard["suggested_questions"])


def test_planfact_detects_russian_detail_columns() -> None:
    service = PlanfactSourceService.__new__(PlanfactSourceService)
    plan = pd.DataFrame(
        columns=["ЦФО", "Статья PL", "L Контрагент 1С УХ", "PL Mar"]
    )
    fact = pd.DataFrame(
        columns=[
            "Дата документа",
            "ЦФО (Документ)",
            "Статьи затрат",
            "Сумма",
            "Контрагент",
            "Договор",
        ]
    )

    plan_config, _ = service._detect_plan_mapping(plan, year=2026)
    fact_config, _ = service._detect_fact_mapping(fact)

    assert plan_config["counterparty_column"] == "L Контрагент 1С УХ"
    assert fact_config["counterparty_column"] == "Контрагент"
    assert fact_config["contract_column"] == "Договор"


def test_planfact_summary_keeps_exact_numbers_only_for_export() -> None:
    service = PlanfactSourceService.__new__(PlanfactSourceService)
    artifact = service._cfo_summary_table_artifact(
        pd.DataFrame(
            {
                "cfo": ["A"],
                "plan_amount": [1437258.4810761],
                "fact_amount": [5229649.31],
                "variance_amount": [3792390.8289239],
            }
        ),
        focus_period="2026-03",
        plan_file_name="plan.xlsx",
        fact_file_name="fact.xlsx",
        metric_type="expense",
    )

    assert artifact is not None
    assert artifact["data"]["data"]["data"][0][1] == "1,4 млн ₽"
    assert artifact["data"]["export_data"]["data"][0][1] == 1437258.4810761
