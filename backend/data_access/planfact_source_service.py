from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, Field

from backend.auth.blob_store import BlobWrite, PostgresBlobStore
from backend.core.json_utils import make_json_safe
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.tabular_preprocessing import (
    TabularPreprocessingOptions,
    read_tabular_dataframe,
)
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.session_source import SessionManifest, SessionSource
from backend.sessions.session_store import SessionStore

PlanfactFileKind = Literal["plan", "fact"]

_MONTHS: dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_ARTICLE_STOP_PHRASES = [
    "услуги по",
    "платежи за",
    "платежи по",
    "расходы на",
    "оплата",
    "приобретение",
    "закупка",
    "покупка",
    "начисление",
]

_ARTICLE_ABBREVIATIONS = {
    "нма": "нематериальные активы",
    "ос": "основные средства",
    "по": "программное обеспечение",
    "ит": "информационные технологии",
    "дмс": "добровольное медицинское страхование",
    "тмц": "товарно материальные ценности",
}

_ARTICLE_DICTIONARY = {
    "услуги по поддержке лицензионного по": "поддержка лицензионного по",
    "услуги по заказной разработке по": "заказная разработка по",
    "услуги по облачным сервисам": "облачные сервисы",
    "логистические услуги": "логистические расходы",
    "прочие hr платежи": "прочие hr",
    "поддержке лицензионного программное обеспечение": "поддержка лицензионного по",
    "поддержка лицензионного программное обеспечение": "поддержка лицензионного по",
    "заказной разработке программное обеспечение": "заказная разработка по",
    "заказная разработка программное обеспечение": "заказная разработка по",
    "облачные сервисы": "облачные сервисы",
    "нематериальные активы": "амортизация нематериальных активов",
}


class PlanfactSourceError(ValueError):
    """Raised when a plan-fact source cannot be detected or confirmed."""


class PlanfactUploadFile(BaseModel):
    file_name: str = Field(..., min_length=1, max_length=255)
    content: bytes = Field(..., min_length=1)
    content_type: str | None = Field(default=None, max_length=255)


class PlanfactDetectResult(BaseModel):
    session_id: str
    source_type: str = "planfact"
    plan: dict[str, Any]
    fact: dict[str, Any]
    suggested_config: dict[str, Any]
    mapping: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


class PlanfactConfirmResult(BaseModel):
    session_id: str
    source_type: str = "planfact"
    csv_session_id: str
    table_names: list[str]
    config: dict[str, Any]
    rows: dict[str, int]
    expires_at: int


class PlanfactSourceService:
    """Separate managed ingestion flow for plan vs actual Excel/CSV pairs."""

    DERIVED_TABLES: ClassVar[list[str]] = [
        "planfact_plan_long",
        "planfact_fact_monthly",
        "planfact_by_cfo_period",
        "planfact_by_cfo_article_period",
        "planfact_focus_by_cfo_period",
        "planfact_focus_by_cfo_article_period",
    ]

    def __init__(
        self,
        *,
        store: SessionStore,
        csv_runtime: CSVSessionRuntime,
        manifest_store: ManifestStore,
        storage_dir: str | Path,
        blob_store: PostgresBlobStore | None = None,
    ) -> None:
        self._store = store
        self._csv_runtime = csv_runtime
        self._manifest_store = manifest_store
        self._storage_dir = Path(storage_dir)
        self._blob_store = blob_store

    def detect(
        self,
        *,
        session_id: str,
        plan_file: PlanfactUploadFile,
        fact_file: PlanfactUploadFile,
        mapping_file: PlanfactUploadFile | None = None,
        user_id: int | None = None,
    ) -> PlanfactDetectResult:
        self._ensure_session(session_id)
        plan_file_name = self._safe_file_name(plan_file.file_name)
        fact_file_name = self._safe_file_name(fact_file.file_name)
        plan_df = self._read_dataframe(plan_file_name, plan_file.content)
        fact_df = self._read_dataframe(fact_file_name, fact_file.content)
        if plan_df.empty or not len(plan_df.columns):
            raise PlanfactSourceError("Plan file has no readable rows or columns")
        if fact_df.empty or not len(fact_df.columns):
            raise PlanfactSourceError("Fact file has no readable rows or columns")

        year = self._infer_fact_year(fact_df) or 2026
        plan_mapping, plan_warnings = self._detect_plan_mapping(plan_df, year=year)
        fact_mapping, fact_warnings = self._detect_fact_mapping(fact_df)
        warnings = [*plan_warnings, *fact_warnings]
        config = {
            "source_type": "planfact",
            "plan": {
                "file_name": plan_file_name,
                "table_name": "planfact_plan_raw",
                **plan_mapping,
            },
            "fact": {
                "file_name": fact_file_name,
                "table_name": "planfact_fact_raw",
                **fact_mapping,
            },
            "join": {
                "keys": ["cfo", "period"],
                "article_keys": ["cfo", "article_key", "period"],
                "extra_keys": [],
                "variance_formula": "fact_amount - plan_amount",
            },
        }
        mapping_preview = None
        if mapping_file is not None:
            mapping_file_name = self._safe_file_name(mapping_file.file_name)
            mapping_df = self._read_dataframe(mapping_file_name, mapping_file.content, preprocess=False)
            article_mapping = self._detect_article_mapping(mapping_df)
            config["article_mapping"] = article_mapping
            config["article_mapping_source"] = {
                "file_name": mapping_file_name,
                "row_count": len(article_mapping),
            }
            mapping_preview = self._preview_payload(mapping_file_name, mapping_df)
        self._apply_pl_article_key_defaults(config["plan"], config["fact"], fact_df)
        try:
            plan_long = self._build_plan_long(plan_df, config["plan"])
            fact_monthly, fact_article = self._build_fact_monthly(fact_df, config["fact"])
            config["cfo_matching"] = self._cfo_match_report(plan_long, fact_monthly)
            config["article_matching"] = self._apply_article_matching(plan_long, fact_article, config)
        except Exception as exc:
            warnings.append(f"Matching preview failed: {exc}")
        self._write_pending_file(session_id, "plan", plan_file_name, plan_file.content)
        self._write_pending_file(session_id, "fact", fact_file_name, fact_file.content)
        if self._blob_store is not None:
            if user_id is None:
                raise PlanfactSourceError("user_id is required for durable planfact uploads")
            self._blob_store.put_many(
                user_id=user_id,
                session_id=session_id,
                kind="planfact_plan",
                items=[
                    BlobWrite(
                        logical_name=plan_file_name,
                        media_type=plan_file.content_type or "application/octet-stream",
                        content=plan_file.content,
                    )
                ],
            )
            self._blob_store.put_many(
                user_id=user_id,
                session_id=session_id,
                kind="planfact_fact",
                items=[
                    BlobWrite(
                        logical_name=fact_file_name,
                        media_type=fact_file.content_type or "application/octet-stream",
                        content=fact_file.content,
                    )
                ],
            )
            if mapping_file is not None:
                self._blob_store.put_many(
                    user_id=user_id,
                    session_id=session_id,
                    kind="planfact_mapping",
                    items=[
                        BlobWrite(
                            logical_name=self._safe_file_name(mapping_file.file_name),
                            media_type=mapping_file.content_type or "application/octet-stream",
                            content=mapping_file.content,
                        )
                    ],
                )
        self._write_config(
            session_id,
            config,
            name="planfact_detected_config.json",
            user_id=user_id,
        )
        return PlanfactDetectResult(
            session_id=session_id,
            plan=self._preview_payload(plan_file_name, plan_df),
            fact=self._preview_payload(fact_file_name, fact_df),
            suggested_config=config,
            mapping=mapping_preview,
            warnings=warnings,
        )

    def confirm(
        self,
        *,
        session_id: str,
        config: dict[str, Any],
        ttl_seconds: int | None = None,
        user_id: int | None = None,
    ) -> PlanfactConfirmResult:
        self._ensure_session(session_id)
        clean_config = self._normalize_config(config)
        plan_file_name, plan_content = self._read_pending_file(session_id, "plan")
        fact_file_name, fact_content = self._read_pending_file(session_id, "fact")
        clean_config["plan"]["file_name"] = clean_config["plan"].get("file_name") or plan_file_name
        clean_config["fact"]["file_name"] = clean_config["fact"].get("file_name") or fact_file_name
        plan_raw = self._read_dataframe(plan_file_name, plan_content)
        fact_raw = self._read_dataframe(fact_file_name, fact_content)
        plan_raw = plan_raw.copy()
        fact_raw = fact_raw.copy()
        plan_raw.insert(0, "plan_source_row_id", range(1, len(plan_raw) + 1))
        fact_raw.insert(0, "fact_source_row_id", range(1, len(fact_raw) + 1))
        self._apply_pl_article_key_defaults(clean_config["plan"], clean_config["fact"], fact_raw)

        plan_long = self._build_plan_long(plan_raw, clean_config["plan"])
        fact_monthly, fact_article = self._build_fact_monthly(fact_raw, clean_config["fact"])
        fact_quality = self._fact_quality_counts(fact_raw, clean_config["fact"])
        fact_monthly, fact_article = self._apply_cfo_mapping(
            plan_long, fact_monthly, fact_article, clean_config
        )
        cfo_match_report = self._cfo_match_report(plan_long, fact_monthly)
        clean_config["cfo_matching"] = cfo_match_report
        article_match_report = self._apply_article_matching(plan_long, fact_article, clean_config)
        clean_config["article_matching"] = article_match_report
        fact_periods = set(fact_monthly["period"].dropna().astype(str))
        joined_plan = plan_long.loc[plan_long["period"].astype(str).isin(fact_periods)].copy()
        by_cfo = self._build_by_cfo_period(joined_plan, fact_monthly)
        by_article = self._build_by_cfo_article_period(joined_plan, fact_article)
        focus_period, focus_by_cfo, focus_by_article = self._first_look_focus_slices(
            fact_monthly=fact_monthly,
            by_cfo=by_cfo,
            by_article=by_article,
        )
        if focus_period:
            clean_config["focus_period"] = focus_period
        tables = {
            "planfact_plan_raw": plan_raw,
            "planfact_fact_raw": fact_raw,
            "planfact_plan_long": plan_long,
            "planfact_fact_monthly": fact_monthly,
            "planfact_by_cfo_period": by_cfo,
            "planfact_by_cfo_article_period": by_article,
            "planfact_focus_by_cfo_period": focus_by_cfo,
            "planfact_focus_by_cfo_article_period": focus_by_article,
        }
        info = self._csv_runtime.register_dataframes(
            session_id=session_id,
            tables=tables,
            ttl_seconds=ttl_seconds,
        )
        clean_config["tables"] = list(tables.keys())
        self._persist_manifest(session_id, clean_config, tables, user_id=user_id)
        self._write_config(session_id, clean_config, user_id=user_id)
        self._store.set_dataset_name(session_id, "План-факт")
        self._store.set_source(
            session_id,
            source_type="planfact",
            source_ref_id="planfact",
            source_label="План-факт",
            source_mode="duckdb",
        )
        self._store.set_csv_runtime_state(
            session_id,
            csv_loaded=True,
            csv_session_id=info.session_id,
            csv_table_names=list(info.table_names),
            csv_expires_at=info.expires_at,
        )
        report_artifacts = self._build_first_look_artifacts(
            plan_long=plan_long,
            fact_monthly=fact_monthly,
            by_cfo=by_cfo,
            by_article=by_article,
            fact_quality=fact_quality,
            article_matching=article_match_report,
            plan_file_name=str(clean_config["plan"].get("file_name") or ""),
            fact_file_name=str(clean_config["fact"].get("file_name") or ""),
        )
        first_look = self._build_first_look_message(
            plan_long=plan_long,
            fact_monthly=fact_monthly,
            by_cfo=by_cfo,
            by_article=by_article,
            plan_file_name=str(clean_config["plan"].get("file_name") or ""),
            fact_file_name=str(clean_config["fact"].get("file_name") or ""),
        )
        if first_look:
            if report_artifacts:
                self._store.add_serialized_artifacts(
                    session_id,
                    report_artifacts,
                    replace_producer_tool="planfact_first_look",
                )
                self._store.add_chat_message(session_id, "ai", first_look, artifacts=report_artifacts)
            else:
                self._store.add_chat_message(session_id, "ai", first_look)
        return PlanfactConfirmResult(
            session_id=session_id,
            csv_session_id=info.session_id,
            table_names=list(info.table_names),
            config=clean_config,
            rows={name: len(df) for name, df in tables.items()},
            expires_at=info.expires_at,
        )

    def get_config(self, session_id: str) -> dict[str, Any]:
        self._ensure_session(session_id)
        path = self._config_path(session_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        source = self._manifest_store.load(session_id).source_by_alias("planfact")
        if source is not None:
            config = (source.preprocessing_summary or {}).get("planfact_config")
            if isinstance(config, dict):
                return config
        blob = (
            self._blob_store.get_latest_for_session(
                session_id=session_id,
                kind="planfact_config",
            )
            if self._blob_store is not None
            else None
        )
        if blob is None:
            raise PlanfactSourceError("Planfact config not found")
        return json.loads(blob.content.decode("utf-8"))

    def update_config(
        self,
        session_id: str,
        patch: dict[str, Any],
        *,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        current = self.get_config(session_id)
        merged = self._deep_merge(current, patch)
        normalized = self._normalize_config(merged)
        self._write_config(session_id, normalized, user_id=user_id)
        return normalized

    def _ensure_session(self, session_id: str) -> None:
        if self._store.load_session(session_id) is None:
            raise PlanfactSourceError("Session not found")

    @staticmethod
    def _safe_file_name(file_name: str) -> str:
        clean = Path(str(file_name or "")).name.strip()
        if not clean:
            raise PlanfactSourceError("File name must not be empty")
        if Path(clean).suffix.lower() not in {".csv", ".xlsx"}:
            raise PlanfactSourceError("Planfact files must be .csv or .xlsx")
        return clean

    def _read_dataframe(
        self, file_name: str, content: bytes, *, preprocess: bool = True
    ) -> pd.DataFrame:
        file_format = "xlsx" if Path(file_name).suffix.lower() == ".xlsx" else "csv"
        try:
            data = read_tabular_dataframe(
                content,
                file_format=file_format,
                options=TabularPreprocessingOptions(enabled=preprocess),
            )
        except Exception as exc:
            raise PlanfactSourceError(f"Failed to read '{file_name}': {exc}") from exc
        df = data.dataframe
        if not isinstance(df, pd.DataFrame):
            raise PlanfactSourceError(f"File '{file_name}' did not parse as a DataFrame")
        return df

    def _detect_plan_mapping(self, df: pd.DataFrame, *, year: int) -> tuple[dict[str, Any], list[str]]:
        columns = [str(col) for col in df.columns]
        warnings: list[str] = []
        monthly_by_metric = self._detect_monthly_columns(columns, year=year)
        has_pl_article = self._pick_column(columns, ["статья pl"]) is not None
        has_cf_article = self._pick_column(columns, ["статья cf"]) is not None
        if has_pl_article and monthly_by_metric.get("PL"):
            metric = "PL"
        elif has_cf_article and monthly_by_metric.get("CF"):
            metric = "CF"
        else:
            metric = (
                "CF" if len(monthly_by_metric.get("CF", {})) >= len(monthly_by_metric.get("PL", {})) else "PL"
            )
        monthly_columns = monthly_by_metric.get(metric, {})
        cfo = self._pick_column(columns, ["cfo", "цфо"])
        article = self._pick_column(
            columns, ["article", f"статья {metric.lower()}", "статья cf", "статья pl", "екдр", "статья"]
        )
        counterparty = self._pick_column(
            columns,
            [
                "plan counterparty",
                "plancounterparty",
                "counterparty",
                "l контрагент 1с ух",
                "контрагент план",
            ],
        )
        if cfo is None:
            warnings.append("Plan CFO column was not detected")
        if article is None:
            warnings.append("Plan article column was not detected")
        if not monthly_columns:
            warnings.append("Plan monthly CF/PL columns were not detected")
        return (
            {
                "cfo_column": cfo,
                "article_column": article,
                "counterparty_column": counterparty,
                "monthly_metric": metric,
                "monthly_columns": monthly_columns,
            },
            warnings,
        )

    def _detect_fact_mapping(self, df: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
        columns = [str(col) for col in df.columns]
        warnings: list[str] = []
        date = self._pick_column(columns, ["docdate", "doc date", "дата документа", "дата"])
        cfo = self._pick_column(
            columns, ["cfo doc", "cfodoc", "cfo", "цфо (документ)", "цфо документ", "цфо (договор)", "цфо договор", "цфо"]
        )
        article = self._pick_column(columns, ["статья ддс", "статья начисления", "статьи затрат", "статья"])
        amount = self._pick_column(columns, ["сумма"])
        service_content = self._pick_column(
            columns, ["Содержание услуги", "Описание услуги", "Назначение платежа"]
        )
        preferred_article = self._pick_column(columns, ["fact article", "factarticle", "article accrual"])
        if preferred_article:
            article = preferred_article
        amount = self._pick_column(columns, ["amount", "сумма"]) or amount
        service_content = self._pick_column(columns, ["servicecontent", "service content"]) or service_content
        counterparty = self._pick_column(
            columns,
            ["fact counterparty", "factcounterparty", "counterparty", "контрагент"],
        )
        contract = self._pick_column(
            columns,
            ["fact contract", "factcontract", "contract", "договор"],
        )
        for label, value in [
            ("Fact date", date),
            ("Fact CFO", cfo),
            ("Fact article", article),
            ("Fact amount", amount),
        ]:
            if value is None:
                warnings.append(f"{label} column was not detected")
        return (
            {
                "date_column": date,
                "cfo_column": cfo,
                "article_column": article,
                "amount_column": amount,
                "service_content_column": service_content,
                "counterparty_column": counterparty,
                "contract_column": contract,
            },
            warnings,
        )

    def _detect_article_mapping(self, df: pd.DataFrame) -> list[dict[str, str]]:
        columns = [str(column) for column in df.columns]
        fact_column = self._pick_column(columns, ["Код 1С"])
        plan_column = self._pick_column(columns, ["Line Code PL"])
        if not plan_column or not fact_column:
            raise PlanfactSourceError("Mapping file must contain columns 'Код 1С' and 'Line Code PL'")

        mapping: dict[str, str] = {}
        for fact_value, plan_value in zip(df[fact_column], df[plan_column], strict=False):
            fact_code = self._mapping_article_code(fact_value)
            plan_code = self._mapping_article_code(plan_value)
            if fact_code and plan_code:
                mapping[fact_code] = plan_code
        if not mapping:
            raise PlanfactSourceError("Mapping file has no valid code pairs")
        return [
            {"fact_article_key": fact_code, "plan_article_key": plan_code}
            for fact_code, plan_code in mapping.items()
        ]

    @staticmethod
    def _mapping_article_code(value: object) -> str:
        text = PlanfactSourceService._clean_value(value)
        numeric = re.fullmatch(r"(\d+)(?:\.(\d+))?", text)
        if numeric:
            integer, decimal = numeric.groups()
            text = integer.zfill(8)
            if decimal and int(decimal):
                text += f".{decimal}"
        return PlanfactSourceService.extract_article_code(text)

    @staticmethod
    def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
        normalized = {PlanfactSourceService._norm_text(col): col for col in columns}
        for candidate in candidates:
            needle = PlanfactSourceService._norm_text(candidate)
            if needle in normalized:
                return normalized[needle]
        for candidate in candidates:
            needle = PlanfactSourceService._norm_text(candidate)
            for key, original in normalized.items():
                if needle and needle in key:
                    return original
        return None

    @staticmethod
    def _detect_monthly_columns(columns: list[str], *, year: int) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {"CF": {}, "PL": {}}
        for col in columns:
            clean = str(col).strip()
            match = re.match(r"^(CF|PL)\s*[-_ ]?\s*([A-Za-z]{3,4})$", clean, flags=re.IGNORECASE)
            if not match:
                continue
            metric = match.group(1).upper()
            month = _MONTHS.get(match.group(2).lower())
            if month is None:
                continue
            result[metric][f"{year:04d}-{month:02d}"] = clean
        return result

    @staticmethod
    def _infer_fact_year(df: pd.DataFrame) -> int | None:
        columns = [str(col) for col in df.columns]
        date_col = PlanfactSourceService._pick_column(columns, ["дата документа", "дата"])
        if date_col is None:
            return None
        dates = PlanfactSourceService._parse_dates(df[date_col])
        years = dates.dropna().dt.year
        if years.empty:
            return None
        return int(years.mode().iloc[0])

    @staticmethod
    def _preview_payload(file_name: str, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "sheets": [0],
            "columns": [str(col) for col in df.columns],
            "row_count": len(df),
            "preview": json.loads(df.head(8).to_json(orient="records", date_format="iso")),
        }

    def _build_plan_long(self, df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
        cfo_col = self._required_column(df, cfg.get("cfo_column"), "plan.cfo_column")
        article_col = self._required_column(df, cfg.get("article_column"), "plan.article_column")
        extra_col = self._optional_column(df, cfg.get("extra_key_column"), "plan.extra_key_column")
        counterparty_col = self._optional_column(df, cfg.get("counterparty_column"), "plan.counterparty_column")
        monthly_columns = cfg.get("monthly_columns")
        if not isinstance(monthly_columns, dict) or not monthly_columns:
            raise PlanfactSourceError("plan.monthly_columns must not be empty")
        records: list[pd.DataFrame] = []
        for period, column in sorted(monthly_columns.items()):
            amount_col = self._required_column(df, column, f"plan.monthly_columns.{period}")
            part = pd.DataFrame(
                {
                    "plan_source_row_id": (
                        df["plan_source_row_id"]
                        if "plan_source_row_id" in df.columns
                        else range(1, len(df) + 1)
                    ),
                    "cfo": df[cfo_col].map(self._clean_value),
                    "article": df[article_col].map(self._clean_value),
                    "period": str(period),
                    "plan_amount": self._to_number(df[amount_col]),
                }
            )
            part["extra_key"] = df[extra_col].map(self._clean_value) if extra_col else ""
            part["plan_counterparty"] = df[counterparty_col].map(self._clean_value) if counterparty_col else ""
            part["period_month"] = int(str(period)[5:7])
            records.append(part)
        out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        key_fn = (
            self.normalize_pl_article_key
            if cfg.get("article_key_mode") == "pl_code"
            else self.normalize_article_key
        )
        out["article_key"] = out["article"].map(key_fn)
        out["extra_key"] = out["extra_key"].map(self.normalize_match_key)
        out["plan_metric"] = str(cfg.get("monthly_metric") or "")
        out["plan_source_file"] = str(cfg.get("file_name") or "")
        return out[
            [
                "plan_source_row_id",
                "cfo",
                "article",
                "article_key",
                "extra_key",
                "plan_counterparty",
                "period",
                "period_month",
                "plan_amount",
                "plan_metric",
                "plan_source_file",
            ]
        ]

    def _build_fact_monthly(self, df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
        date_col = self._required_column(df, cfg.get("date_column"), "fact.date_column")
        cfo_col = self._required_column(df, cfg.get("cfo_column"), "fact.cfo_column")
        article_col = self._required_column(df, cfg.get("article_column"), "fact.article_column")
        amount_col = self._required_column(df, cfg.get("amount_column"), "fact.amount_column")
        extra_col = self._optional_column(df, cfg.get("extra_key_column"), "fact.extra_key_column")
        article_key_cols = self._article_key_columns(df, cfg, article_col)
        article_values = self._first_nonempty_column_values(df, article_key_cols)
        service_content_col = self._optional_column(
            df, cfg.get("service_content_column"), "fact.service_content_column"
        )
        counterparty_col = self._optional_column(df, cfg.get("counterparty_column"), "fact.counterparty_column")
        contract_col = self._optional_column(df, cfg.get("contract_column"), "fact.contract_column")
        base = pd.DataFrame(
            {
                "fact_source_row_id": (
                    df["fact_source_row_id"]
                    if "fact_source_row_id" in df.columns
                    else range(1, len(df) + 1)
                ),
                "cfo": df[cfo_col].map(self._clean_value),
                "article": article_values,
                "date": self._parse_dates(df[date_col]),
                "fact_amount": self._to_number(df[amount_col]),
            }
        )
        base["extra_key"] = df[extra_col].map(self._clean_value) if extra_col else ""
        base["service_content"] = (
            df[service_content_col].map(self._clean_value) if service_content_col else ""
        )
        base["fact_counterparty"] = df[counterparty_col].map(self._clean_value) if counterparty_col else ""
        base["fact_contract"] = df[contract_col].map(self._clean_value) if contract_col else ""
        base = base.dropna(subset=["date"])
        base["period"] = base["date"].dt.strftime("%Y-%m")
        base["period_month"] = base["date"].dt.month.astype(int)
        base["article_key"] = self._article_key_series(
            df.loc[base.index], article_key_cols, cfg.get("article_key_mode")
        )
        base["extra_key"] = base["extra_key"].map(self.normalize_match_key)
        base["fact_source_file"] = str(cfg.get("file_name") or "")
        fact_monthly = (
            base.groupby(["cfo", "period", "period_month"], dropna=False)
            .agg(
                fact_amount=("fact_amount", "sum"),
                fact_rows=("fact_amount", "size"),
                fact_source_file=("fact_source_file", "first"),
                fact_source_row_ids=("fact_source_row_id", self._collect_source_row_ids),
            )
            .reset_index()
        )
        fact_article = (
            base.groupby(["cfo", "article_key", "extra_key", "period", "period_month"], dropna=False)
            .agg(
                article=("article", "first"),
                service_content=("service_content", self._collapse_text_values),
                fact_counterparty=("fact_counterparty", self._collapse_text_values),
                fact_contract=("fact_contract", self._collapse_text_values),
                fact_amount=("fact_amount", "sum"),
                fact_rows=("fact_amount", "size"),
                fact_source_file=("fact_source_file", "first"),
                fact_source_row_ids=("fact_source_row_id", self._collect_source_row_ids),
            )
            .reset_index()
        )
        return fact_monthly, fact_article[
            [
                "cfo",
                "article",
                "article_key",
                "extra_key",
                "service_content",
                "fact_counterparty",
                "fact_contract",
                "period",
                "period_month",
                "fact_amount",
                "fact_rows",
                "fact_source_file",
                "fact_source_row_ids",
            ]
        ]

    def _apply_pl_article_key_defaults(
        self,
        plan_cfg: dict[str, Any],
        fact_cfg: dict[str, Any],
        fact_df: pd.DataFrame,
    ) -> None:
        if str(plan_cfg.get("monthly_metric") or "").upper() != "PL":
            return
        article_columns = self._detect_pl_fact_article_columns(fact_df)
        if article_columns:
            plan_cfg["article_key_mode"] = "pl_code"
            fact_cfg["article_key_mode"] = "pl_code"
            fact_cfg["article_column"] = article_columns[0]
            fact_cfg["article_key_columns"] = article_columns

    def _detect_pl_fact_article_columns(self, df: pd.DataFrame) -> list[str]:
        columns = [str(col) for col in df.columns]
        result: list[str] = []
        for candidate in ["статьи затрат", "номенклатурная группа"]:
            column = self._pick_column(columns, [candidate])
            if column and column not in result:
                result.append(column)
        return result

    def _article_key_columns(
        self, df: pd.DataFrame, cfg: dict[str, Any], article_col: str
    ) -> list[str]:
        configured = cfg.get("article_key_columns")
        if not isinstance(configured, list) or not configured:
            return [article_col]
        return [self._required_column(df, column, "fact.article_key_columns") for column in configured]

    def _first_nonempty_column_values(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.Series:
        result = pd.Series([""] * len(df), index=df.index)
        for column in columns:
            values = df[column].map(self._clean_value)
            result = result.mask(result.eq("") & values.ne(""), values)
        return result

    def _article_key_series(
        self, df: pd.DataFrame, columns: list[str], mode: object
    ) -> pd.Series:
        key_fn = self.extract_article_code if mode == "pl_code" else self.normalize_article_key
        result = pd.Series([""] * len(df), index=df.index)
        for column in columns:
            values = df[column].map(key_fn)
            result = result.mask(result.eq("") & values.ne(""), values)
        return result

    def _fact_quality_counts(self, df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, int]:
        date_col = self._required_column(df, cfg.get("date_column"), "fact.date_column")
        amount_col = self._required_column(df, cfg.get("amount_column"), "fact.amount_column")
        parsed_dates = self._parse_dates(df[date_col])
        parsed_amounts = self._to_number(df[amount_col])
        raw_amount_text = df[amount_col].astype(str).str.strip()
        return {
            "empty_dates": int(parsed_dates.isna().sum()),
            "empty_amounts": int(
                (
                    raw_amount_text.eq("")
                    | raw_amount_text.str.lower().isin(["nan", "none", "null"])
                    | parsed_amounts.isna()
                ).sum()
            ),
        }

    def _apply_cfo_mapping(
        self,
        plan_long: pd.DataFrame,
        fact_monthly: pd.DataFrame,
        fact_article: pd.DataFrame,
        config: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        mapping = self._manual_cfo_mapping(config, plan_long)
        if not mapping:
            return fact_monthly, fact_article
        for frame in [fact_monthly, fact_article]:
            if frame.empty or "cfo" not in frame.columns:
                continue
            frame["original_cfo"] = frame["cfo"]
            frame["cfo_match_type"] = "exact"
            frame["matched_plan_cfo"] = ""
            for index, row in frame.iterrows():
                clean_key = self.normalize_match_key(row.get("cfo"))
                plan_cfo = mapping.get(clean_key)
                if plan_cfo:
                    frame.at[index, "cfo"] = plan_cfo
                    frame.at[index, "cfo_match_type"] = "manual"
                    frame.at[index, "matched_plan_cfo"] = plan_cfo
        fact_monthly = (
            fact_monthly.groupby(["cfo", "period", "period_month"], dropna=False)
            .agg(
                fact_amount=("fact_amount", "sum"),
                fact_rows=("fact_rows", "sum"),
                fact_source_file=("fact_source_file", "first"),
                fact_source_row_ids=("fact_source_row_ids", self._collect_source_row_ids),
            )
            .reset_index()
            if not fact_monthly.empty
            else fact_monthly
        )
        fact_article = (
            fact_article.groupby(["cfo", "article_key", "extra_key", "period", "period_month"], dropna=False)
            .agg(
                article=("article", "first"),
                service_content=("service_content", self._collapse_text_values),
                fact_counterparty=("fact_counterparty", self._collapse_text_values),
                fact_contract=("fact_contract", self._collapse_text_values),
                fact_amount=("fact_amount", "sum"),
                fact_rows=("fact_rows", "sum"),
                fact_source_file=("fact_source_file", "first"),
                fact_source_row_ids=("fact_source_row_ids", self._collect_source_row_ids),
            )
            .reset_index()
            if not fact_article.empty
            else fact_article
        )
        return fact_monthly, fact_article

    @staticmethod
    def _manual_cfo_mapping(config: dict[str, Any], plan_long: pd.DataFrame | None = None) -> dict[str, str]:
        mapping: dict[str, str] = {}
        plan_lookup: dict[str, str] = {}
        if isinstance(plan_long, pd.DataFrame) and not plan_long.empty and "cfo" in plan_long.columns:
            for value in plan_long["cfo"].dropna().unique():
                display = PlanfactSourceService._display_value(value, fallback="")
                key = PlanfactSourceService.normalize_match_key(display)
                if key:
                    plan_lookup[key] = display
        raw = config.get("cfo_mapping")
        if not isinstance(raw, list):
            raw = (config.get("join") or {}).get("cfo_mapping")
        if not isinstance(raw, list):
            return mapping
        for item in raw:
            if not isinstance(item, dict):
                continue
            fact_key = PlanfactSourceService.normalize_match_key(
                item.get("fact_cfo") or item.get("fact_cfo_key")
            )
            plan_key = PlanfactSourceService.normalize_match_key(
                item.get("plan_cfo") or item.get("plan_cfo_key")
            )
            if fact_key and plan_key:
                mapping[fact_key] = plan_lookup.get(
                    plan_key, PlanfactSourceService._display_value(item.get("plan_cfo"), fallback=plan_key)
                )
        return mapping

    def _cfo_match_report(self, plan_long: pd.DataFrame, fact_monthly: pd.DataFrame) -> dict[str, Any]:
        if plan_long.empty and fact_monthly.empty:
            return self._cfo_match_empty_report()

        fact_periods = (
            set(fact_monthly["period"].dropna().astype(str).unique())
            if "period" in fact_monthly.columns
            else set()
        )
        plan_scope = plan_long
        if fact_periods and "period" in plan_long.columns:
            plan_scope = plan_long.loc[plan_long["period"].astype(str).isin(fact_periods)]

        plan_base = plan_scope.copy()
        fact_base = fact_monthly.copy()
        plan_base["cfo_key"] = (
            plan_base["cfo"].map(self.normalize_match_key) if "cfo" in plan_base.columns else ""
        )
        fact_base["cfo_key"] = (
            fact_base["cfo"].map(self.normalize_match_key) if "cfo" in fact_base.columns else ""
        )

        plan_group = (
            plan_base.groupby("cfo_key", dropna=False)
            .agg(cfo=("cfo", "first"), plan_amount=("plan_amount", "sum"))
            .reset_index()
            if not plan_base.empty
            else pd.DataFrame(columns=["cfo_key", "cfo", "plan_amount"])
        )
        fact_group = (
            fact_base.groupby("cfo_key", dropna=False)
            .agg(cfo=("cfo", "first"), fact_amount=("fact_amount", "sum"), fact_rows=("fact_rows", "sum"))
            .reset_index()
            if not fact_base.empty
            else pd.DataFrame(columns=["cfo_key", "cfo", "fact_amount", "fact_rows"])
        )
        merged = plan_group.merge(fact_group, on="cfo_key", how="outer", suffixes=("_plan", "_fact"))
        if merged.empty:
            return self._cfo_match_empty_report()
        for column in ["plan_amount", "fact_amount", "fact_rows"]:
            if column not in merged.columns:
                merged[column] = 0
            merged[column] = merged[column].fillna(0)

        rows: list[dict[str, Any]] = []
        counts = {"matched": 0, "fact_only": 0, "plan_only": 0}
        fact_only_amount = 0.0
        plan_only_amount = 0.0
        for _, row in merged.iterrows():
            plan_amount = self._amount_value(row.get("plan_amount"))
            fact_amount = self._amount_value(row.get("fact_amount"))
            has_plan = abs(plan_amount) > 0
            has_fact = abs(fact_amount) > 0 or self._amount_value(row.get("fact_rows")) > 0
            if has_plan and has_fact:
                status = "matched"
            elif has_fact:
                status = "fact_only"
                fact_only_amount += abs(fact_amount)
            else:
                status = "plan_only"
                plan_only_amount += abs(plan_amount)
            counts[status] += 1
            cfo = (
                self._display_value(row.get("cfo_fact"), fallback="")
                or self._display_value(row.get("cfo_plan"), fallback="")
                or "не указано"
            )
            rows.append(
                {
                    "cfo": cfo,
                    "cfo_key": self._display_value(row.get("cfo_key"), fallback=""),
                    "plan_amount": plan_amount,
                    "fact_amount": fact_amount,
                    "variance_amount": fact_amount - plan_amount,
                    "fact_rows": int(self._amount_value(row.get("fact_rows"))),
                    "status": status,
                }
            )
        rows.sort(
            key=lambda item: (
                0 if item["status"] != "matched" else 1,
                -abs(float(item.get("variance_amount") or 0)),
            )
        )
        plan_cfos = self._plan_cfo_options(plan_long)
        return {
            "stats": {
                **counts,
                "total": len(rows),
                "fact_only_amount": fact_only_amount,
                "plan_only_amount": plan_only_amount,
            },
            "rows": rows[:300],
            "plan_cfos": plan_cfos[:300],
        }

    @staticmethod
    def _cfo_match_empty_report() -> dict[str, Any]:
        return {
            "stats": {
                "matched": 0,
                "fact_only": 0,
                "plan_only": 0,
                "total": 0,
                "fact_only_amount": 0.0,
                "plan_only_amount": 0.0,
            },
            "rows": [],
            "plan_cfos": [],
        }

    @staticmethod
    def _plan_cfo_options(plan_long: pd.DataFrame) -> list[dict[str, str]]:
        if plan_long.empty or "cfo" not in plan_long.columns:
            return []
        options: list[dict[str, str]] = []
        seen: set[str] = set()
        for value in sorted(plan_long["cfo"].dropna().unique(), key=lambda item: str(item).lower()):
            cfo = PlanfactSourceService._display_value(value, fallback="")
            key = PlanfactSourceService.normalize_match_key(cfo)
            if not key or key in seen:
                continue
            seen.add(key)
            options.append({"cfo": cfo, "cfo_key": key})
        return options

    def _apply_article_matching(
        self,
        plan_long: pd.DataFrame,
        fact_article: pd.DataFrame,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        for column, default in [
            ("original_article_key", ""),
            ("matched_plan_article_key", ""),
            ("matched_plan_article", ""),
            ("article_match_type", "unmatched"),
            ("article_match_confidence", 0.0),
        ]:
            if column not in fact_article.columns:
                fact_article[column] = default
        if fact_article.empty or plan_long.empty:
            return self._article_match_report([])

        manual_map = self._manual_article_mapping(config)
        plan_lookup = self._plan_article_lookup(plan_long)
        allow_fuzzy = not any(
            section.get("article_key_mode") == "pl_code"
            for section in (config.get("plan", {}), config.get("fact", {}))
            if isinstance(section, dict)
        )
        match_rows: list[dict[str, Any]] = []

        for index, row in fact_article.iterrows():
            cfo = self._display_value(row.get("cfo"), fallback="")
            extra_key = self._display_value(row.get("extra_key"), fallback="")
            original_key = self._display_value(row.get("article_key"), fallback="")
            fact_article_text = self._display_value(row.get("article"), fallback="")
            candidates = plan_lookup.get((cfo, extra_key), [])
            match = self._match_fact_article(
                fact_key=original_key,
                fact_article=fact_article_text,
                candidates=candidates,
                manual_map=manual_map,
                allow_fuzzy=allow_fuzzy,
            )
            fact_article.at[index, "original_article_key"] = original_key
            fact_article.at[index, "article_match_type"] = match["match_type"]
            fact_article.at[index, "article_match_confidence"] = match["confidence"]
            fact_article.at[index, "matched_plan_article_key"] = match.get("plan_article_key", "")
            fact_article.at[index, "matched_plan_article"] = match.get("plan_article", "")
            if match["match_type"] in {"dictionary", "fuzzy_auto", "manual"} and match.get(
                "plan_article_key"
            ):
                fact_article.at[index, "article_key"] = match["plan_article_key"]
            match_rows.append(
                {
                    "fact_article": fact_article_text or "не указано",
                    "fact_article_key": original_key,
                    "plan_article": match.get("plan_article", ""),
                    "plan_article_key": match.get("plan_article_key", ""),
                    "confidence": match["confidence"],
                    "match_type": match["match_type"],
                    "cfo": cfo,
                    "extra_key": extra_key,
                    "period": self._display_value(row.get("period"), fallback=""),
                    "fact_amount": self._amount_value(row.get("fact_amount")),
                    "fact_rows": int(self._amount_value(row.get("fact_rows"))),
                }
            )
        return self._article_match_report(match_rows, plan_long=plan_long)

    @staticmethod
    def _manual_article_mapping(config: dict[str, Any]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        raw = config.get("article_mapping")
        if not isinstance(raw, list):
            raw = (config.get("join") or {}).get("article_mapping")
        if not isinstance(raw, list):
            return mapping
        for item in raw:
            if not isinstance(item, dict):
                continue
            fact = PlanfactSourceService.normalize_article_key(
                item.get("fact_article") or item.get("fact_article_key")
            )
            plan = PlanfactSourceService.normalize_article_key(
                item.get("plan_article") or item.get("plan_article_key")
            )
            if fact and plan:
                mapping[fact] = plan
        return mapping

    @staticmethod
    def _plan_article_lookup(plan_long: pd.DataFrame) -> dict[tuple[str, str], list[dict[str, str]]]:
        lookup: dict[tuple[str, str], list[dict[str, str]]] = {}
        if plan_long.empty:
            return lookup
        rows = (
            plan_long[["cfo", "extra_key", "article_key", "article"]]
            .drop_duplicates()
            .sort_values(["cfo", "extra_key", "article_key"])
        )
        for _, row in rows.iterrows():
            cfo = PlanfactSourceService._display_value(row.get("cfo"), fallback="")
            extra_key = PlanfactSourceService._display_value(row.get("extra_key"), fallback="")
            key = PlanfactSourceService._display_value(row.get("article_key"), fallback="")
            if not key:
                continue
            lookup.setdefault((cfo, extra_key), []).append(
                {
                    "article_key": key,
                    "article": PlanfactSourceService._display_value(row.get("article"), fallback=key),
                }
            )
        return lookup

    def _match_fact_article(
        self,
        *,
        fact_key: str,
        fact_article: str,
        candidates: list[dict[str, str]],
        manual_map: dict[str, str],
        allow_fuzzy: bool = True,
    ) -> dict[str, Any]:
        clean_fact_key = self.normalize_article_key(fact_key or fact_article)
        if not clean_fact_key:
            return {"match_type": "unmatched", "confidence": 0.0}

        for candidate in candidates:
            if clean_fact_key == candidate["article_key"]:
                return {
                    "match_type": "exact",
                    "confidence": 1.0,
                    "plan_article_key": candidate["article_key"],
                    "plan_article": candidate["article"],
                }

        dictionary_key = manual_map.get(clean_fact_key) or _ARTICLE_DICTIONARY.get(clean_fact_key)
        if dictionary_key:
            normalized_dictionary_key = self.normalize_article_key(dictionary_key)
            for candidate in candidates:
                if normalized_dictionary_key == candidate["article_key"]:
                    return {
                        "match_type": "manual" if clean_fact_key in manual_map else "dictionary",
                        "confidence": 1.0,
                        "plan_article_key": candidate["article_key"],
                        "plan_article": candidate["article"],
                    }

        if not allow_fuzzy:
            return {"match_type": "unmatched", "confidence": 0.0}

        best: dict[str, Any] | None = None
        for candidate in candidates:
            score = self._article_similarity(clean_fact_key, candidate["article_key"])
            if best is None or score > float(best["confidence"]):
                best = {
                    "match_type": "fuzzy_auto"
                    if score >= 0.90
                    else "fuzzy_suggested"
                    if score >= 0.75
                    else "unmatched",
                    "confidence": score,
                    "plan_article_key": candidate["article_key"],
                    "plan_article": candidate["article"],
                }
        if best and best["match_type"] != "unmatched":
            return best
        return {"match_type": "unmatched", "confidence": float(best["confidence"]) if best else 0.0}

    @staticmethod
    def _article_similarity(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        ratio = SequenceMatcher(None, left, right).ratio()
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        token_ratio = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
        if min(len(left_tokens), len(right_tokens)) >= 2:
            containment = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
        else:
            containment = 0.0
        return max(ratio, token_ratio, containment)

    @staticmethod
    def _article_match_report(
        rows: list[dict[str, Any]], *, plan_long: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        counts = {
            "exact": 0,
            "dictionary": 0,
            "fuzzy_auto": 0,
            "fuzzy_suggested": 0,
            "manual": 0,
            "unmatched": 0,
        }
        suggestions: list[dict[str, Any]] = []
        for row in rows:
            match_type = str(row.get("match_type") or "unmatched")
            counts[match_type] = counts.get(match_type, 0) + 1
            if match_type in {"dictionary", "fuzzy_auto", "fuzzy_suggested", "manual", "unmatched"}:
                suggestions.append(row)
        plan_articles: list[dict[str, str]] = []
        if isinstance(plan_long, pd.DataFrame) and not plan_long.empty:
            plan_articles = [
                {
                    "article": PlanfactSourceService._display_value(row.get("article"), fallback=""),
                    "article_key": PlanfactSourceService._display_value(row.get("article_key"), fallback=""),
                    "cfo": PlanfactSourceService._display_value(row.get("cfo"), fallback=""),
                }
                for _, row in plan_long[["cfo", "article", "article_key"]]
                .drop_duplicates()
                .sort_values(["cfo", "article"])
                .iterrows()
                if PlanfactSourceService._display_value(row.get("article_key"), fallback="")
            ]
        return {
            "stats": {
                **counts,
                "auto_matched": counts["exact"]
                + counts["dictionary"]
                + counts["fuzzy_auto"]
                + counts["manual"],
                "needs_confirmation": counts["fuzzy_suggested"],
                "total": len(rows),
            },
            "matches": rows[:500],
            "suggestions": suggestions[:200],
            "plan_articles": plan_articles[:500],
        }

    @staticmethod
    def _build_by_cfo_period(plan_long: pd.DataFrame, fact_monthly: pd.DataFrame) -> pd.DataFrame:
        plan = (
            plan_long.groupby(["cfo", "period", "period_month"], dropna=False)
            .agg(
                plan_amount=("plan_amount", "sum"),
                plan_source_row_ids=("plan_source_row_id", PlanfactSourceService._collect_source_row_ids),
            )
            .reset_index()
        )
        fact = fact_monthly[
            ["cfo", "period", "period_month", "fact_amount", "fact_source_row_ids"]
        ]
        merged = plan.merge(fact, on=["cfo", "period", "period_month"], how="outer")
        return PlanfactSourceService._with_variance(merged)[
            [
                "cfo",
                "period",
                "period_month",
                "plan_amount",
                "fact_amount",
                "variance_amount",
                "variance_pct",
                "plan_source_row_ids",
                "fact_source_row_ids",
            ]
        ]

    @staticmethod
    def _build_by_cfo_article_period(plan_long: pd.DataFrame, fact_article: pd.DataFrame) -> pd.DataFrame:
        plan = (
            plan_long.groupby(["cfo", "article_key", "extra_key", "period", "period_month"], dropna=False)
            .agg(
                plan_article=("article", "first"),
                plan_counterparty=("plan_counterparty", PlanfactSourceService._collapse_text_values),
                plan_amount=("plan_amount", "sum"),
                plan_source_row_ids=(
                    "plan_source_row_id",
                    PlanfactSourceService._collect_source_row_ids,
                ),
            )
            .reset_index()
        )
        fact = fact_article.rename(columns={"article": "fact_article"})
        merged = plan.merge(
            fact[
                [
                    "cfo",
                    "article_key",
                    "extra_key",
                    "service_content",
                    "fact_counterparty",
                    "fact_contract",
                    "period",
                    "period_month",
                    "fact_article",
                    "fact_amount",
                    "fact_rows",
                    "original_article_key",
                    "matched_plan_article_key",
                    "matched_plan_article",
                    "article_match_type",
                    "article_match_confidence",
                    "fact_source_row_ids",
                ]
            ],
            on=["cfo", "article_key", "extra_key", "period", "period_month"],
            how="outer",
        )
        article = merged["plan_article"].fillna(merged["fact_article"])
        merged.insert(1, "article", article)
        out = PlanfactSourceService._with_variance(merged)[
            [
                "cfo",
                "article",
                "article_key",
                "extra_key",
                "service_content",
                "plan_counterparty",
                "fact_counterparty",
                "fact_contract",
                "period",
                "period_month",
                "plan_article",
                "fact_article",
                "plan_amount",
                "fact_amount",
                "variance_amount",
                "variance_pct",
                "fact_rows",
                "original_article_key",
                "matched_plan_article_key",
                "matched_plan_article",
                "article_match_type",
                "article_match_confidence",
                "plan_source_row_ids",
                "fact_source_row_ids",
            ]
        ]
        out["service_content"] = out["service_content"].astype(object).where(
            pd.notna(out["service_content"]),
            None,
        )
        for column in ["plan_counterparty", "fact_counterparty", "fact_contract"]:
            out[column] = out[column].astype(object).where(pd.notna(out[column]), None)
        return out

    def _build_first_look_message(
        self,
        *,
        plan_long: pd.DataFrame,
        fact_monthly: pd.DataFrame,
        by_cfo: pd.DataFrame,
        by_article: pd.DataFrame,
        plan_file_name: str,
        fact_file_name: str,
    ) -> str | None:
        if by_cfo.empty and by_article.empty:
            return None

        period, cfo_slice, article_slice = self._first_look_focus_slices(
            fact_monthly=fact_monthly,
            by_cfo=by_cfo,
            by_article=by_article,
        )
        total_plan = (
            float(cfo_slice["plan_amount"].fillna(0).sum())
            if not cfo_slice.empty
            else float(plan_long["plan_amount"].fillna(0).sum())
        )
        total_fact = (
            float(cfo_slice["fact_amount"].fillna(0).sum())
            if not cfo_slice.empty
            else float(fact_monthly["fact_amount"].fillna(0).sum())
        )
        total_variance = total_fact - total_plan

        fact_without_plan = article_slice.loc[
            (article_slice["plan_amount"].fillna(0) == 0) & (article_slice["fact_amount"].fillna(0) > 0)
        ]
        plan_without_fact = article_slice.loc[
            (article_slice["fact_amount"].fillna(0) == 0) & (article_slice["plan_amount"].fillna(0) > 0)
        ]
        top = article_slice.head(5)
        top_lines = self._format_top_variance_lines(top)
        if not top_lines:
            top_lines = ["- Нет строк для показа."]

        lines = [
            "## AI First Look",
            f"Файлы: план `{plan_file_name}` и факт `{fact_file_name}`.",
        ]
        if period:
            lines.append(f"Фокусный период: `{period}`.")
        lines.extend(
            [
                "",
                "Коротко по картине:",
                f"- План: {self._format_money(total_plan)}",
                f"- Факт: {self._format_money(total_fact)}",
                f"- Отклонение: {self._format_money(total_variance)}",
            ]
        )
        if period and not cfo_slice.empty:
            cfo_name, cfo_variance = self._pick_top_cfo(cfo_slice)
            if cfo_name:
                lines.append(
                    f"- Самое заметное ЦФО: `{cfo_name}` с отклонением {self._format_money(cfo_variance)}."
                )
        lines.extend(
            [
                "",
                "Топ отклонений по статьям:",
                *top_lines,
                "",
                f"Факт без плана: {len(fact_without_plan)} строк(и).",
                f"План без факта: {len(plan_without_fact)} строк(и).",
            ]
        )
        if not fact_without_plan.empty:
            lines.append(self._format_missing_rows(fact_without_plan.head(3), "Факт без плана"))
        if not plan_without_fact.empty:
            lines.append(self._format_missing_rows(plan_without_fact.head(3), "План без факта"))
        lines.extend(
            [
                "",
                "Что смотреть дальше:",
                "- Если отклонение крупное, разложи его по ЦФО и статьям.",
                "- Если нужен разбор причин, уточни конкретный месяц или подразделение.",
            ]
        )
        return "\n".join(lines).strip()

    def _build_first_look_artifacts(
        self,
        *,
        plan_long: pd.DataFrame,
        fact_monthly: pd.DataFrame,
        by_cfo: pd.DataFrame,
        by_article: pd.DataFrame,
        fact_quality: dict[str, int] | None = None,
        article_matching: dict[str, Any] | None = None,
        plan_file_name: str,
        fact_file_name: str,
    ) -> list[dict[str, Any]]:
        period, cfo_slice, article_slice = self._first_look_focus_slices(
            fact_monthly=fact_monthly,
            by_cfo=by_cfo,
            by_article=by_article,
        )
        if cfo_slice.empty and article_slice.empty:
            return []

        plan_metric = self._first_plan_metric(plan_long)
        metric_type = self._metric_business_type(plan_metric)
        dashboard_payload = self._build_first_look_dashboard_payload(
            period=period,
            cfo_slice=cfo_slice,
            article_slice=article_slice,
            plan_long=plan_long,
            fact_monthly=fact_monthly,
            fact_quality=fact_quality or {},
            article_matching=article_matching or {},
            plan_file_name=plan_file_name,
            fact_file_name=fact_file_name,
        )
        artifacts: list[dict[str, Any]] = []
        artifacts.append(
            self._json_artifact(
                artifact_id="dashboard",
                title="Первичный ИИ-анализ план-факта",
                payload=dashboard_payload,
                focus_period=str(period or "all"),
                plan_file_name=plan_file_name,
                fact_file_name=fact_file_name,
            )
        )
        cfo_table_artifact = self._cfo_summary_table_artifact(
            cfo_slice,
            focus_period=str(period or "all"),
            plan_file_name=plan_file_name,
            fact_file_name=fact_file_name,
            metric_type=metric_type,
        )
        if cfo_table_artifact is not None:
            artifacts.append(cfo_table_artifact)
        article_table_artifact = self._article_summary_table_artifact(
            article_slice,
            focus_period=str(period or "all"),
            plan_file_name=plan_file_name,
            fact_file_name=fact_file_name,
            metric_type=metric_type,
        )
        if article_table_artifact is not None:
            artifacts.append(article_table_artifact)
        focus_period = period or "все периоды"
        if not article_slice.empty:
            article_chart = article_slice.copy()
            for column in ["plan_amount", "fact_amount", "variance_amount"]:
                article_chart[column] = article_chart[column].fillna(0.0)

            status_totals: dict[str, float] = {}
            for _, row in article_chart.iterrows():
                status = self._variance_status(
                    self._amount_value(row.get("plan_amount")),
                    self._amount_value(row.get("fact_amount")),
                )
                status_totals[status] = status_totals.get(status, 0.0) + abs(
                    self._amount_value(row.get("variance_amount"))
                )
            status_order = ["Превышение", "Экономия", "Факт без плана", "План без факта"]
            donut_labels = [status for status in status_order if status_totals.get(status, 0) > 0]
            donut_values = [status_totals[status] for status in donut_labels]
            if donut_values:
                artifacts.append(
                    self._plot_artifact(
                        artifact_id="variance_donut",
                        title=f"Структура отклонений · {focus_period}",
                        traces=[
                            {
                                "type": "pie",
                                "labels": donut_labels,
                                "values": donut_values,
                                "hole": 0.62,
                                "sort": False,
                                "textinfo": "percent",
                                "hovertemplate": "%{label}<br>%{value:,.0f} ₽ · %{percent}<extra></extra>",
                                "marker": {
                                    "colors": [
                                        {
                                            "Превышение": "#f43f5e",
                                            "Экономия": "#10b981",
                                            "Факт без плана": "#fb923c",
                                            "План без факта": "#60a5fa",
                                        }[status]
                                        for status in donut_labels
                                    ]
                                },
                            }
                        ],
                        layout={
                            "height": 400,
                            "margin": {"l": 20, "r": 20, "t": 54, "b": 72},
                            "showlegend": True,
                            "legend": {
                                "orientation": "h",
                                "x": 0.5,
                                "xanchor": "center",
                                "y": -0.12,
                                "yanchor": "top",
                            },
                            "annotations": [
                                {
                                    "text": "Отклонения",
                                    "showarrow": False,
                                    "font": {"size": 14},
                                }
                            ],
                        },
                        focus_period=focus_period,
                        plan_file_name=plan_file_name,
                        fact_file_name=fact_file_name,
                        board_width_units=4,
                    )
                )

            drivers = article_chart.loc[article_chart["variance_amount"].abs() > 0.01].sort_values(
                "variance_amount",
                key=lambda values: values.abs(),
                ascending=False,
            ).head(7)
            total_plan = float(article_chart["plan_amount"].sum())
            total_fact = float(article_chart["fact_amount"].sum())
            driver_values = [float(value) for value in drivers["variance_amount"].tolist()]
            other_variance = total_fact - total_plan - sum(driver_values)
            bridge_labels = ["План"]
            bridge_full_labels = ["План"]
            bridge_values = [total_plan]
            bridge_measures = ["absolute"]
            for _, row in drivers.iterrows():
                full_label = self._format_article_label(row)
                bridge_labels.append(self._department_abbreviation(self._display_value(row.get("cfo"))))
                bridge_full_labels.append(full_label)
                bridge_values.append(self._amount_value(row.get("variance_amount")))
                bridge_measures.append("relative")
            if abs(other_variance) > 0.01:
                bridge_labels.append("Прочие")
                bridge_full_labels.append("Прочие статьи")
                bridge_values.append(other_variance)
                bridge_measures.append("relative")
            bridge_labels.append("Факт")
            bridge_full_labels.append("Факт")
            bridge_values.append(0.0)
            bridge_measures.append("total")
            bridge_text = [self._format_compact_money(value, signed=index > 0) for index, value in enumerate(bridge_values)]
            bridge_text[-1] = self._format_compact_money(total_fact)
            bridge_positions = [f"step-{index}" for index in range(len(bridge_labels))]
            artifacts.append(
                self._plot_artifact(
                    artifact_id="plan_to_fact_waterfall",
                    title=f"Вклад статей в отклонение · {focus_period}",
                    traces=[
                        {
                            "type": "waterfall",
                            "orientation": "v",
                            "measure": bridge_measures,
                            "x": bridge_positions,
                            "y": bridge_values,
                            "text": bridge_text,
                            "textposition": "outside",
                            "customdata": bridge_full_labels,
                            "hovertemplate": "%{customdata}<br>%{text}<extra></extra>",
                            "connector": {"line": {"color": "#94a3b8"}},
                            "increasing": {"marker": {"color": "#f43f5e"}},
                            "decreasing": {"marker": {"color": "#10b981"}},
                            "totals": {"marker": {"color": "#3b82f6"}},
                        }
                    ],
                    layout={
                        "height": 400,
                        "margin": {"l": 72, "r": 32, "t": 54, "b": 110},
                        "showlegend": False,
                        "xaxis": {
                            "tickangle": -25,
                            "tickmode": "array",
                            "tickvals": bridge_positions,
                            "ticktext": bridge_labels,
                        },
                        "yaxis": {"title": "Сумма"},
                    },
                    focus_period=focus_period,
                    plan_file_name=plan_file_name,
                    fact_file_name=fact_file_name,
                    board_width_units=8,
                )
            )
        if not cfo_slice.empty:
            top_cfo = cfo_slice.copy().sort_values("variance_amount", key=lambda s: s.abs(), ascending=False).head(10)
            top_cfo = top_cfo.sort_values("variance_amount", ascending=True).reset_index(drop=True)
            full_labels = [self._display_value(value) for value in top_cfo["cfo"].tolist()]
            labels = [self._department_abbreviation(value) for value in full_labels]
            variances = [float(value) for value in top_cfo["variance_amount"].fillna(0).tolist()]
            artifacts.append(
                self._plot_artifact(
                    artifact_id="focus_cfo_variance",
                    title=f"Отклонение от плана по ЦФО · {focus_period}",
                    traces=[
                        {
                            "type": "bar",
                            "orientation": "h",
                            "name": "Отклонение",
                            "y": labels,
                            "x": variances,
                            "text": [self._format_compact_money(value, signed=True) for value in variances],
                            "textposition": "outside",
                            "customdata": full_labels,
                            "marker": {"color": [self._variance_color(value, metric_type=metric_type) for value in variances]},
                            "hovertemplate": "%{customdata}<br>Отклонение: %{text}<extra></extra>",
                        },
                    ],
                    layout={
                        "height": 460,
                        "margin": {"l": 170, "r": 72, "t": 64, "b": 52},
                        "xaxis": {"title": "Отклонение"},
                        "yaxis": {"autorange": "reversed"},
                        "showlegend": False,
                    },
                    focus_period=focus_period,
                    plan_file_name=plan_file_name,
                    fact_file_name=fact_file_name,
                )
            )
        if not article_slice.empty:
            top_articles = article_slice.copy().sort_values("variance_amount", key=lambda s: s.abs(), ascending=False).head(10)
            top_articles = top_articles.sort_values("variance_amount", ascending=True).reset_index(drop=True)
            labels = [self._short_label(self._article_axis_label(row), limit=46) for _, row in top_articles.iterrows()]
            full_labels = [self._format_article_label(row) for _, row in top_articles.iterrows()]
            variances = [float(value) for value in top_articles["variance_amount"].fillna(0).tolist()]
            artifacts.append(
                self._plot_artifact(
                    artifact_id="focus_article_variance",
                    title=f"Топ отклонений по статьям · {focus_period}",
                    traces=[
                        {
                            "type": "bar",
                            "orientation": "h",
                            "name": "Отклонение",
                            "y": labels,
                            "x": variances,
                            "text": [self._format_compact_money(value, signed=True) for value in variances],
                            "textposition": "outside",
                            "customdata": full_labels,
                            "marker": {"color": [self._variance_color(value, metric_type=metric_type) for value in variances]},
                            "hovertemplate": "%{customdata}<br>Отклонение: %{text}<extra></extra>",
                        }
                    ],
                    layout={
                        "height": 480,
                        "margin": {"l": 210, "r": 72, "t": 64, "b": 52},
                        "xaxis": {"title": "Отклонение"},
                        "yaxis": {"autorange": "reversed"},
                        "showlegend": False,
                    },
                    focus_period=focus_period,
                    plan_file_name=plan_file_name,
                    fact_file_name=fact_file_name,
                )
            )
        return artifacts

    def _build_first_look_dashboard_payload(
        self,
        *,
        period: str | None,
        cfo_slice: pd.DataFrame,
        article_slice: pd.DataFrame,
        plan_long: pd.DataFrame,
        fact_monthly: pd.DataFrame,
        fact_quality: dict[str, int],
        article_matching: dict[str, Any],
        plan_file_name: str,
        fact_file_name: str,
    ) -> dict[str, Any]:
        total_plan = float(cfo_slice["plan_amount"].fillna(0).sum()) if not cfo_slice.empty else float(plan_long["plan_amount"].fillna(0).sum())
        total_fact = float(cfo_slice["fact_amount"].fillna(0).sum()) if not cfo_slice.empty else float(fact_monthly["fact_amount"].fillna(0).sum())
        total_variance = total_fact - total_plan
        execution_pct = (total_fact / total_plan * 100) if total_plan else None
        plan_metric = self._first_plan_metric(plan_long)
        metric_type = self._metric_business_type(plan_metric)
        article = article_slice.copy()
        for column in ["plan_amount", "fact_amount", "variance_amount"]:
            if column in article.columns:
                article[column] = article[column].fillna(0.0)
        fact_without_plan = article.loc[(article["plan_amount"] == 0) & (article["fact_amount"] > 0)] if not article.empty else article
        plan_without_fact = article.loc[(article["fact_amount"] == 0) & (article["plan_amount"] > 0)] if not article.empty else article
        unmatched_articles = int((article["article_key"].fillna("").astype(str).str.strip() == "").sum()) if "article_key" in article.columns else 0
        unmatched_cfo = int((article["cfo"].fillna("").astype(str).str.strip() == "").sum()) if "cfo" in article.columns else 0
        cfo_overruns = int((cfo_slice["variance_amount"].fillna(0) > 0).sum()) if not cfo_slice.empty else 0
        cfo_savings = int((cfo_slice["variance_amount"].fillna(0) < 0).sum()) if not cfo_slice.empty else 0
        rows = [
            self._dashboard_table_row(row, metric_type=metric_type)
            for _, row in article.sort_values("variance_amount", key=lambda s: s.abs(), ascending=False).head(30).iterrows()
        ]
        cfo_rows = [
            self._dashboard_cfo_row(row, metric_type=metric_type)
            for _, row in cfo_slice.sort_values("variance_amount", key=lambda s: s.abs(), ascending=False).head(12).iterrows()
        ]
        cfo_name, cfo_variance = self._pick_top_cfo(cfo_slice)
        key_deviations = [row for row in rows if row["variance_amount"] > 0][:3]
        if total_variance > 0:
            primary_article = next((row for row in rows if row["variance_amount"] > 0), rows[0] if rows else None)
        elif total_variance < 0:
            primary_article = next((row for row in rows if row["variance_amount"] < 0), rows[0] if rows else None)
        else:
            primary_article = rows[0] if rows else None
        return {
            "period": period,
            "period_label": self._period_label(period),
            "plan_file_name": plan_file_name,
            "fact_file_name": fact_file_name,
            "plan_metric": plan_metric,
            "metric_type": metric_type,
            "kpi": {
                "plan": total_plan,
                "fact": total_fact,
                "variance": total_variance,
                "execution_pct": execution_pct,
                "cfo_overruns": cfo_overruns,
                "cfo_savings": cfo_savings,
                "fact_without_plan_count": int(len(fact_without_plan)),
                "plan_without_fact_count": int(len(plan_without_fact)),
                "tone": self._variance_tone(total_variance, metric_type=metric_type),
                "metric_type": metric_type,
                "plan_metric": plan_metric,
            },
            "control": {
                "fact_without_plan_count": int(len(fact_without_plan)),
                "fact_without_plan_amount": float(fact_without_plan["fact_amount"].sum()) if not fact_without_plan.empty else 0.0,
                "plan_without_fact_count": int(len(plan_without_fact)),
                "plan_without_fact_amount": float(plan_without_fact["plan_amount"].sum()) if not plan_without_fact.empty else 0.0,
                "unmatched_articles": unmatched_articles,
                "unmatched_cfo": unmatched_cfo,
                "empty_dates": int(fact_quality.get("empty_dates", 0)),
                "empty_amounts": int(fact_quality.get("empty_amounts", 0)),
            },
            "article_matching": article_matching,
            "article_total_count": int(len(article)),
            "cfo_table": cfo_rows,
            "table": rows,
            "summary": {
                "headline": self._executive_headline(total_variance, execution_pct),
                "main_driver": cfo_name,
                "main_driver_variance": cfo_variance,
                "key_deviations": key_deviations,
                "primary_article": primary_article,
                "attention": [
                    item
                    for item in [
                        "есть факт без плана" if len(fact_without_plan) else "",
                        "есть статьи с нулевым планом" if len(fact_without_plan) else "",
                        "требуется проверить сопоставление статей" if len(fact_without_plan) or len(plan_without_fact) else "",
                    ]
                    if item
                ],
            },
            "suggested_questions": [
                f"Покажи факт без плана за {period or 'период'}",
                f"Покажи план без факта за {period or 'период'}",
                "Покажи только превышения бюджета",
                "Покажи только экономию",
                "Сделай управленческую записку",
            ],
        }

    def _dashboard_table_row(self, row: pd.Series, *, metric_type: str = "expense") -> dict[str, Any]:
        plan = self._amount_value(row.get("plan_amount"))
        fact = self._amount_value(row.get("fact_amount"))
        variance = self._amount_value(row.get("variance_amount"))
        variance_pct = (variance / plan * 100) if plan else None
        return {
            "cfo": self._display_value(row.get("cfo")),
            "article": self._display_value(row.get("article") or row.get("fact_article") or row.get("plan_article")),
            "article_key": self._display_value(row.get("article_key"), fallback=""),
            "service_content": self._display_value(row.get("service_content"), fallback=""),
            "plan_counterparty": self._display_value(row.get("plan_counterparty"), fallback=""),
            "fact_counterparty": self._display_value(row.get("fact_counterparty"), fallback=""),
            "fact_contract": self._display_value(row.get("fact_contract"), fallback=""),
            "plan_amount": plan,
            "fact_amount": fact,
            "variance_amount": variance,
            "variance_pct": variance_pct,
            "status": self._variance_status(plan, fact),
            "tone": self._variance_tone(variance, metric_type=metric_type),
        }

    def _dashboard_cfo_row(self, row: pd.Series, *, metric_type: str = "expense") -> dict[str, Any]:
        plan = self._amount_value(row.get("plan_amount"))
        fact = self._amount_value(row.get("fact_amount"))
        variance = self._amount_value(row.get("variance_amount"))
        execution_pct = (fact / plan * 100) if plan else None
        return {
            "cfo": self._display_value(row.get("cfo")),
            "plan_amount": plan,
            "fact_amount": fact,
            "variance_amount": variance,
            "execution_pct": execution_pct,
            "status": self._variance_status(plan, fact),
            "tone": self._variance_tone(variance, metric_type=metric_type),
        }

    def _first_look_executive_overview_markdown(self, payload: dict[str, Any]) -> str:
        kpi = payload.get("kpi") if isinstance(payload.get("kpi"), dict) else {}
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        control = payload.get("control") if isinstance(payload.get("control"), dict) else {}
        table = payload.get("table") if isinstance(payload.get("table"), list) else []
        cfo_table = payload.get("cfo_table") if isinstance(payload.get("cfo_table"), list) else []
        article_matching = (
            payload.get("article_matching") if isinstance(payload.get("article_matching"), dict) else {}
        )
        article_matching_stats = (
            article_matching.get("stats") if isinstance(article_matching.get("stats"), dict) else {}
        )
        suggested = (
            payload.get("suggested_questions") if isinstance(payload.get("suggested_questions"), list) else []
        )
        period_label = str(payload.get("period_label") or "период")
        plan = self._amount_value(kpi.get("plan"))
        fact = self._amount_value(kpi.get("fact"))
        variance = self._amount_value(kpi.get("variance"))
        driver = str(summary.get("main_driver") or "").strip()
        driver_variance = self._amount_value(summary.get("main_driver_variance"))
        top_row = summary.get("primary_article") if isinstance(summary.get("primary_article"), dict) else {}
        if not top_row:
            top_row = table[0] if table and isinstance(table[0], dict) else {}
        top_article = str(top_row.get("article") or "").strip()
        top_article_variance = self._amount_value(top_row.get("variance_amount"))
        fact_without_plan = int(self._amount_value(control.get("fact_without_plan_count")))
        plan_without_fact = int(self._amount_value(control.get("plan_without_fact_count")))
        variance_label = self._format_compact_money(variance, signed=True)
        execution_label = self._format_percent(kpi.get("execution_pct"))
        driver_variance_label = self._format_compact_money(driver_variance, signed=True)
        top_article_variance_label = self._format_compact_money(top_article_variance, signed=True)
        auto_matched = int(self._amount_value(article_matching_stats.get("auto_matched")))
        needs_confirmation = int(self._amount_value(article_matching_stats.get("needs_confirmation")))
        dictionary_matched = int(self._amount_value(article_matching_stats.get("dictionary")))
        fuzzy_auto = int(self._amount_value(article_matching_stats.get("fuzzy_auto")))
        fuzzy_suggested = int(self._amount_value(article_matching_stats.get("fuzzy_suggested")))
        lines = [
            "## Управленческий обзор",
            "",
            f"**Краткий вывод.** Бюджет за {period_label} отклонился от плана "
            f"на **{variance_label}**; исполнение составило **{execution_label}**.",
        ]
        if driver:
            lines.append(f"Основной вклад внесло ЦФО **{driver}** ({driver_variance_label}).")
        if top_article:
            lines.append(f"Ключевая статья отклонения: **{top_article}** ({top_article_variance_label}).")
        lines.extend(
            [
                "",
                "### KPI",
                "| Показатель | Значение |",
                "|---|---:|",
                f"| План | {self._format_compact_money(plan)} |",
                f"| Факт | {self._format_compact_money(fact)} |",
                f"| Отклонение | {self._format_compact_money(variance, signed=True)} |",
                f"| Исполнение | {self._format_percent(kpi.get('execution_pct'))} |",
                f"| Факт без плана | {fact_without_plan} |",
                f"| План без факта | {plan_without_fact} |",
                f"| Статей сопоставлено автоматически | {auto_matched} |",
                f"| Требуют подтверждения | {needs_confirmation} |",
                "",
                "### Главное из детализации",
                "**По ЦФО:**",
                *self._overview_cfo_findings(cfo_table),
                "",
                "**По статьям:**",
                *self._overview_article_findings(table),
                "",
                "### Зоны внимания",
                f"- Превышение бюджета: {self._format_compact_money(variance, signed=True)}.",
                f"- Факт без плана: {fact_without_plan} строк.",
                f"- План без факта: {plan_without_fact} строк.",
                f"- Сопоставление статей: по словарю {dictionary_matched}, "
                f"fuzzy auto {fuzzy_auto}, на подтверждение {fuzzy_suggested}.",
            ]
        )
        if top_article:
            lines.append(f"- Наибольшее отклонение по статье: {top_article}.")
        if driver:
            lines.append(f"- Наибольшее отклонение по ЦФО: {driver}.")
        findings = (
            [str(item).strip() for item in summary.get("attention", []) if str(item).strip()]
            if isinstance(summary.get("attention"), list)
            else []
        )
        lines.extend(["", "### AI обнаружил"])
        if findings:
            lines.extend([f"- {item}" for item in findings[:5]])
        else:
            lines.append("- Основные отклонения требуют детализации по ЦФО и статьям.")
        lines.extend(
            [
                "",
                "### Рекомендуется проверить",
                "- операции без планового бюджета;",
                "- крупнейшие отклонения по статьям;",
                "- ЦФО с максимальным вкладом в отклонение;",
                "- плановые статьи без фактического исполнения.",
                "",
                "### Попробуйте спросить",
            ]
        )
        if suggested:
            lines.extend([f"- {question!s}" for question in suggested[:6]])
        else:
            lines.extend(
                [
                    "- Покажи факт без плана.",
                    "- Покажи только статьи с превышением бюджета.",
                    "- Объясни причины превышения бюджета.",
                    "- Сформируй управленческую записку.",
                ]
            )
        return "\n".join(lines)

    def _overview_cfo_findings(self, rows: list[Any], *, limit: int = 5) -> list[str]:
        findings: list[str] = []
        for item in rows[:limit]:
            if not isinstance(item, dict):
                continue
            cfo = str(item.get("cfo") or "").strip() or "не указано"
            variance = self._amount_value(item.get("variance_amount"))
            fact = self._amount_value(item.get("fact_amount"))
            plan = self._amount_value(item.get("plan_amount"))
            execution = self._format_percent(
                item.get("execution_pct") if item.get("execution_pct") is not None else None
            )
            status = str(item.get("status") or "").strip()
            plan_label = self._format_compact_money(plan)
            fact_label = self._format_compact_money(fact)
            variance_label = self._format_compact_money(variance, signed=True)
            findings.append(
                f"- **{cfo}**: план {plan_label}, факт {fact_label}, "
                f"отклонение {variance_label}, исполнение {execution}; статус: {status}."
            )
        return findings or ["- Существенных отклонений по ЦФО не обнаружено."]

    def _overview_article_findings(self, rows: list[Any], *, limit: int = 5) -> list[str]:
        findings: list[str] = []
        for item in rows[:limit]:
            if not isinstance(item, dict):
                continue
            cfo = str(item.get("cfo") or "").strip() or "не указано"
            article = str(item.get("article") or "").strip() or "не указано"
            service_content = str(item.get("service_content") or "").strip()
            variance = self._amount_value(item.get("variance_amount"))
            fact = self._amount_value(item.get("fact_amount"))
            plan = self._amount_value(item.get("plan_amount"))
            variance_pct = item.get("variance_pct") if item.get("variance_pct") is not None else None
            status = str(item.get("status") or "").strip()
            article_label = (
                "строка без указанной статьи"
                if self._norm_text(article) in {"не указано", "nan", "none", "null", ""}
                else f"статья **{article}**"
            )
            if service_content:
                article_label = f"{article_label} [Содержание услуги: {service_content}]"
            fact_label = self._format_compact_money(fact)
            variance_label = self._format_compact_money(variance, signed=True)
            findings.append(
                f"- {cfo}: {article_label} - план {self._format_compact_money(plan)}, "
                f"факт {fact_label}, отклонение {variance_label} "
                f"({self._format_percent(variance_pct)}); статус: {status}."
            )
        return findings or ["- Существенных отклонений по статьям не обнаружено."]

    def _first_look_story_markdown(self, payload: dict[str, Any]) -> str:
        kpi = payload.get("kpi") if isinstance(payload.get("kpi"), dict) else {}
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        control = payload.get("control") if isinstance(payload.get("control"), dict) else {}
        table = payload.get("table") if isinstance(payload.get("table"), list) else []
        period_label = str(payload.get("period_label") or "период")
        variance = self._amount_value(kpi.get("variance"))
        execution_pct = kpi.get("execution_pct")
        driver = str(summary.get("main_driver") or "").strip()
        driver_variance = self._amount_value(summary.get("main_driver_variance"))
        top_row = table[0] if table and isinstance(table[0], dict) else {}
        top_article = str(top_row.get("article") or "").strip()
        top_article_variance = self._amount_value(top_row.get("variance_amount"))
        variance_label = self._format_compact_money(variance, signed=True)
        execution_label = self._format_percent(execution_pct)
        driver_variance_label = self._format_compact_money(driver_variance, signed=True)
        top_article_variance_label = self._format_compact_money(top_article_variance, signed=True)
        lines = [
            "## Почему бюджет не выполнен?",
            "",
            f"За {period_label} факт отклонился от плана на **{variance_label}**; "
            f"исполнение бюджета составило **{execution_label}**.",
        ]
        if driver:
            lines.append(f"Основная зона внимания — **{driver}** ({driver_variance_label}).")
        if top_article:
            lines.append(f"Крупнейшая статья отклонения — **{top_article}** ({top_article_variance_label}).")
        if self._amount_value(control.get("fact_without_plan_count")) or self._amount_value(
            control.get("plan_without_fact_count")
        ):
            lines.append(
                "Дополнительно выявлены расхождения качества данных: "
                "есть факт без плана и/или плановые статьи без факта."
            )
        lines.extend(
            [
                "",
                "### Как читать обзор",
                "1. Сначала определить, где возникло отклонение бюджета.",
                "2. Затем посмотреть, какие статьи сформировали отклонение.",
                "3. После этого проверить качество данных и операции без бюджета.",
            ]
        )
        return "\n".join(lines)

    def _first_look_next_actions_markdown(self, payload: dict[str, Any]) -> str:
        period_label = str(payload.get("period_label") or "период")
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        driver = str(summary.get("main_driver") or "").strip()
        actions = [
            f"Показать факт без плана за {period_label}.",
            f"Показать план без факта за {period_label}.",
            "Показать только превышения бюджета.",
            "Показать только экономию.",
            "Сформировать управленческую записку для финансового директора.",
        ]
        if driver:
            actions.insert(0, f"Разобрать причины отклонения по ЦФО «{driver}».")
        return "\n".join(["## Что проверить дальше?", "", *[f"- {item}" for item in actions]])

    def _quality_table_artifact(
        self,
        payload: dict[str, Any],
        *,
        focus_period: str,
        plan_file_name: str,
        fact_file_name: str,
    ) -> dict[str, Any] | None:
        control = payload.get("control") if isinstance(payload.get("control"), dict) else {}
        checks = [
            (
                "Факт без плана",
                int(self._amount_value(control.get("fact_without_plan_count"))),
                self._format_compact_money(self._amount_value(control.get("fact_without_plan_amount"))),
                "Расходы прошли без планового бюджета.",
            ),
            (
                "План без факта",
                int(self._amount_value(control.get("plan_without_fact_count"))),
                self._format_compact_money(self._amount_value(control.get("plan_without_fact_amount"))),
                "Бюджет заложен, но фактического исполнения нет.",
            ),
            (
                "Статьи без сопоставления",
                int(self._amount_value(control.get("unmatched_articles"))),
                "—",
                "Требуется проверить справочник или правила сопоставления.",
            ),
            (
                "ЦФО без сопоставления",
                int(self._amount_value(control.get("unmatched_cfo"))),
                "—",
                "Есть строки без надежной привязки к подразделению.",
            ),
            (
                "Пустые даты",
                int(self._amount_value(control.get("empty_dates"))),
                "—",
                "Нельзя корректно отнести операции к периоду.",
            ),
            (
                "Пустые суммы",
                int(self._amount_value(control.get("empty_amounts"))),
                "—",
                "Нельзя корректно рассчитать факт.",
            ),
        ]
        active_checks = [list(item) for item in checks if item[1] > 0]
        if not active_checks:
            return None
        return self._table_artifact(
            artifact_id="quality_control",
            title="Какие данные требуют проверки?",
            columns=["Проверка", "Количество", "Сумма", "Что это значит"],
            rows=active_checks,
            focus_period=focus_period,
            plan_file_name=plan_file_name,
            fact_file_name=fact_file_name,
            story_order=40,
        )

    def _article_matching_table_artifact(
        self,
        payload: dict[str, Any],
        *,
        focus_period: str,
        plan_file_name: str,
        fact_file_name: str,
    ) -> dict[str, Any] | None:
        matching = (
            payload.get("article_matching") if isinstance(payload.get("article_matching"), dict) else {}
        )
        suggestions = matching.get("suggestions") if isinstance(matching.get("suggestions"), list) else []
        rows: list[list[Any]] = []
        for item in suggestions:
            if not isinstance(item, dict):
                continue
            match_type = str(item.get("match_type") or "unmatched")
            if match_type == "exact":
                continue
            confidence = self._amount_value(item.get("confidence"))
            action = (
                "Принято"
                if match_type in {"dictionary", "fuzzy_auto", "manual"}
                else "Подтвердить"
                if match_type == "fuzzy_suggested"
                else "Нет пары"
            )
            rows.append(
                [
                    str(item.get("cfo") or ""),
                    str(item.get("fact_article") or "не указано"),
                    str(item.get("plan_article") or ""),
                    self._format_percent(confidence * 100),
                    match_type,
                    action,
                ]
            )
        if not rows:
            return None
        return self._table_artifact(
            artifact_id="article_matching",
            title="Проверка сопоставления статей",
            columns=["ЦФО", "Статья факта", "Предложенная статья плана", "Уверенность", "Тип", "Действие"],
            rows=rows[:50],
            focus_period=focus_period,
            plan_file_name=plan_file_name,
            fact_file_name=fact_file_name,
            story_order=16,
        )

    def _cfo_summary_table_artifact(
        self,
        cfo_slice: pd.DataFrame,
        *,
        focus_period: str,
        plan_file_name: str,
        fact_file_name: str,
        metric_type: str,
    ) -> dict[str, Any] | None:
        if cfo_slice.empty:
            return None
        df = cfo_slice.copy()
        for column in ["plan_amount", "fact_amount", "variance_amount"]:
            if column in df.columns:
                df[column] = df[column].fillna(0.0)
        total_abs_variance = float(df["variance_amount"].abs().sum()) if "variance_amount" in df.columns else 0.0
        rows: list[list[Any]] = []
        export_rows: list[list[Any]] = []
        for _, row in df.sort_values("variance_amount", key=lambda s: s.abs(), ascending=False).iterrows():
            plan = self._amount_value(row.get("plan_amount"))
            fact = self._amount_value(row.get("fact_amount"))
            variance = self._amount_value(row.get("variance_amount"))
            execution_pct = (fact / plan * 100) if plan else None
            share_pct = (abs(variance) / total_abs_variance * 100) if total_abs_variance else 0.0
            cfo = self._display_value(row.get("cfo"))
            status = self._variance_status(plan, fact)
            priority = self._priority_label(share_pct, variance, metric_type=metric_type)
            rows.append(
                [
                    cfo,
                    self._format_compact_money(plan),
                    self._format_compact_money(fact),
                    self._format_compact_money(variance, signed=True),
                    self._format_percent(execution_pct),
                    self._format_percent(share_pct),
                    status,
                    priority,
                ]
            )
            export_rows.append([cfo, plan, fact, variance, execution_pct, share_pct, status, priority])
        return self._table_artifact(
            artifact_id="cfo_summary",
            title="План-факт по ЦФО",
            columns=["ЦФО", "План", "Факт", "Отклонение", "Исполнение", "Доля отклонения", "Статус", "Приоритет"],
            rows=rows,
            export_rows=export_rows,
            focus_period=focus_period,
            plan_file_name=plan_file_name,
            fact_file_name=fact_file_name,
            story_order=10,
        )

    def _article_summary_table_artifact(
        self,
        article_slice: pd.DataFrame,
        *,
        focus_period: str,
        plan_file_name: str,
        fact_file_name: str,
        metric_type: str,
    ) -> dict[str, Any] | None:
        if article_slice.empty:
            return None
        df = article_slice.copy()
        for column in ["plan_amount", "fact_amount", "variance_amount"]:
            if column in df.columns:
                df[column] = df[column].fillna(0.0)
        total_abs_variance = float(df["variance_amount"].abs().sum()) if "variance_amount" in df.columns else 0.0
        rows: list[list[Any]] = []
        export_rows: list[list[Any]] = []
        for _, row in df.sort_values("variance_amount", key=lambda s: s.abs(), ascending=False).iterrows():
            plan = self._amount_value(row.get("plan_amount"))
            fact = self._amount_value(row.get("fact_amount"))
            variance = self._amount_value(row.get("variance_amount"))
            variance_pct = (variance / plan * 100) if plan else None
            share_pct = (abs(variance) / total_abs_variance * 100) if total_abs_variance else 0.0
            dimensions = [
                self._display_value(row.get("cfo")),
                self._display_value(row.get("article") or row.get("fact_article") or row.get("plan_article")),
                self._display_value(row.get("service_content"), fallback=""),
                self._display_value(row.get("plan_counterparty"), fallback=""),
                self._display_value(row.get("fact_counterparty"), fallback=""),
                self._display_value(row.get("fact_contract"), fallback=""),
            ]
            status = self._variance_status(plan, fact)
            priority = self._priority_label(share_pct, variance, metric_type=metric_type)
            rows.append(
                [
                    *dimensions,
                    self._format_compact_money(plan),
                    self._format_compact_money(fact),
                    self._format_compact_money(variance, signed=True),
                    self._format_percent(variance_pct),
                    self._format_percent(share_pct),
                    status,
                    priority,
                ]
            )
            export_rows.append(
                [*dimensions, plan, fact, variance, variance_pct, share_pct, status, priority]
            )
        return self._table_artifact(
            artifact_id="article_summary",
            title="Отклонения по статьям",
            columns=["ЦФО", "Статья", "Содержание услуги", "Контрагент план", "Контрагент факт", "Договор", "План", "Факт", "Отклонение", "Отклонение, %", "Доля отклонения", "Статус", "Приоритет"],
            rows=rows,
            export_rows=export_rows,
            focus_period=focus_period,
            plan_file_name=plan_file_name,
            fact_file_name=fact_file_name,
            story_order=20,
        )

    @staticmethod
    def _variance_status(plan: float, fact: float) -> str:
        if plan == 0 and fact > 0:
            return "Факт без плана"
        if fact == 0 and plan > 0:
            return "План без факта"
        if fact > plan:
            return "Превышение"
        if fact < plan:
            return "Экономия"
        return "В плане"

    @staticmethod
    def _variance_tone(value: float, *, metric_type: str = "expense") -> str:
        if value > 0:
            return "saving" if PlanfactSourceService._positive_variance_is_good(metric_type) else "warning"
        if value < 0:
            return "warning" if PlanfactSourceService._positive_variance_is_good(metric_type) else "saving"
        return "neutral"

    @staticmethod
    def _priority_label(share_pct: float, variance: float, *, metric_type: str = "expense") -> str:
        tone = PlanfactSourceService._variance_tone(variance, metric_type=metric_type)
        if share_pct >= 20 or (share_pct >= 10 and tone == "warning"):
            return "Высокий"
        if share_pct >= 5:
            return "Средний"
        return "Низкий"

    @staticmethod
    def _first_plan_metric(plan_long: pd.DataFrame) -> str:
        if plan_long.empty or "plan_metric" not in plan_long.columns:
            return ""
        values = plan_long["plan_metric"].dropna().astype(str).str.strip()
        values = values.loc[values != ""]
        return str(values.iloc[0]) if not values.empty else ""

    @staticmethod
    def _metric_business_type(metric: str) -> str:
        normalized = PlanfactSourceService._norm_text(metric)
        if any(token in normalized for token in ("revenue", "sales", "income", "выруч", "доход")):
            return "revenue"
        if any(token in normalized for token in ("cash_in", "net_cash", "поступ", "приток")):
            return "revenue"
        if any(
            token in normalized
            for token in ("expense", "cost", "opex", "capex", "расход", "затрат", "pl", "cf")
        ):
            return "expense"
        return "expense"

    @staticmethod
    def _positive_variance_is_good(metric_type: str) -> bool:
        normalized = PlanfactSourceService._norm_text(metric_type)
        return any(
            token in normalized
            for token in (
                "revenue",
                "sales",
                "income",
                "profit",
                "margin",
                "cash_in",
                "net_cash",
                "выруч",
                "доход",
                "прибыл",
            )
        )

    @staticmethod
    def _variance_color(value: float, *, metric_type: str = "expense") -> str:
        if value > 0:
            return "#16a34a" if PlanfactSourceService._positive_variance_is_good(metric_type) else "#dc2626"
        if value < 0:
            return "#dc2626" if PlanfactSourceService._positive_variance_is_good(metric_type) else "#16a34a"
        return "#64748b"

    @staticmethod
    def _short_label(value: str, *, limit: int = 42) -> str:
        clean = str(value or "").strip()
        if len(clean) <= limit:
            return clean
        return f"{clean[: max(0, limit - 1)].rstrip()}…"

    @staticmethod
    def _department_abbreviation(value: str) -> str:
        clean = str(value or "").strip()
        words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", clean)
        stop_words = {"и", "в", "во", "для", "на", "по", "с", "со"}
        significant = [word for word in words if word.lower() not in stop_words]
        if len(significant) < 2 or len(clean) <= 12:
            return PlanfactSourceService._short_label(clean, limit=42)
        abbreviation = "".join(word[0] for word in significant).upper()
        return abbreviation if len(abbreviation) >= 2 else PlanfactSourceService._short_label(clean, limit=42)

    @staticmethod
    def _article_axis_label(row: pd.Series) -> str:
        article = PlanfactSourceService._display_value(row.get("article") or row.get("fact_article") or row.get("plan_article"))
        article_key = PlanfactSourceService._display_value(row.get("article_key"), fallback="")
        return f"{article_key} · {article}" if article_key else article

    @staticmethod
    def _period_label(period: str | None) -> str:
        if not period:
            return "все периоды"
        months = {
            "01": "январь",
            "02": "февраль",
            "03": "март",
            "04": "апрель",
            "05": "май",
            "06": "июнь",
            "07": "июль",
            "08": "август",
            "09": "сентябрь",
            "10": "октябрь",
            "11": "ноябрь",
            "12": "декабрь",
        }
        year, _, month = period.partition("-")
        return f"{months.get(month, period)} {year}" if month else period

    def _executive_headline(self, variance: float, execution_pct: float | None) -> str:
        execution_label = self._format_percent(execution_pct)
        if variance > 0:
            variance_label = self._format_compact_money(variance, signed=True)
            return f"Факт превысил план на {variance_label}, исполнение составило {execution_label}."
        if variance < 0:
            variance_label = self._format_compact_money(abs(variance))
            return f"Факт ниже плана на {variance_label}, исполнение составило {execution_label}."
        return f"Факт совпал с планом, исполнение составило {execution_label}."

    def _first_look_focus_slices(
        self,
        *,
        fact_monthly: pd.DataFrame,
        by_cfo: pd.DataFrame,
        by_article: pd.DataFrame,
    ) -> tuple[str | None, pd.DataFrame, pd.DataFrame]:
        period = None
        if not fact_monthly.empty and "period" in fact_monthly.columns:
            periods = [
                str(value) for value in fact_monthly["period"].dropna().astype(str) if str(value).strip()
            ]
            if periods:
                period = sorted(periods)[-1]
        if period is None and not by_cfo.empty:
            periods = [str(value) for value in by_cfo["period"].dropna().astype(str) if str(value).strip()]
            if periods:
                period = sorted(periods)[-1]

        if period is not None:
            cfo_slice = by_cfo.loc[by_cfo["period"] == period].copy()
            article_slice = by_article.loc[by_article["period"] == period].copy()
        else:
            cfo_slice = by_cfo.copy()
            article_slice = by_article.copy()

        cfo_slice = cfo_slice.sort_values("variance_amount", key=lambda s: s.abs(), ascending=False)
        article_slice = article_slice.sort_values("variance_amount", key=lambda s: s.abs(), ascending=False)
        return period, cfo_slice, article_slice

    @staticmethod
    def _format_top_variance_lines(df: pd.DataFrame) -> list[str]:
        lines: list[str] = []
        for _, row in df.iterrows():
            cfo = PlanfactSourceService._display_value(row.get("cfo"))
            article = PlanfactSourceService._display_value(
                row.get("article") or row.get("fact_article") or row.get("plan_article")
            )
            plan_amount = PlanfactSourceService._amount_value(row.get("plan_amount"))
            fact_amount = PlanfactSourceService._amount_value(row.get("fact_amount"))
            variance = PlanfactSourceService._amount_value(row.get("variance_amount"))
            lines.append(
                f"- `{cfo}` · `{article}`{PlanfactSourceService._service_content_suffix(row)}: "
                f"план {PlanfactSourceService._format_money(plan_amount)}, "
                f"факт {PlanfactSourceService._format_money(fact_amount)}, "
                f"отклонение {PlanfactSourceService._format_money(variance)}"
            )
        return lines

    @staticmethod
    def _pick_top_cfo(df: pd.DataFrame) -> tuple[str | None, float]:
        if df.empty:
            return None, 0.0
        idx = (df["variance_amount"].fillna(0).abs()).idxmax()
        row = df.loc[idx]
        cfo = PlanfactSourceService._display_value(row.get("cfo"))
        variance = PlanfactSourceService._amount_value(row.get("variance_amount"))
        return cfo, variance

    @staticmethod
    def _format_missing_rows(df: pd.DataFrame, header: str) -> str:
        items = []
        for _, row in df.iterrows():
            cfo = PlanfactSourceService._display_value(row.get("cfo"))
            article = PlanfactSourceService._display_value(
                row.get("article") or row.get("fact_article") or row.get("plan_article")
            )
            amount_source = row.get("fact_amount")
            if pd.isna(amount_source) or float(amount_source or 0) == 0:
                amount_source = row.get("plan_amount")
            amount = PlanfactSourceService._amount_value(amount_source)
            items.append(
                f"- {header}: `{cfo}` · `{article}`{PlanfactSourceService._service_content_suffix(row)} "
                f"· {PlanfactSourceService._format_money(amount)}"
            )
        return "\n".join(items)

    @staticmethod
    def _format_money(value: float) -> str:
        return f"{value:,.0f}".replace(",", " ")

    @staticmethod
    def _format_compact_money(value: float, *, signed: bool = False) -> str:
        sign = ""
        if signed and value > 0:
            sign = "+"
        abs_value = abs(value)
        if abs_value >= 1_000_000:
            amount = value / 1_000_000
            return f"{sign}{amount:.1f}".replace(".", ",") + " млн ₽"
        amount = value / 1_000
        return f"{sign}{amount:.1f}".replace(".", ",") + " тыс. ₽"

    @staticmethod
    def _format_percent(value: float | None) -> str:
        if value is None or pd.isna(value):
            return "н/д"
        return f"{value:.1f}".replace(".", ",") + "%"

    @staticmethod
    def _display_value(value: object, fallback: str = "не указано") -> str:
        text = PlanfactSourceService._clean_value(value)
        if not text:
            return fallback
        if text.lower() in {"nan", "none", "null"}:
            return fallback
        return text

    @staticmethod
    def _format_article_label(row: pd.Series) -> str:
        cfo = PlanfactSourceService._display_value(row.get("cfo"))
        article = PlanfactSourceService._display_value(
            row.get("article") or row.get("fact_article") or row.get("plan_article")
        )
        return f"{cfo} · {article}{PlanfactSourceService._detail_suffix(row)}"

    @staticmethod
    def _detail_suffix(row: Any) -> str:
        details = [
            ("Содержание услуги", row.get("service_content")),
            ("Контрагент план", row.get("plan_counterparty")),
            ("Контрагент факт", row.get("fact_counterparty")),
            ("Договор", row.get("fact_contract")),
        ]
        text = "; ".join(
            f"{label}: {value}"
            for label, raw_value in details
            if (value := PlanfactSourceService._display_value(raw_value, fallback=""))
        )
        return f" [{text}]" if text else ""

    @staticmethod
    def _service_content_suffix(row: pd.Series) -> str:
        service_content = PlanfactSourceService._display_value(row.get("service_content"), fallback="")
        if not service_content:
            return ""
        return f" [Содержание услуги: {service_content}]"

    @staticmethod
    def _collapse_text_values(values: pd.Series) -> str:
        seen: list[str] = []
        for value in values.tolist():
            text = PlanfactSourceService._clean_value(value)
            if not text or text.lower() in {"nan", "none", "null"}:
                continue
            if text not in seen:
                seen.append(text)
            if len(seen) >= 3:
                break
        return "; ".join(seen)

    @staticmethod
    def _collect_source_row_ids(values: pd.Series) -> str:
        result: list[int] = []
        for value in values.tolist():
            if isinstance(value, str):
                candidates = value.split(",")
            else:
                candidates = value if isinstance(value, list | tuple | set) else [value]
            for candidate in candidates:
                try:
                    row_id = int(candidate)
                except (TypeError, ValueError):
                    continue
                if row_id not in result:
                    result.append(row_id)
        return ",".join(str(row_id) for row_id in result)

    @staticmethod
    def _amount_value(value: object) -> float:
        if value is None:
            return 0.0
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return 0.0
        if pd.isna(amount):
            return 0.0
        return amount

    def _plot_artifact(
        self,
        *,
        artifact_id: str,
        title: str,
        traces: list[dict[str, Any]],
        layout: dict[str, Any],
        focus_period: str,
        plan_file_name: str,
        fact_file_name: str,
        board_width_units: int = 12,
    ) -> dict[str, Any]:
        story_title_by_id = {
            "variance_donut": "Структура отклонений",
            "plan_to_fact_waterfall": "Вклад статей в отклонение",
            "focus_cfo_variance": "Где возникло отклонение бюджета?",
            "focus_article_variance": "Какие статьи сформировали отклонение?",
        }
        story_insight_by_id = {
            "variance_donut": "Кольцо показывает структуру абсолютного отклонения по типам: превышение, экономия, факт без плана и план без факта.",
            "plan_to_fact_waterfall": "График начинается с плана, показывает вклад крупнейших статей и заканчивается итоговым фактом.",
            "focus_cfo_variance": "График показывает, какие ЦФО сформировали основное отклонение. Сортировка выполнена по абсолютному влиянию на бюджет.",
            "focus_article_variance": "График показывает статьи, которые дали максимальный вклад в отклонение бюджета. Первые строки требуют приоритетной проверки.",
        }
        story_actions_by_id = {
            "variance_donut": ["Показать состав", "Открыть статьи"],
            "plan_to_fact_waterfall": ["Разобрать вклад", "Показать операции"],
            "focus_cfo_variance": ["Подробнее", "Объяснить причины", "Показать детализацию"],
            "focus_article_variance": ["Исследовать отклонение", "Показать операции", "Сформировать вывод"],
        }
        story_order_by_id = {
            "variance_donut": 30,
            "plan_to_fact_waterfall": 40,
            "focus_cfo_variance": 50,
            "focus_article_variance": 60,
        }
        resolved_title = story_title_by_id.get(artifact_id, title)
        return make_json_safe(
            {
                "id": f"planfact_first_look_{artifact_id}",
                "type": "plot",
                "text": resolved_title,
                "role": "ai",
                "meta": {
                    "producer_tool": "planfact_first_look",
                    "source_type": "planfact",
                    "report_kind": "chart",
                    "full_width": board_width_units == 12,
                    "board_width_units": board_width_units,
                    "story_order": story_order_by_id.get(artifact_id, 100),
                    "insight": story_insight_by_id.get(artifact_id),
                    "suggested_actions": story_actions_by_id.get(artifact_id, []),
                    "focus_period": focus_period,
                    "plan_file_name": plan_file_name,
                    "fact_file_name": fact_file_name,
                },
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "data": {
                    "format": "plotly-json",
                    "data": {
                        "data": traces,
                        "layout": {
                            "title": {"text": resolved_title},
                            "margin": {"l": 140, "r": 24, "t": 64, "b": 48},
                            "height": 380,
                            "showlegend": True,
                            **layout,
                        },
                    },
                },
            }
        )

    def _note_artifact(
        self,
        *,
        artifact_id: str,
        title: str,
        content: str,
        focus_period: str,
        plan_file_name: str,
        fact_file_name: str,
        story_order: int,
    ) -> dict[str, Any]:
        return make_json_safe(
            {
                "id": f"planfact_first_look_{artifact_id}",
                "type": "note",
                "text": title,
                "role": "ai",
                "meta": {
                    "producer_tool": "planfact_first_look",
                    "source_type": "planfact",
                    "report_kind": "story",
                    "story_order": story_order,
                    "focus_period": focus_period,
                    "plan_file_name": plan_file_name,
                    "fact_file_name": fact_file_name,
                },
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "data": {"format": "markdown", "data": {"content": content}},
            }
        )

    def _table_artifact(
        self,
        *,
        artifact_id: str,
        title: str,
        columns: list[str],
        rows: list[list[Any]],
        export_rows: list[list[Any]] | None = None,
        focus_period: str,
        plan_file_name: str,
        fact_file_name: str,
        story_order: int,
    ) -> dict[str, Any]:
        return make_json_safe(
            {
                "id": f"planfact_first_look_{artifact_id}",
                "type": "table",
                "text": title,
                "role": "ai",
                "meta": {
                    "producer_tool": "planfact_first_look",
                    "source_type": "planfact",
                    "report_kind": "quality_control",
                    "full_width": True,
                    "story_order": story_order,
                    "focus_period": focus_period,
                    "plan_file_name": plan_file_name,
                    "fact_file_name": fact_file_name,
                },
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "data": {
                    "format": "split",
                    "data": {
                        "columns": columns,
                        "index": list(range(1, len(rows) + 1)),
                        "data": rows,
                    },
                    "export_data": {
                        "columns": columns,
                        "data": export_rows,
                    }
                    if export_rows is not None
                    else None,
                },
            }
        )

    def _json_artifact(
        self,
        *,
        artifact_id: str,
        title: str,
        payload: dict[str, Any],
        focus_period: str,
        plan_file_name: str,
        fact_file_name: str,
    ) -> dict[str, Any]:
        return make_json_safe(
            {
                "id": f"planfact_first_look_{artifact_id}",
                "type": "json",
                "text": title,
                "role": "ai",
                "meta": {
                    "producer_tool": "planfact_first_look",
                    "source_type": "planfact",
                    "report_kind": "dashboard",
                    "focus_period": focus_period,
                    "plan_file_name": plan_file_name,
                    "fact_file_name": fact_file_name,
                },
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "data": {"format": "json", "data": payload},
            }
        )

    @staticmethod
    def _with_variance(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["plan_amount"] = out["plan_amount"].fillna(0.0)
        out["fact_amount"] = out["fact_amount"].fillna(0.0)
        out["variance_amount"] = out["fact_amount"] - out["plan_amount"]
        out["variance_pct"] = out["variance_amount"] / out["plan_amount"].where(out["plan_amount"] != 0)
        return out

    @staticmethod
    def normalize_article_key(value: object) -> str:
        text = PlanfactSourceService._clean_value(value).lower()
        text = text.replace("ё", "е")
        if text in {"nan", "none", "null"}:
            return ""
        numeric_code = re.fullmatch(r"(?:pl)?(\d{8,}(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if numeric_code:
            return numeric_code.group(1)
        text = re.sub(r"^cf\d{4,}\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\d{4,}\s*(?:[|:;,\-–—]\s*)?", "", text)
        text = re.sub(r"\s*(?:[|:;,\-–—]\s*)?\d{4,}\s*$", "", text)
        text = re.sub(r"\s*\(\d{4,}\)\s*$", "", text)
        text = re.sub(r"[|:;,\-–—()\[\]{}]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        for phrase in _ARTICLE_STOP_PHRASES:
            text = re.sub(rf"\b{re.escape(phrase)}\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        for source, target in _ARTICLE_ABBREVIATIONS.items():
            text = re.sub(rf"\b{re.escape(source)}\b", target, text)
        text = re.sub(r"\bподдержке\b", "поддержка", text)
        text = re.sub(r"\bразработке\b", "разработка", text)
        text = re.sub(r"\bзаказной\b", "заказная", text)
        text = re.sub(r"\bоблачным\b", "облачные", text)
        text = re.sub(r"\bсервисам\b", "сервисы", text)
        text = re.sub(r"\bконсалтингу\b", "консалтинг", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = _ARTICLE_DICTIONARY.get(text, text)
        return text

    @staticmethod
    def normalize_pl_article_key(value: object) -> str:
        return PlanfactSourceService.extract_article_code(value)

    @staticmethod
    def extract_article_code(value: object) -> str:
        text = PlanfactSourceService._clean_value(value).lower()
        code = r"(\d{8,}(?:\.\d+)?)"
        patterns = [
            rf"^(?:pl)?{code}(?=\s*(?:[|:;,–—-]|$))",
            rf"\(\s*(?:pl)?{code}\s*\)\s*$",
            rf"^(?:pl)?{code}$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    @staticmethod
    def normalize_match_key(value: object) -> str:
        text = PlanfactSourceService._clean_value(value).lower()
        if text in {"nan", "none", "null"}:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _clean_value(value: object) -> str:
        if pd.isna(value):
            return ""
        return re.sub(r"\s+", " ", str(value).strip())

    @staticmethod
    def _to_number(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce").fillna(0.0)
        cleaned = (
            series.astype(str)
            .str.replace("\u00a0", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)

    @staticmethod
    def _parse_dates(series: pd.Series) -> pd.Series:
        as_text = series.dropna().astype(str).str.strip()
        iso_like = as_text.str.match(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}").sum()
        if iso_like >= max(1, len(as_text) // 2):
            return pd.to_datetime(series, errors="coerce", dayfirst=False)
        return pd.to_datetime(series, errors="coerce", dayfirst=True)

    @staticmethod
    def _required_column(df: pd.DataFrame, column: object, label: str) -> str:
        clean = str(column or "").strip()
        if not clean or clean not in df.columns:
            raise PlanfactSourceError(f"Column '{label}' is not configured or not found: {clean}")
        return clean

    @staticmethod
    def _optional_column(df: pd.DataFrame, column: object, label: str) -> str | None:
        clean = str(column or "").strip()
        if not clean:
            return None
        if clean not in df.columns:
            raise PlanfactSourceError(f"Column '{label}' was configured but not found: {clean}")
        return clean

    def _persist_manifest(
        self,
        session_id: str,
        config: dict[str, Any],
        tables: dict[str, pd.DataFrame],
        *,
        user_id: int | None = None,
    ) -> None:
        manifest = self._manifest_store.load(session_id)
        original = SessionManifest.from_dict(manifest.to_dict())
        source_dir = self._session_dir(session_id) / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        snapshot_blob_ids: list[str] = []
        try:
            parquet_rel = "sources/planfact_plan_long.parquet"
            parquet_abs = self._session_dir(session_id) / parquet_rel
            tables["planfact_plan_long"].to_parquet(parquet_abs, engine="pyarrow")
            written.append(parquet_abs)
            table_paths: dict[str, str] = {}
            table_dir = source_dir / "planfact_tables"
            table_dir.mkdir(parents=True, exist_ok=True)
            for table_name, df in tables.items():
                table_rel = f"sources/planfact_tables/{table_name}.parquet"
                table_abs = self._session_dir(session_id) / table_rel
                df.to_parquet(table_abs, engine="pyarrow")
                written.append(table_abs)
                table_paths[table_name] = table_rel
            table_blob_ids: dict[str, str] = {}
            if self._blob_store is not None:
                if user_id is None:
                    raise PlanfactSourceError("user_id is required for durable planfact snapshots")
                snapshot_blob_ids = self._blob_store.put_many(
                    user_id=user_id,
                    session_id=session_id,
                    kind="runtime_snapshot",
                    items=[
                        BlobWrite(
                            logical_name=f"{table_name}.parquet",
                            media_type="application/vnd.apache.parquet",
                            content=(self._session_dir(session_id) / table_rel).read_bytes(),
                            metadata={"table_name": table_name},
                        )
                        for table_name, table_rel in table_paths.items()
                    ],
                )
                table_blob_ids = dict(zip(table_paths, snapshot_blob_ids, strict=True))
            source = SessionSource(
                alias="planfact",
                source_type="planfact",
                display_name="План-факт",
                variable_name="planfact_data",
                file_name=f"{config['plan'].get('file_name')} + {config['fact'].get('file_name')}",
                parquet_path=parquet_rel,
                csv_session_id=session_id,
                csv_table_names=list(tables.keys()),
                schema_hint={name: f"{len(df)} rows" for name, df in tables.items()},
                preprocessing_summary={
                    "planfact_config": config,
                    "duckdb_table_paths": table_paths,
                    "duckdb_table_blob_ids": table_blob_ids,
                },
                row_count=len(tables["planfact_by_cfo_article_period"]),
                column_count=len(tables["planfact_by_cfo_article_period"].columns),
            )
            manifest.add_source(source)
            self._manifest_store.save(session_id, manifest)
        except Exception:
            self._manifest_store.save(session_id, original)
            if self._blob_store is not None and user_id is not None:
                self._blob_store.delete_many(user_id=user_id, blob_ids=snapshot_blob_ids)
            for path in written:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            raise

    def _normalize_config(self, config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise PlanfactSourceError("Config must be an object")
        out = json.loads(json.dumps(config, ensure_ascii=False))
        if out.get("source_type") != "planfact":
            out["source_type"] = "planfact"
        out.setdefault("plan", {})
        out.setdefault("fact", {})
        out.setdefault("join", {})
        return out

    @staticmethod
    def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        out = dict(base)
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = PlanfactSourceService._deep_merge(out[key], value)
            else:
                out[key] = value
        return out

    @staticmethod
    def _norm_text(value: object) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _session_dir(self, session_id: str) -> Path:
        return self._storage_dir / "sessions" / session_id

    def _planfact_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "planfact"

    def _config_path(self, session_id: str) -> Path:
        return self._planfact_dir(session_id) / "planfact_config.json"

    def _write_config(
        self,
        session_id: str,
        config: dict[str, Any],
        *,
        name: str = "planfact_config.json",
        user_id: int | None = None,
    ) -> None:
        path = self._planfact_dir(session_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        if self._blob_store is not None and user_id is not None:
            self._blob_store.put_many(
                user_id=user_id,
                session_id=session_id,
                kind="planfact_config",
                items=[
                    BlobWrite(
                        logical_name=name,
                        media_type="application/json",
                        content=json.dumps(config, ensure_ascii=False).encode("utf-8"),
                    )
                ],
            )

    def _write_pending_file(
        self, session_id: str, kind: PlanfactFileKind, file_name: str, content: bytes
    ) -> None:
        folder = self._planfact_dir(session_id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{kind}.bin").write_bytes(content)
        (folder / f"{kind}.name").write_text(file_name, encoding="utf-8")

    def _read_pending_file(self, session_id: str, kind: PlanfactFileKind) -> tuple[str, bytes]:
        folder = self._planfact_dir(session_id)
        content_path = folder / f"{kind}.bin"
        name_path = folder / f"{kind}.name"
        if not content_path.exists() or not name_path.exists():
            blob = (
                self._blob_store.get_latest_for_session(
                    session_id=session_id,
                    kind=f"planfact_{kind}",
                )
                if self._blob_store is not None
                else None
            )
            if blob is None:
                raise PlanfactSourceError("Run planfact detect before confirm")
            return blob.logical_name, blob.content
        return name_path.read_text(encoding="utf-8").strip(), content_path.read_bytes()
