from __future__ import annotations

import pandas as pd

from backend.agent.services.runtime_context import build_chat_data_context
from backend.tools.capabilities import (
    RuntimeTableDescriptorPromptOptions,
    build_runtime_capability_context,
    coerce_runtime_table_descriptors,
    format_runtime_table_descriptors,
)


def test_rag_runtime_context_uses_knowledge_base_mode() -> None:
    context = build_runtime_capability_context(
        available_tool_keys=["rag_tool"],
        has_dataframe=False,
        has_db_source=False,
        has_knowledge_base=True,
    )

    assert context["source_mode"] == "knowledge_base"
    assert "knowledge_base_search" in context["available_capability_keys"]
    assert "knowledge_base_search" not in context["unavailable_capability_keys"]
    assert "none" not in context["prompt_block"]


def test_rag_session_source_adds_prompt_context() -> None:
    text = build_chat_data_context(
        None,
        {
            "source_type": "rag",
            "source_label": "База знаний",
            "source_mode": "lightrag",
        },
    )

    assert "База знаний" in text
    assert "rag_tool" in text


def test_runtime_capability_context_includes_uploaded_table_descriptors() -> None:
    context = build_runtime_capability_context(
        available_tool_keys=["sql_tool"],
        has_dataframe=False,
        has_db_source=True,
        csv_table_names=["orders"],
        csv_table_descriptors=[
            {
                "table_name": "orders",
                "qualified_name": "orders",
                "columns": ["customer_id", "amount"],
                "file_name": "orders.csv",
                "display_name": "Orders upload",
                "source_alias": "orders_csv",
                "row_count": 42,
                "preprocessing_summary": {"header_row_index": 1},
            }
        ],
    )

    prompt_block = context["prompt_block"]

    assert "orders.csv" in prompt_block
    assert "orders_csv" in prompt_block
    assert "customer_id" in prompt_block
    assert "amount" in prompt_block
    assert "42" in prompt_block


def test_runtime_capability_context_prompts_catalog_first_for_multiple_tables() -> None:
    context = build_runtime_capability_context(
        available_tool_keys=["data_catalog_tool", "sql_tool"],
        has_dataframe=False,
        has_db_source=True,
        source_table_count=2,
        source_count=2,
    )

    prompt_block = context["prompt_block"]

    assert "CATALOG-FIRST" in prompt_block
    assert "data_catalog_tool" in prompt_block
    assert "qualified_name" in prompt_block


def test_chat_data_context_describes_uploaded_files_individually() -> None:
    text = build_chat_data_context(
        pd.DataFrame({"first_file_only": [1]}),
        {
            "source_type": "csv",
            "source_label": "orders.csv, customers.csv",
            "csv_loaded": True,
            "csv_table_names": ["orders", "customers"],
            "csv_table_descriptors": [
                {
                    "table_name": "orders",
                    "qualified_name": "orders",
                    "columns": ["order_id", "amount"],
                    "file_name": "orders.csv",
                    "display_name": "Orders upload",
                    "row_count": 11,
                    "column_count": 2,
                },
                {
                    "table_name": "customers",
                    "qualified_name": "customers",
                    "columns": ["customer_id", "customer_name"],
                    "file_name": "customers.csv",
                    "display_name": "Customers upload",
                    "row_count": 22,
                    "column_count": 2,
                },
            ],
        },
    )

    assert "orders.csv" in text
    assert "customers.csv" in text
    assert "`orders`" in text
    assert "`customers`" in text
    assert "order_id" in text
    assert "customer_name" in text
    assert "11" in text
    assert "22" in text
    assert "first_file_only" not in text


def test_runtime_table_descriptor_formatter_deduplicates_sources_and_omits_invalid_items() -> None:
    descriptors = coerce_runtime_table_descriptors(
        [
            {
                "table_name": "orders",
                "qualified_name": "main.orders",
                "columns": ["order_id", "amount", "customer_id"],
                "display_name": "orders.csv",
                "file_name": "orders.csv",
                "source_alias": "orders_csv",
                "row_count": 12,
                "column_count": 3,
            },
            {"table_name": "   ", "columns": ["ignored"]},
            "not-a-descriptor",
        ]
    )

    lines = format_runtime_table_descriptors(
        descriptors,
        RuntimeTableDescriptorPromptOptions(
            header="- tables:",
            table_template="  - `{table_name}` sources={sources}; stats={stats}; columns={columns}",
            hidden_tables_template="  - hidden={hidden_tables}",
            unknown_columns_label="unknown",
            rows_label="rows",
            columns_label="columns",
            column_overflow_template="+{hidden_columns} more",
            max_columns=2,
        ),
    )

    assert descriptors[0].table_name == "orders"
    assert len(descriptors) == 1
    assert lines == [
        "- tables:",
        "  - `main.orders` sources=`orders.csv`, `orders_csv`; stats=12 rows, 3 columns; "
        "columns=`order_id`, `amount`, +1 more",
    ]
