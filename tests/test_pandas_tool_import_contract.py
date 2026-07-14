from __future__ import annotations

import pandas as pd

from backend.tools.impl.pandas_tool import PandasTool
from backend.tools.impl.plotly_tool import PlotlyTool
from backend.tools.sandbox import SAFE_BUILTINS, SessionSandbox


def test_pandas_tool_allows_regex_and_datetime_for_key_normalization() -> None:
    df = pd.DataFrame(
        {
            "raw_cf": ["CF02080000 - revenue", "no code"],
            "doc_date": ["01.03.2026", "15.03.2026"],
        }
    )
    sandbox = SessionSandbox()
    sandbox.bind_dataframe(df)
    tool = PandasTool(
        df,
        sandbox=sandbox,
        execution_timeout_sec=5.0,
        tool_cache_size=0,
    )

    text, payload = tool._run(
        """
import re
import datetime

result = df.copy()
result["cf_code"] = result["raw_cf"].astype(str).apply(
    lambda value: (
        re.search(r"(CF\\d+)", value).group(1)
        if re.search(r"(CF\\d+)", value)
        else None
    )
)
result["month"] = result["doc_date"].apply(
    lambda value: datetime.datetime.strptime(value, "%d.%m.%Y").month
)

tool_result = {"normalized_keys": result[["cf_code", "month"]]}
""".strip()
    )

    assert "normalized_keys" in text
    normalized = payload["table"]["normalized_keys"]
    assert normalized.loc[0, "cf_code"] == "CF02080000"
    assert pd.isna(normalized.loc[1, "cf_code"])
    assert normalized["month"].tolist() == [3, 3]


def test_pandas_tool_still_rejects_system_imports() -> None:
    tool = PandasTool(pd.DataFrame({"a": [1]}), tool_cache_size=0)

    valid, message = tool.validate_libraries("import os\nresult = 1")

    assert not valid
    assert "os" in message


def test_pandas_tool_uses_last_assigned_dataframe_when_tool_result_missing() -> None:
    df = pd.DataFrame({"segment": ["A", "A", "B"], "revenue": [10, 20, 5]})
    sandbox = SessionSandbox()
    sandbox.bind_dataframe(df)
    tool = PandasTool(df, sandbox=sandbox, tool_cache_size=0)

    text, payload = tool._run(
        """
summary = df.groupby("segment", as_index=False)["revenue"].sum()
""".strip()
    )

    assert "summary" in text
    assert "# pandas_tool inferred `tool_result` from result variable(s): `summary`" in text
    result = payload["table"]["summary"]
    assert result.to_dict(orient="records") == [
        {"segment": "A", "revenue": 30},
        {"segment": "B", "revenue": 5},
    ]


def test_pandas_tool_wraps_print_output_when_tool_result_missing() -> None:
    df = pd.DataFrame({"segment": ["A"], "revenue": [10]})
    sandbox = SessionSandbox()
    sandbox.bind_dataframe(df)
    tool = PandasTool(df, sandbox=sandbox, tool_cache_size=0)

    text, payload = tool._run("print(df.columns.tolist())")

    assert "output" in text
    assert "# pandas_tool inferred `tool_result` from printed stdout" in text
    output = payload["table"]["output"]
    assert output["output"].tolist() == ["['segment', 'revenue']"]


def test_exec_tools_reject_namespace_introspection_names_before_execution() -> None:
    df = pd.DataFrame({"a": [1]})
    tools = [
        PandasTool(df, tool_cache_size=0),
        PlotlyTool(df, tool_cache_size=0),
    ]

    for tool in tools:
        valid_globals, message_globals = tool.validate_code_patterns(
            "available = globals\nresult = df.copy()"
        )
        valid_locals, message_locals = tool.validate_code_patterns(
            "available = locals\nresult = df.copy()"
        )

        assert not valid_globals
        assert not valid_locals
        assert "Python" in message_globals
        assert "Python" in message_locals


def test_session_sandbox_does_not_expose_globals_builtin() -> None:
    assert "globals" not in SAFE_BUILTINS
