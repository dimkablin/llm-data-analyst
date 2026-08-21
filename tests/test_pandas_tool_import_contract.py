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


def test_pandas_tool_keeps_print_output_with_explicit_tool_result() -> None:
    df = pd.DataFrame({"segment": ["A"], "revenue": [10]})
    sandbox = SessionSandbox()
    sandbox.bind_dataframe(df)
    tool = PandasTool(df, sandbox=sandbox, tool_cache_size=0)

    text, payload = tool._run(
        "print('rows=1')\n"
        "tool_result = {"
        "'schema_version': '1.0', "
        "'artifact_type': 'table', "
        "'items': {'summary': df.copy()}"
        "}"
    )

    assert "rows=1" in text
    assert payload["table"]["summary"].equals(df)


def test_pandas_tool_keeps_print_output_when_execution_fails() -> None:
    df = pd.DataFrame({"month": ["2024-05"]})
    sandbox = SessionSandbox()
    sandbox.bind_dataframe(df)
    sandbox.put("monthly", df)
    tool = PandasTool(df, sandbox=sandbox, tool_cache_size=0)

    text, payload = tool._run(
        "print('available months: 2024-05')\n"
        "monthly.set_index('month').loc['2024-08']"
    )

    assert payload["status"] == "error"
    assert "KeyError: 2024-08" in text
    assert "STDOUT_FOR_LLM_CONTEXT:\navailable months: 2024-05" in text
    assert "observed_values:" in text
    assert "month: ['2024-05']" in text
    assert "correct the source mapping before retrying downstream code" in text


def test_tool_names_in_comments_are_not_treated_as_nested_calls() -> None:
    tool = PandasTool(pd.DataFrame({"a": [1]}), tool_cache_size=0)

    valid, message = tool.validate_code_patterns(
        "# Use the result from sql_tool\nresult = df.copy()"
    )

    assert valid, message


def test_nested_tool_call_returns_atomic_next_action_guidance() -> None:
    tool = PandasTool(pd.DataFrame({"a": [1]}), tool_cache_size=0)

    valid, message = tool.validate_code_patterns(
        'result = sql_tool(mode="execute_sql", sql="SELECT 1")'
    )

    assert not valid
    assert "one action" in message
    assert "next top-level tool call" in message


def test_pandas_tool_publishes_tables_with_duplicate_input_labels() -> None:
    duplicate_columns = pd.DataFrame([[1.23456, 2.34567]], columns=["value", "value"])
    tool = PandasTool(pd.DataFrame({"a": [1]}), tool_cache_size=0)

    processed = tool.post_process_tool_result({"result": duplicate_columns})

    assert list(processed["result"].columns) == ["value", "value_2"]
    assert processed["result"].iloc[0].tolist() == [1.2346, 2.3457]


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


def test_code_validation_does_not_treat_data_labels_as_python_capabilities() -> None:
    tool = PandasTool(pd.DataFrame({"os": ["linux"]}), tool_cache_size=0)

    valid, message = tool.validate_code_patterns(
        'result = df[df["os"] == "linux"].copy()\nresult["note"] = "open os.path"'
    )

    assert valid, message


def test_code_validation_rejects_real_system_access_via_ast() -> None:
    tool = PandasTool(pd.DataFrame({"a": [1]}), tool_cache_size=0)

    valid_import, _ = tool.validate_code_patterns("import os\nresult = df.copy()")
    valid_builtin, _ = tool.validate_code_patterns('result = open("secret.txt")')

    assert not valid_import
    assert not valid_builtin


def test_session_sandbox_does_not_expose_globals_builtin() -> None:
    assert "globals" not in SAFE_BUILTINS


def test_failed_pandas_execution_does_not_publish_assigned_variables() -> None:
    df = pd.DataFrame({"value": [1, 2]})
    sandbox = SessionSandbox()
    sandbox.bind_dataframe(df)
    tool = PandasTool(df, sandbox=sandbox, tool_cache_size=0)

    _, payload = tool._run("leaked = df.copy()\n1 / 0")

    assert payload["status"] == "error"
    assert "leaked" not in sandbox.get_user_scope()


def test_failed_pandas_execution_does_not_mutate_input_dataframe() -> None:
    df = pd.DataFrame({"value": [1, 2]})
    sandbox = SessionSandbox()
    sandbox.bind_dataframe(df)
    tool = PandasTool(df, sandbox=sandbox, tool_cache_size=0)

    tool._run("df.loc[0, 'value'] = 99\n1 / 0")

    assert df["value"].tolist() == [1, 2]


def test_successful_pandas_execution_publishes_only_declared_items() -> None:
    df = pd.DataFrame({"value": [1, 2]})
    sandbox = SessionSandbox()
    sandbox.bind_dataframe(df)
    tool = PandasTool(df, sandbox=sandbox, tool_cache_size=0)

    _, payload = tool._run(
        "temporary = df.copy()\n"
        "declared = temporary.assign(value=temporary['value'] * 2)\n"
        "tool_result = {'declared': declared}"
    )

    assert payload["artifact_type"] == "table"
    scope = sandbox.get_user_scope()
    assert scope["declared"]["value"].tolist() == [2, 4]
    assert "temporary" not in scope
    assert "tool_result" not in scope


def test_atomic_execution_preserves_reserved_user_artifacts_without_result_bleed() -> None:
    for name in ("table", "plot", "data", "result"):
        sandbox = SessionSandbox()
        source = pd.DataFrame({"value": [1, 2]})
        sandbox.put(name, source)
        tool = PandasTool(source, sandbox=sandbox, tool_cache_size=0)

        _, payload = tool._run(
            f"copied = {name}.copy()\ntool_result = {{'copied': copied}}"
        )

        assert payload["artifact_type"] == "table"
        assert sandbox.get_user_scope()[name].equals(source)

    sandbox = SessionSandbox()
    sandbox.put("result", {"stale": True})
    code = "def helper():\n    result = 'fresh'\n    return result\nprint(helper())"
    assert sandbox.execute(code, tool_name="test", isolated=True) == "fresh"
    assert sandbox.get_user_scope()["result"] == {"stale": True}


def test_timed_out_pandas_execution_preserves_previous_artifacts() -> None:
    df = pd.DataFrame({"value": [1, 2]})
    sandbox = SessionSandbox()
    sandbox.bind_dataframe(df)
    previous = pd.DataFrame({"stable": [42]})
    sandbox.put("previous", previous)
    tool = PandasTool(
        df,
        sandbox=sandbox,
        execution_timeout_sec=0.01,
        tool_cache_size=0,
    )

    _, payload = tool._run("while True:\n    pass")

    assert payload["status"] == "error"
    assert sandbox.get_user_scope()["previous"].equals(previous)
