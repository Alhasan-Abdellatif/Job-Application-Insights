"""Tests for the provider-agnostic tool-use agent (agents/tool_use.py).

Three layers, each tested independently:

* :func:`dispatch` — pure: name + args → tool result.
* :func:`serialise` — pure: tool result → JSON-able dict.
* :func:`run_tool_use_loop` — the loop, driven by a stub
  :class:`ToolUseSession`. No network, no SDK imports in tests.

The three live backends (Gemini / Anthropic / OpenAI) share this loop
code; only the SDK-specific session encoding differs. Each live backend
is exercised by manual smoke tests, not in this unit-test file.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from job_application_insights.agents.backends import (
    StubToolUseSession,
    TextTurn,
    ToolCallTurn,
)
from job_application_insights.agents.tool_use import (
    ToolUseResult,
    build_tool_specs,
    dispatch,
    run_tool_use_loop,
    serialise,
)
from job_application_insights.structured.table import load_applications_table
from job_application_insights.structured.tools import (
    CompanyCount,
    MonthCount,
    RoleRecord,
)

SAMPLE_CSV = (
    "From,Subject,Date,Body,Company,Role\n"
    '"a@gsk.com","s1","Mon, 12 Aug 2024 10:00:00 +0000","b","Gsk","Data Scientist"\n'
    '"b@gsk.com","s2","Tue, 5 Mar 2024 09:00:00 +0000","b","Gsk","ML Engineer"\n'
    '"c@gsk.com","s3","Wed, 21 Feb 2024 08:00:00 +0000","b","Gsk",""\n'
    '"d@mediatek.com","s4","Wed, 21 Feb 2024 08:00:00 +0000","b","MediaTek","ML Engineer"\n'
    '"e@ed.ac.uk","s5","Mon, 1 Jan 2024 10:00:00 +0000","b","Edinburgh","Research Associate"\n'
)


@pytest.fixture
def con(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    csv = tmp_path / "sample.csv"
    csv.write_text(SAMPLE_CSV)
    return load_applications_table(csv)


# ────────────────────────── dispatch ──────────────────────────


def test_dispatch_count_applications(con: duckdb.DuckDBPyConnection) -> None:
    assert dispatch(con, "count_applications", {}) == 5
    assert dispatch(con, "count_applications", {"company": "Gsk"}) == 3
    assert dispatch(con, "count_applications", {"company": "GSK"}) == 3  # case-insensitive


def test_dispatch_count_coerces_date_strings(con: duckdb.DuckDBPyConnection) -> None:
    """LLM hands us ISO strings; the dispatcher coerces them to date."""
    assert (
        dispatch(
            con,
            "count_applications",
            {"since": "2024-03-01", "until": "2024-08-31"},
        )
        == 2
    )


def test_dispatch_count_treats_empty_string_as_none(con: duckdb.DuckDBPyConnection) -> None:
    """Some LLMs send empty-string for omitted optional params."""
    assert dispatch(con, "count_applications", {"company": "", "since": ""}) == 5


def test_dispatch_top_companies(con: duckdb.DuckDBPyConnection) -> None:
    out = dispatch(con, "top_companies", {"n": 2})
    assert isinstance(out, list)
    assert out[0] == CompanyCount(company="Gsk", n=3)


def test_dispatch_top_companies_default_n(con: duckdb.DuckDBPyConnection) -> None:
    """No `n` argument → default of 10 from the underlying function."""
    out = dispatch(con, "top_companies", {})
    assert isinstance(out, list)
    assert len(out) == 3  # only 3 companies in the sample


def test_dispatch_applications_by_month(con: duckdb.DuckDBPyConnection) -> None:
    out = dispatch(con, "applications_by_month", {})
    assert isinstance(out, list)
    assert out[0] == MonthCount(year=2024, month=1, n=1)


def test_dispatch_find_company_role(con: duckdb.DuckDBPyConnection) -> None:
    out = dispatch(con, "find_company_role", {"company": "MediaTek"})
    assert out == [RoleRecord(doc_id="msg_000003", role="ML Engineer")]


def test_dispatch_find_company_role_rejects_missing_company(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """Empty company arg is a model error — raise so the loop can report it."""
    with pytest.raises(ValueError, match="requires a non-empty"):
        dispatch(con, "find_company_role", {})


def test_dispatch_unknown_tool_raises(con: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(ValueError, match="unknown tool"):
        dispatch(con, "totally_made_up_tool", {})


# ────────────────────────── serialise ──────────────────────────


def test_serialise_int_wraps_as_value() -> None:
    assert serialise(7) == {"value": 7}


def test_serialise_list_wraps_as_rows() -> None:
    rows = [CompanyCount(company="A", n=1), CompanyCount(company="B", n=2)]
    out = serialise(rows)
    assert out == {"rows": [{"company": "A", "n": 1}, {"company": "B", "n": 2}]}


def test_serialise_empty_list() -> None:
    assert serialise([]) == {"rows": []}


# ────────────────────────── tool specs ──────────────────────────


def test_tool_specs_cover_all_four_tools() -> None:
    specs = build_tool_specs()
    names = {s.name for s in specs}
    assert names == {
        "count_applications",
        "top_companies",
        "applications_by_month",
        "find_company_role",
    }


def test_tool_specs_have_descriptions() -> None:
    """Descriptions are prompt-engineering — empty ones are a regression."""
    for s in build_tool_specs():
        assert s.description and len(s.description) > 40


def test_find_company_role_requires_company() -> None:
    """The schema must mark `company` as required."""
    specs = {s.name: s for s in build_tool_specs()}
    spec = specs["find_company_role"]
    assert spec.parameters.get("required") == ["company"]


def test_tool_specs_parameters_are_json_schema_dicts() -> None:
    """All providers' SDKs accept JSON-schema dicts — make sure we emit them."""
    for s in build_tool_specs():
        params = s.parameters
        assert params["type"] == "object"
        assert "properties" in params


# ────────────────────────── the loop (StubToolUseSession) ──────────────────────────


def test_loop_returns_immediately_on_text_response(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """If the session returns text on the first turn, no tool runs."""
    session = StubToolUseSession([TextTurn(text="Hi, no tools needed.")])
    result = run_tool_use_loop("hello", con=con, session=session)
    assert isinstance(result, ToolUseResult)
    assert result.text == "Hi, no tools needed."
    assert result.tool_calls == []
    assert result.stopped_reason == "final_answer"


def test_loop_executes_one_tool_then_text(con: duckdb.DuckDBPyConnection) -> None:
    """Classic: tool_call → execute → tool_result → final text."""
    session = StubToolUseSession(
        [
            ToolCallTurn(name="count_applications", args={"company": "Gsk"}),
            TextTurn(text="You sent 3 applications to GSK."),
        ]
    )
    result = run_tool_use_loop("How many GSK applications?", con=con, session=session)
    assert result.text == "You sent 3 applications to GSK."
    assert result.stopped_reason == "final_answer"
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "count_applications"
    assert tc.arguments == {"company": "Gsk"}
    assert tc.output == {"value": 3}


def test_loop_chains_two_tools(con: duckdb.DuckDBPyConnection) -> None:
    """Two tool calls in sequence — checks the loop continues correctly."""
    session = StubToolUseSession(
        [
            ToolCallTurn(name="count_applications", args={}),
            ToolCallTurn(name="top_companies", args={"n": 2}),
            TextTurn(text="You sent 5 total. Top: Gsk (3), Edinburgh (1)."),
        ]
    )
    result = run_tool_use_loop("Summarise my applications", con=con, session=session)
    assert result.stopped_reason == "final_answer"
    assert [tc.name for tc in result.tool_calls] == [
        "count_applications",
        "top_companies",
    ]


def test_loop_hits_max_steps_when_model_loops(con: duckdb.DuckDBPyConnection) -> None:
    """A misbehaving model that only ever emits tool calls is bounded."""
    session = StubToolUseSession(
        [ToolCallTurn(name="count_applications", args={}) for _ in range(10)]
    )
    result = run_tool_use_loop("loop forever", con=con, session=session, max_steps=3)
    assert result.stopped_reason == "max_steps"
    assert len(result.tool_calls) == 3  # exactly the cap


def test_loop_unknown_tool_short_circuits(con: duckdb.DuckDBPyConnection) -> None:
    """The model invents a tool we don't have → stop with an error message."""
    session = StubToolUseSession([ToolCallTurn(name="nonexistent_tool", args={})])
    result = run_tool_use_loop("trick the agent", con=con, session=session)
    assert result.stopped_reason == "unknown_tool"
    assert "unknown tool" in result.text.lower()
    assert result.tool_calls == []


def test_loop_records_serialised_output(con: duckdb.DuckDBPyConnection) -> None:
    """`tool_calls[].output` is the JSON-shaped dict, not raw model objects."""
    session = StubToolUseSession(
        [
            ToolCallTurn(name="top_companies", args={"n": 1}),
            TextTurn(text="Top company: Gsk."),
        ]
    )
    result = run_tool_use_loop("top one?", con=con, session=session)
    assert result.tool_calls[0].output == {"rows": [{"company": "Gsk", "n": 3}]}


def test_loop_session_records_question_and_results(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """The session sees the right inputs: the question first, then the tool result."""
    session = StubToolUseSession(
        [
            ToolCallTurn(name="count_applications", args={}),
            TextTurn(text="ok"),
        ]
    )
    run_tool_use_loop("question text", con=con, session=session)
    assert session.questions_seen == ["question text"]
    assert len(session.tool_results_seen) == 1
    name, result = session.tool_results_seen[0]
    assert name == "count_applications"
    assert result == {"value": 5}
