"""Tests for the Gemini tool-use agent over the structured table.

Three layers, each tested independently:

* :func:`dispatch` — pure: name + args → tool result.
* :func:`serialise` — pure: tool result → JSON-able dict.
* :func:`run_tool_use_loop` — the loop, driven by a ``generate_fn``
  stub that returns canned Gemini responses. No network.

The live :class:`GeminiToolUseAgent` is exercised separately via a
smoke test against the real API (run by hand, not in CI).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import pytest
from job_application_insights.agents.tool_use import (
    ToolUseResult,
    build_function_declarations,
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
    """Some Gemini setups send empty-string for omitted optional params."""
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


# ────────────────────────── function declarations ──────────────────────────


def test_function_declarations_cover_all_four_tools() -> None:
    decls = build_function_declarations()
    names = {d.name for d in decls}
    assert names == {
        "count_applications",
        "top_companies",
        "applications_by_month",
        "find_company_role",
    }


def test_function_declarations_have_descriptions() -> None:
    """Descriptions are prompt-engineering — empty ones are a regression."""
    for d in build_function_declarations():
        assert d.description and len(d.description) > 40


def test_find_company_role_requires_company() -> None:
    """The schema must mark `company` as required."""
    decls = {d.name: d for d in build_function_declarations()}
    decl = decls["find_company_role"]
    assert decl.parameters is not None
    assert decl.parameters.required == ["company"]


# ────────────────────────── the loop (stubbed Gemini) ──────────────────────────


def _text_response(text: str) -> Any:
    """Build a stub Gemini response that contains only text."""
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text=text, function_call=None)],
                )
            )
        ]
    )


def _call_response(name: str, args: dict[str, Any]) -> Any:
    """Build a stub Gemini response that contains one function_call."""
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text=None,
                            function_call=SimpleNamespace(name=name, args=args),
                        )
                    ],
                )
            )
        ]
    )


def _scripted_generate(responses: list[Any]) -> Any:
    """Return a generate_fn that yields the next canned response per call."""
    state = {"i": 0}
    contents_seen: list[Any] = []
    configs_seen: list[Any] = []

    def _fn(contents: Any, config: Any) -> Any:
        contents_seen.append(contents)
        configs_seen.append(config)
        i = state["i"]
        state["i"] = i + 1
        if i >= len(responses):
            raise AssertionError(f"generate_fn called {i + 1} times; only {len(responses)} canned")
        return responses[i]

    _fn.contents_seen = contents_seen  # type: ignore[attr-defined]
    _fn.configs_seen = configs_seen  # type: ignore[attr-defined]
    _fn.state = state  # type: ignore[attr-defined]
    return _fn


def test_loop_returns_immediately_on_text_response(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """If Gemini returns text on the first call, we don't call any tool."""
    gen = _scripted_generate([_text_response("Hi, I have no tools to call.")])
    result = run_tool_use_loop("hello", con=con, generate_fn=gen)
    assert isinstance(result, ToolUseResult)
    assert result.text == "Hi, I have no tools to call."
    assert result.tool_calls == []
    assert result.stopped_reason == "final_answer"


def test_loop_executes_one_tool_then_text(con: duckdb.DuckDBPyConnection) -> None:
    """Classic: function_call → execute → function_response → final text."""
    gen = _scripted_generate(
        [
            _call_response("count_applications", {"company": "Gsk"}),
            _text_response("You sent 3 applications to GSK."),
        ]
    )
    result = run_tool_use_loop("How many GSK applications?", con=con, generate_fn=gen)
    assert result.text == "You sent 3 applications to GSK."
    assert result.stopped_reason == "final_answer"
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "count_applications"
    assert tc.arguments == {"company": "Gsk"}
    assert tc.output == {"value": 3}


def test_loop_chains_two_tools(con: duckdb.DuckDBPyConnection) -> None:
    """Two tool calls in sequence — checks the loop continues correctly."""
    gen = _scripted_generate(
        [
            _call_response("count_applications", {}),
            _call_response("top_companies", {"n": 2}),
            _text_response("You sent 5 total. Top: Gsk (3), Edinburgh (1)."),
        ]
    )
    result = run_tool_use_loop("Summarise my applications", con=con, generate_fn=gen)
    assert result.stopped_reason == "final_answer"
    assert [tc.name for tc in result.tool_calls] == [
        "count_applications",
        "top_companies",
    ]


def test_loop_hits_max_steps_when_model_loops(con: duckdb.DuckDBPyConnection) -> None:
    """A misbehaving model that only ever emits function calls is bounded."""
    gen = _scripted_generate([_call_response("count_applications", {}) for _ in range(10)])
    result = run_tool_use_loop("loop forever", con=con, generate_fn=gen, max_steps=3)
    assert result.stopped_reason == "max_steps"
    assert len(result.tool_calls) == 3  # exactly the cap


def test_loop_unknown_tool_short_circuits(con: duckdb.DuckDBPyConnection) -> None:
    """The model invents a tool we don't have → stop with an error message."""
    gen = _scripted_generate([_call_response("nonexistent_tool", {})])
    result = run_tool_use_loop("trick the agent", con=con, generate_fn=gen)
    assert result.stopped_reason == "unknown_tool"
    assert "unknown tool" in result.text.lower()
    assert result.tool_calls == []


def test_loop_records_serialised_output(con: duckdb.DuckDBPyConnection) -> None:
    """`tool_calls[].output` is the JSON-shaped dict, not raw model objects."""
    gen = _scripted_generate(
        [
            _call_response("top_companies", {"n": 1}),
            _text_response("Top company: Gsk."),
        ]
    )
    result = run_tool_use_loop("top one?", con=con, generate_fn=gen)
    assert result.tool_calls[0].output == {"rows": [{"company": "Gsk", "n": 3}]}


def test_loop_passes_tools_in_config(con: duckdb.DuckDBPyConnection) -> None:
    """The config sent to Gemini must include the four tool declarations."""
    gen = _scripted_generate([_text_response("ok")])
    run_tool_use_loop("noop", con=con, generate_fn=gen)
    config = gen.configs_seen[0]
    decls = config.tools[0].function_declarations
    assert {d.name for d in decls} == {
        "count_applications",
        "top_companies",
        "applications_by_month",
        "find_company_role",
    }
