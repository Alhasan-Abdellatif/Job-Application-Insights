"""Tests for the question router (agents/router.py).

The classifier wraps one Gemini call and parses its JSON output. Tests
drive it with stub responses — same SimpleNamespace shape the tool-use
tests use, so we never need google.genai for these.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from job_application_insights.agents.router import (
    RouterDecision,
    _parse_router_json,
    classify,
)
from pydantic import ValidationError


def _text_response(text: str) -> Any:
    """Build a stub Gemini response carrying only text."""
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]))]
    )


def _const_generate(text: str) -> Any:
    """Return a generate_fn that always yields the same text response."""

    def _fn(_contents: Any, _config: Any) -> Any:
        return _text_response(text)

    return _fn


# ────────────────────────── RouterDecision validator ──────────────────────────


def test_router_decision_rag_no_subquestions() -> None:
    d = RouterDecision(mode="rag")
    assert d.mode == "rag"
    assert d.structured_question is None
    assert d.rag_question is None


def test_router_decision_structured_no_subquestions() -> None:
    d = RouterDecision(mode="structured")
    assert d.mode == "structured"


def test_router_decision_both_requires_subquestions() -> None:
    with pytest.raises(ValueError, match="requires both"):
        RouterDecision(mode="both")
    with pytest.raises(ValueError, match="requires both"):
        RouterDecision(mode="both", structured_question="x")
    with pytest.raises(ValueError, match="requires both"):
        RouterDecision(mode="both", rag_question="y")


def test_router_decision_both_with_subquestions_succeeds() -> None:
    d = RouterDecision(
        mode="both",
        structured_question="How many?",
        rag_question="What role?",
    )
    assert d.mode == "both"


def test_router_decision_rejects_subquestions_on_single_modes() -> None:
    with pytest.raises(ValueError, match="must not carry sub-questions"):
        RouterDecision(mode="rag", structured_question="x")
    with pytest.raises(ValueError, match="must not carry sub-questions"):
        RouterDecision(mode="structured", rag_question="y")


# ────────────────────────── JSON parser ──────────────────────────


def test_parse_valid_rag() -> None:
    assert _parse_router_json('{"mode": "rag"}') == RouterDecision(mode="rag")


def test_parse_valid_structured() -> None:
    assert _parse_router_json('{"mode": "structured"}') == RouterDecision(mode="structured")


def test_parse_valid_both() -> None:
    raw = '{"mode": "both", "structured_question": "Q1", "rag_question": "Q2"}'
    assert _parse_router_json(raw) == RouterDecision(
        mode="both",
        structured_question="Q1",
        rag_question="Q2",
    )


def test_parse_strips_markdown_fence() -> None:
    """Some Gemini setups wrap JSON in ```json … ``` despite the system prompt."""
    raw = '```json\n{"mode": "rag"}\n```'
    assert _parse_router_json(raw) == RouterDecision(mode="rag")


def test_parse_strips_bare_fence() -> None:
    raw = '```\n{"mode": "structured"}\n```'
    assert _parse_router_json(raw) == RouterDecision(mode="structured")


def test_parse_invalid_json_returns_none() -> None:
    assert _parse_router_json("not json at all") is None
    assert _parse_router_json("{mode: rag}") is None  # unquoted
    assert _parse_router_json("") is None


def test_parse_invalid_schema_returns_none() -> None:
    """Valid JSON but wrong mode value — caller falls back to rag."""
    assert _parse_router_json('{"mode": "wibble"}') is None


def test_parse_missing_subquestions_returns_none() -> None:
    """``mode='both'`` without subquestions is a schema violation."""
    assert _parse_router_json('{"mode": "both"}') is None


# ────────────────────────── classify (live loop with stub Gemini) ──────────────────────────


def test_classify_returns_rag_decision() -> None:
    gen = _const_generate('{"mode": "rag"}')
    assert classify("Did I apply to GSK?", generate_fn=gen) == RouterDecision(mode="rag")


def test_classify_returns_structured_decision() -> None:
    gen = _const_generate('{"mode": "structured"}')
    assert classify("How many?", generate_fn=gen) == RouterDecision(mode="structured")


def test_classify_returns_both_decision() -> None:
    raw = '{"mode": "both", "structured_question": "Q1", "rag_question": "Q2"}'
    gen = _const_generate(raw)
    out = classify("Compound", generate_fn=gen)
    assert out.mode == "both"
    assert out.structured_question == "Q1"
    assert out.rag_question == "Q2"


def test_classify_falls_back_to_rag_on_bad_json() -> None:
    """Conservative default — bad model output never crashes the agent."""
    gen = _const_generate("I think this is a RAG question.")
    assert classify("anything", generate_fn=gen) == RouterDecision(mode="rag")


def test_classify_falls_back_to_rag_on_invalid_schema() -> None:
    gen = _const_generate('{"mode": "make_up_a_mode"}')
    assert classify("anything", generate_fn=gen) == RouterDecision(mode="rag")


def test_classify_falls_back_on_empty_response() -> None:
    """No candidates / no parts — same fallback path."""
    empty = SimpleNamespace(candidates=[])

    def _gen(_c: Any, _cfg: Any) -> Any:
        return empty

    assert classify("anything", generate_fn=_gen) == RouterDecision(mode="rag")


def test_classify_rejects_empty_question() -> None:
    gen = _const_generate('{"mode": "rag"}')
    with pytest.raises(ValueError, match="non-empty"):
        classify("   ", generate_fn=gen)


def test_classify_passes_question_in_contents() -> None:
    """The generate_fn must be called with the question as the user part."""
    seen: list[Any] = []

    def _gen(contents: Any, _config: Any) -> Any:
        seen.append(contents)
        return _text_response('{"mode": "rag"}')

    classify("Tell me something", generate_fn=_gen)
    contents = seen[0]
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Tell me something"


def test_router_decision_is_frozen() -> None:
    """The decision is a value object — orchestrator can pass it freely."""
    d = RouterDecision(mode="rag")
    with pytest.raises(ValidationError):
        d.mode = "structured"
