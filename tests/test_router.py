"""Tests for the question router (agents/router.py).

The classifier wraps one LLMClient.complete() call and parses its JSON
output. Tests drive it with a stub LLMClient — provider-agnostic now
that the router doesn't import google.genai directly.
"""

from __future__ import annotations

import pytest
from job_application_insights.agents.router import (
    RouterDecision,
    _parse_router_json,
    classify,
)
from pydantic import ValidationError


class _StubLLMClient:
    """Returns a canned text completion, records the prompts it was called with."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[tuple[str, str]] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        del max_tokens, temperature
        self.calls.append((system, user))
        return self._text


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
    """Anthropic/OpenAI sometimes wrap JSON in ```json … ``` despite the system prompt."""
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


# ────────────────────────── classify with stub LLMClient ──────────────────────────


def test_classify_returns_rag_decision() -> None:
    client = _StubLLMClient('{"mode": "rag"}')
    assert classify("Did I apply to GSK?", llm_client=client) == RouterDecision(mode="rag")


def test_classify_returns_structured_decision() -> None:
    client = _StubLLMClient('{"mode": "structured"}')
    assert classify("How many?", llm_client=client) == RouterDecision(mode="structured")


def test_classify_returns_both_decision() -> None:
    raw = '{"mode": "both", "structured_question": "Q1", "rag_question": "Q2"}'
    client = _StubLLMClient(raw)
    out = classify("Compound", llm_client=client)
    assert out.mode == "both"
    assert out.structured_question == "Q1"
    assert out.rag_question == "Q2"


def test_classify_falls_back_to_rag_on_bad_json() -> None:
    """Conservative default — bad model output never crashes the agent."""
    client = _StubLLMClient("I think this is a RAG question.")
    assert classify("anything", llm_client=client) == RouterDecision(mode="rag")


def test_classify_falls_back_to_rag_on_invalid_schema() -> None:
    client = _StubLLMClient('{"mode": "make_up_a_mode"}')
    assert classify("anything", llm_client=client) == RouterDecision(mode="rag")


def test_classify_falls_back_on_empty_response() -> None:
    """Empty completion — same fallback path."""
    client = _StubLLMClient("")
    assert classify("anything", llm_client=client) == RouterDecision(mode="rag")


def test_classify_rejects_empty_question() -> None:
    client = _StubLLMClient('{"mode": "rag"}')
    with pytest.raises(ValueError, match="non-empty"):
        classify("   ", llm_client=client)


def test_classify_passes_question_as_user_prompt() -> None:
    """The LLMClient must be called with the question as ``user``."""
    client = _StubLLMClient('{"mode": "rag"}')
    classify("Tell me something", llm_client=client)
    assert len(client.calls) == 1
    system, user = client.calls[0]
    assert "router" in system.lower()
    assert user == "Tell me something"


def test_router_decision_is_frozen() -> None:
    """The decision is a value object — orchestrator can pass it freely."""
    d = RouterDecision(mode="rag")
    with pytest.raises(ValidationError):
        d.mode = "structured"
