"""Tests for :mod:`job_application_insights.generate`."""

from __future__ import annotations

import pytest
from job_application_insights.generate import (
    DEFAULT_ANTHROPIC_MODEL,
    SYSTEM_PROMPT,
    Answer,
    Citation,
    EchoClient,
    format_prompt,
    generate_answer,
)
from job_application_insights.ingest.chunk import Chunk
from job_application_insights.retrieval.vector_store import RetrievalResult
from pydantic import ValidationError

# ───── helpers ─────


def _result(chunk_id: str, text: str, score: float = 0.7) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("__", maxsplit=1)[0],
        chunk_index=0,
        text=text,
        n_tokens=10,
        subject="hello",
        sender="x@y.com",
    )
    return RetrievalResult(chunk=chunk, score=score)


# ───── format_prompt ─────


def test_format_prompt_returns_system_and_user():
    system, user = format_prompt("what?", [_result("msg_001__c000", "hello")])
    assert system == SYSTEM_PROMPT
    assert "<question>" in user
    assert "what?" in user
    assert "<context>" in user
    assert "[source: msg_001__c000]" in user
    assert "hello" in user


def test_format_prompt_includes_every_chunk_with_source_tag():
    results = [
        _result("msg_001__c000", "first chunk text"),
        _result("msg_002__c003", "second chunk text"),
        _result("msg_003__c000", "third chunk text"),
    ]
    _, user = format_prompt("q", results)
    for r in results:
        assert f"[source: {r.chunk.chunk_id}]" in user
        assert r.chunk.text in user


def test_format_prompt_handles_empty_results():
    system, user = format_prompt("q", [])
    assert system == SYSTEM_PROMPT
    assert "no context retrieved" in user


def test_format_prompt_orders_chunks_as_given():
    """Chunk order in the prompt mirrors the input list order."""
    a = _result("msg_a__c000", "AAA")
    b = _result("msg_b__c000", "BBB")
    _, user = format_prompt("q", [a, b])
    assert user.find("AAA") < user.find("BBB")
    # And the other way around when we swap
    _, user2 = format_prompt("q", [b, a])
    assert user2.find("BBB") < user2.find("AAA")


# ───── EchoClient ─────


def test_echo_client_returns_chunk_ids_in_response():
    client = EchoClient()
    response = client.complete(
        system="you are a test",
        user="<context>[source: msg_x] hi [source: msg_y] there</context>",
    )
    assert "msg_x" in response
    assert "msg_y" in response
    assert "2 sources" in response


def test_echo_client_handles_no_sources():
    client = EchoClient()
    response = client.complete(system="s", user="no source tags here")
    assert "no sources" in response


def test_echo_client_prefix_customisable():
    client = EchoClient(prefix="STUB")
    response = client.complete(system="s", user="[source: x]")
    assert response.startswith("STUB:")


# ───── generate_answer ─────


def test_generate_answer_rejects_empty_query():
    with pytest.raises(ValueError, match="non-empty"):
        generate_answer("", [], EchoClient())


def test_generate_answer_rejects_whitespace_query():
    with pytest.raises(ValueError, match="non-empty"):
        generate_answer("   \n  ", [], EchoClient())


def test_generate_answer_round_trips_with_echo_client():
    results = [
        _result("msg_001__c000", "Tomorrow Climate ML role posting", score=0.84),
        _result("msg_002__c000", "ETH Zurich postdoc opportunity", score=0.71),
    ]
    answer = generate_answer("what roles?", results, EchoClient())
    assert isinstance(answer, Answer)
    assert answer.query == "what roles?"
    assert "msg_001__c000" in answer.text
    assert "msg_002__c000" in answer.text
    assert len(answer.citations) == 2


def test_generate_answer_attaches_citations_for_every_retrieved_chunk():
    results = [
        _result("msg_a__c000", "chunk A", score=0.9),
        _result("msg_b__c001", "chunk B", score=0.7),
    ]
    answer = generate_answer("q", results, EchoClient())
    cited_ids = {c.chunk_id for c in answer.citations}
    assert cited_ids == {"msg_a__c000", "msg_b__c001"}


def test_generate_answer_snippets_are_truncated():
    long_text = "x" * 1000
    results = [_result("msg_a__c000", long_text)]
    answer = generate_answer("q", results, EchoClient())
    assert len(answer.citations[0].snippet) <= 200


def test_generate_answer_empty_context_still_calls_llm():
    """With no retrieved chunks, the LLM should still be asked — and it should
    say 'I don't know' or similar. The EchoClient's specific behaviour
    doesn't matter here; we just want the call to go through."""
    answer = generate_answer("q", [], EchoClient())
    assert isinstance(answer, Answer)
    assert answer.citations == []


# ───── Answer / Citation invariants ─────


def test_citation_is_frozen():
    cit = Citation(chunk_id="x__c000", doc_id="x", score=0.5, snippet="hello")
    with pytest.raises(ValidationError):
        cit.score = 0.9  # type: ignore[misc]


def test_citation_score_bounds():
    Citation(chunk_id="x__c000", doc_id="x", score=-1.0, snippet="")
    Citation(chunk_id="x__c000", doc_id="x", score=1.0, snippet="")
    with pytest.raises(ValidationError):
        Citation(chunk_id="x__c000", doc_id="x", score=1.5, snippet="")


def test_answer_is_frozen():
    ans = Answer(query="q", text="t", citations=[])
    with pytest.raises(ValidationError):
        ans.text = "boom"  # type: ignore[misc]


def test_answer_rejects_empty_query():
    with pytest.raises(ValidationError):
        Answer(query="", text="t", citations=[])


# ───── constants exist ─────


def test_default_anthropic_model_constant():
    assert DEFAULT_ANTHROPIC_MODEL.startswith("claude-")
