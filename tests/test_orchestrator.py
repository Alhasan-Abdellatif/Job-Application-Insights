"""Tests for the AgenticAgent orchestrator (agents/orchestrator.py).

All four collaborators are stubbed: classifier, tool agent, retriever,
LLM client. We assert that the *right* engines run for each routing
mode and that the resulting :class:`AgenticAnswer` carries the
expected evidence.
"""

from __future__ import annotations

from typing import Any

import pytest
from job_application_insights.agents.orchestrator import AgenticAgent, AgenticAnswer
from job_application_insights.agents.router import RouterDecision
from job_application_insights.agents.tool_use import ToolCall, ToolUseResult
from job_application_insights.ingest.chunk import Chunk


def _chunk(chunk_id: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        chunk_index=0,
        n_tokens=len(text.split()),
    )


class _StubToolAgent:
    """Returns a canned ToolUseResult and records the question asked."""

    def __init__(self, result: ToolUseResult) -> None:
        self._result = result
        self.questions_seen: list[str] = []

    def answer(self, question: str) -> ToolUseResult:
        self.questions_seen.append(question)
        return self._result


class _StubLLMClient:
    """Records the (system, user) it was called with; returns canned text."""

    def __init__(self, text: str = "stub composed answer") -> None:
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
        # When the production code uses this client for the *RAG* path
        # via generate_answer, the user prompt embeds the question and
        # the chunks — we just echo a short canned answer either way.
        return self._text


def _make_classifier(decision: RouterDecision) -> Any:
    """Return a classifier callable that always yields ``decision``."""
    return lambda _q: decision


def _make_retriever(chunk_ids: list[str]) -> Any:
    """Return a retriever callable that always yields ``chunk_ids``."""
    return lambda _q, _k: list(chunk_ids)


# ────────────────────────── rag path ──────────────────────────


def test_rag_mode_calls_retriever_and_llm() -> None:
    chunks = {
        "c1": _chunk("c1", "msg_000001", "Hello from GSK"),
    }
    agent = AgenticAgent(
        classifier=_make_classifier(RouterDecision(mode="rag")),
        tool_agent=_StubToolAgent(
            ToolUseResult(question="x", text="should not be called", stopped_reason="final_answer")
        ),
        retriever=_make_retriever(["c1"]),
        chunks_by_id=chunks,
        rag_client=_StubLLMClient("RAG answer here."),
    )
    out = agent.answer("Did I apply to GSK?")
    assert isinstance(out, AgenticAnswer)
    assert out.decision.mode == "rag"
    assert out.text == "RAG answer here."
    assert len(out.citations) == 1
    assert out.citations[0].chunk_id == "c1"
    assert out.tool_calls == []  # no structured engine ran


def test_rag_mode_skips_tool_agent() -> None:
    """Tool agent must NOT be called when the router says 'rag'."""
    tool = _StubToolAgent(
        ToolUseResult(question="x", text="should not be called", stopped_reason="final_answer")
    )
    agent = AgenticAgent(
        classifier=_make_classifier(RouterDecision(mode="rag")),
        tool_agent=tool,
        retriever=_make_retriever(["c1"]),
        chunks_by_id={"c1": _chunk("c1", "msg_1", "body")},
        rag_client=_StubLLMClient(),
    )
    agent.answer("anything")
    assert tool.questions_seen == []


# ────────────────────────── structured path ──────────────────────────


def test_structured_mode_calls_tool_agent_only() -> None:
    tool = _StubToolAgent(
        ToolUseResult(
            question="x",
            text="You sent 6 GSK applications in 2025.",
            tool_calls=[
                ToolCall(
                    name="count_applications",
                    arguments={"company": "GSK", "since": "2025-01-01", "until": "2025-12-31"},
                    output={"value": 6},
                )
            ],
            stopped_reason="final_answer",
        )
    )
    agent = AgenticAgent(
        classifier=_make_classifier(RouterDecision(mode="structured")),
        tool_agent=tool,
        retriever=_make_retriever(["should_not_be_used"]),
        chunks_by_id={},
        rag_client=_StubLLMClient(),
    )
    out = agent.answer("How many GSK applications in 2025?")
    assert out.decision.mode == "structured"
    assert out.text == "You sent 6 GSK applications in 2025."
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "count_applications"
    assert out.citations == []  # no RAG
    assert tool.questions_seen == ["How many GSK applications in 2025?"]


def test_structured_mode_propagates_stopped_reason() -> None:
    """If the tool-use loop hit max_steps, the agentic answer says so."""
    tool = _StubToolAgent(ToolUseResult(question="x", text="", stopped_reason="max_steps"))
    agent = AgenticAgent(
        classifier=_make_classifier(RouterDecision(mode="structured")),
        tool_agent=tool,
        retriever=_make_retriever([]),
        chunks_by_id={},
        rag_client=_StubLLMClient(),
    )
    out = agent.answer("anything")
    assert out.stopped_reason == "max_steps"


# ────────────────────────── both path ──────────────────────────


def _setup_both_agent(
    *,
    composed_text: str = "Combined answer",
) -> tuple[AgenticAgent, _StubToolAgent, _StubLLMClient]:
    """Build an agent configured for a 'both' decision with one chunk + one tool call."""
    decision = RouterDecision(
        mode="both",
        structured_question="How many GSK applications?",
        rag_question="What role did I apply for at GSK?",
    )
    tool = _StubToolAgent(
        ToolUseResult(
            question="x",
            text="6.",
            tool_calls=[
                ToolCall(
                    name="count_applications",
                    arguments={"company": "GSK"},
                    output={"value": 6},
                )
            ],
            stopped_reason="final_answer",
        )
    )
    llm = _StubLLMClient(composed_text)
    chunks = {"c1": _chunk("c1", "msg_001", "GSK Data Scientist role")}
    agent = AgenticAgent(
        classifier=_make_classifier(decision),
        tool_agent=tool,
        retriever=_make_retriever(["c1"]),
        chunks_by_id=chunks,
        rag_client=llm,
    )
    return agent, tool, llm


def test_both_mode_runs_both_engines() -> None:
    agent, tool, _llm = _setup_both_agent()
    out = agent.answer("How many GSK applications and what role?")
    assert out.decision.mode == "both"
    assert out.text == "Combined answer"
    assert len(out.tool_calls) == 1
    assert len(out.citations) == 1
    # tool was asked the structured sub-question
    assert tool.questions_seen == ["How many GSK applications?"]


def test_both_mode_compose_prompt_carries_both_evidences() -> None:
    """The composition LLM sees both pieces of evidence in the prompt."""
    agent, _tool, llm = _setup_both_agent()
    agent.answer("Compound question")
    assert len(llm.calls) == 1
    _system, user = llm.calls[0]
    assert "structured_evidence" in user
    assert "count_applications" in user
    assert "rag_evidence" in user
    assert "[source: c1]" in user
    assert "GSK Data Scientist role" in user


def test_both_mode_handles_empty_rag_results() -> None:
    """If the retriever returns nothing, the compose prompt notes it."""
    decision = RouterDecision(
        mode="both",
        structured_question="How many?",
        rag_question="What role?",
    )
    tool = _StubToolAgent(
        ToolUseResult(
            question="x",
            text="",
            tool_calls=[ToolCall(name="count_applications", arguments={}, output={"value": 0})],
            stopped_reason="final_answer",
        )
    )
    llm = _StubLLMClient("no role context found")
    agent = AgenticAgent(
        classifier=_make_classifier(decision),
        tool_agent=tool,
        retriever=_make_retriever([]),  # nothing retrieved
        chunks_by_id={},
        rag_client=llm,
    )
    out = agent.answer("anything")
    assert out.citations == []
    assert "no chunks retrieved" in llm.calls[0][1]


# ────────────────────────── misc ──────────────────────────


def test_empty_question_raises() -> None:
    agent = AgenticAgent(
        classifier=_make_classifier(RouterDecision(mode="rag")),
        tool_agent=_StubToolAgent(
            ToolUseResult(question="x", text="", stopped_reason="final_answer")
        ),
        retriever=_make_retriever([]),
        chunks_by_id={},
        rag_client=_StubLLMClient(),
    )
    with pytest.raises(ValueError, match="non-empty"):
        agent.answer("   ")
