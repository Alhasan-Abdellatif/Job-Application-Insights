"""Tests for the LangGraph-based orchestrator (agents/langgraph_orchestrator.py).

Two layers of confidence:

1. **Per-mode tests** mirror the direct ``AgenticAgent`` tests — each
   routing mode hits the right nodes, the right collaborators are
   called the right number of times.
2. **Equivalence tests** assert that ``LangGraphAgent.answer(q)``
   produces the *same* :class:`AgenticAnswer` as ``AgenticAgent.answer(q)``
   when given identical stubbed collaborators. This is the
   non-regression contract: switching orchestrators should never
   change the externally visible behaviour.
"""

from __future__ import annotations

from typing import Any

import pytest
from job_application_insights.agents.langgraph_orchestrator import LangGraphAgent
from job_application_insights.agents.orchestrator import AgenticAgent
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
    def __init__(self, result: ToolUseResult) -> None:
        self._result = result
        self.questions_seen: list[str] = []

    def answer(self, question: str) -> ToolUseResult:
        self.questions_seen.append(question)
        return self._result


class _StubLLMClient:
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
        return self._text


def _make_classifier(decision: RouterDecision) -> Any:
    return lambda _q: decision


def _make_retriever(chunk_ids: list[str]) -> Any:
    return lambda _q, _k: list(chunk_ids)


# ────────────────────────── per-mode flow tests ──────────────────────────


def test_rag_mode_flows_through_rag_node() -> None:
    """mode=rag → retriever + RAG LLM are called; tool agent is NOT."""
    chunks = {"c1": _chunk("c1", "msg_001", "Hello from GSK")}
    tool = _StubToolAgent(
        ToolUseResult(question="x", text="never called", stopped_reason="final_answer")
    )
    agent = LangGraphAgent(
        classifier=_make_classifier(RouterDecision(mode="rag")),
        tool_agent=tool,
        retriever=_make_retriever(["c1"]),
        chunks_by_id=chunks,
        rag_client=_StubLLMClient("RAG answer."),
    )
    out = agent.answer("Did I apply to GSK?")
    assert out.decision.mode == "rag"
    assert out.text == "RAG answer."
    assert len(out.citations) == 1
    assert out.citations[0].chunk_id == "c1"
    assert out.tool_calls == []
    assert tool.questions_seen == []  # tool node never ran


def test_structured_mode_flows_through_tools_node_and_terminates() -> None:
    """mode=structured → tools node runs, then END; no retrieve, no compose."""
    tool = _StubToolAgent(
        ToolUseResult(
            question="x",
            text="You sent 6 GSK applications in 2025.",
            tool_calls=[
                ToolCall(
                    name="count_applications",
                    arguments={"company": "GSK", "since": "2025-01-01"},
                    output={"value": 6},
                )
            ],
            stopped_reason="final_answer",
        )
    )
    llm = _StubLLMClient("compose should never run")
    agent = LangGraphAgent(
        classifier=_make_classifier(RouterDecision(mode="structured")),
        tool_agent=tool,
        retriever=_make_retriever(["should_not_be_used"]),
        chunks_by_id={},
        rag_client=llm,
    )
    out = agent.answer("How many GSK in 2025?")
    assert out.decision.mode == "structured"
    assert out.text == "You sent 6 GSK applications in 2025."
    assert len(out.tool_calls) == 1
    assert out.citations == []
    assert tool.questions_seen == ["How many GSK in 2025?"]
    # The composing LLM client was NOT invoked (this is the key
    # difference from "both" mode).
    assert llm.calls == []


def test_both_mode_flows_through_tools_then_retrieve_then_compose() -> None:
    """mode=both → all four collaborator types are exercised."""
    decision = RouterDecision(
        mode="both",
        structured_question="How many GSK applications?",
        rag_question="What role did I apply for at GSK?",
    )
    tool = _StubToolAgent(
        ToolUseResult(
            question="x",
            text="6.",  # discarded — compose rewrites
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
    llm = _StubLLMClient("You sent 6 GSK applications, applying for Data Scientist roles.")
    chunks = {"c1": _chunk("c1", "msg_001", "GSK Data Scientist role")}
    agent = LangGraphAgent(
        classifier=_make_classifier(decision),
        tool_agent=tool,
        retriever=_make_retriever(["c1"]),
        chunks_by_id=chunks,
        rag_client=llm,
    )
    out = agent.answer("How many GSK and what role?")
    assert out.decision.mode == "both"
    # Compose call's output replaces the tool agent's text.
    assert "Data Scientist" in out.text
    assert len(out.tool_calls) == 1
    assert len(out.citations) == 1
    # Tool agent was called with the *structured sub-question*, not
    # the original compound one.
    assert tool.questions_seen == ["How many GSK applications?"]
    # The compose LLM saw both pieces of evidence in its prompt.
    assert len(llm.calls) == 1
    _system, user = llm.calls[0]
    assert "count_applications" in user
    assert "[source: c1]" in user


def test_both_mode_handles_empty_rag_results() -> None:
    """retrieve returns nothing — compose still runs, no citations."""
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
    agent = LangGraphAgent(
        classifier=_make_classifier(decision),
        tool_agent=tool,
        retriever=_make_retriever([]),
        chunks_by_id={},
        rag_client=llm,
    )
    out = agent.answer("compound")
    assert out.citations == []
    assert "no chunks retrieved" in llm.calls[0][1]


# ────────────────────────── propagation ──────────────────────────


def test_stopped_reason_propagates_from_tool_agent() -> None:
    """If the tool loop hit max_steps, the LangGraph agent reports that."""
    tool = _StubToolAgent(ToolUseResult(question="x", text="", stopped_reason="max_steps"))
    agent = LangGraphAgent(
        classifier=_make_classifier(RouterDecision(mode="structured")),
        tool_agent=tool,
        retriever=_make_retriever([]),
        chunks_by_id={},
        rag_client=_StubLLMClient(),
    )
    out = agent.answer("anything")
    assert out.stopped_reason == "max_steps"


def test_empty_question_raises() -> None:
    agent = LangGraphAgent(
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


# ────────────────────────── equivalence with AgenticAgent ──────────────────────────
#
# The whole point of the LangGraph rewrite is "same answers, different
# mechanics." If a future change accidentally diverges the two
# orchestrators, these tests catch it.


@pytest.mark.parametrize(
    "decision",
    [
        RouterDecision(mode="rag"),
        RouterDecision(mode="structured"),
        RouterDecision(
            mode="both",
            structured_question="How many?",
            rag_question="What role?",
        ),
    ],
    ids=["rag", "structured", "both"],
)
def test_answer_equivalent_to_agentic_agent(decision: RouterDecision) -> None:
    """Identical stubbed collaborators → identical AgenticAnswer."""

    def make_stubs() -> tuple[Any, _StubToolAgent, Any, dict[str, Chunk], _StubLLMClient]:
        return (
            _make_classifier(decision),
            _StubToolAgent(
                ToolUseResult(
                    question="x",
                    text="tool answer",
                    tool_calls=[
                        ToolCall(name="count_applications", arguments={}, output={"value": 7})
                    ],
                    stopped_reason="final_answer",
                )
            ),
            _make_retriever(["c1"]),
            {"c1": _chunk("c1", "msg_001", "Test chunk")},
            _StubLLMClient("stub composed/rag answer"),
        )

    classifier_a, tool_a, retriever_a, chunks_a, llm_a = make_stubs()
    classifier_b, tool_b, retriever_b, chunks_b, llm_b = make_stubs()

    direct = AgenticAgent(
        classifier=classifier_a,
        tool_agent=tool_a,
        retriever=retriever_a,
        chunks_by_id=chunks_a,
        rag_client=llm_a,
    )
    langgraph = LangGraphAgent(
        classifier=classifier_b,
        tool_agent=tool_b,
        retriever=retriever_b,
        chunks_by_id=chunks_b,
        rag_client=llm_b,
    )

    out_direct = direct.answer("compound test")
    out_langgraph = langgraph.answer("compound test")

    assert out_direct.decision == out_langgraph.decision
    assert out_direct.text == out_langgraph.text
    assert out_direct.stopped_reason == out_langgraph.stopped_reason
    # Citations and tool calls are lists of Pydantic models — same
    # equality semantics as dataclasses.
    assert out_direct.citations == out_langgraph.citations
    assert out_direct.tool_calls == out_langgraph.tool_calls


# ────────────────────────── graph structure smoke ──────────────────────────


def test_compiled_graph_lists_expected_nodes() -> None:
    """The compiled graph exposes its nodes — quick check we wired five."""
    agent = LangGraphAgent(
        classifier=_make_classifier(RouterDecision(mode="rag")),
        tool_agent=_StubToolAgent(
            ToolUseResult(question="x", text="", stopped_reason="final_answer")
        ),
        retriever=_make_retriever([]),
        chunks_by_id={},
        rag_client=_StubLLMClient(),
    )
    # LangGraph's compiled graph exposes the nodes dict.
    nodes = agent._graph.nodes
    # __start__ and __end__ are framework-added; our five business nodes:
    expected = {"route", "rag", "tools", "retrieve", "compose"}
    assert expected.issubset(set(nodes.keys()))
