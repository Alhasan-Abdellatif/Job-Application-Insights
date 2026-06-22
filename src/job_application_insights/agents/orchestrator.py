"""Agentic orchestrator: route each question to RAG, structured, or both.

This is the top-level entry point for Week 3's agentic loop. It holds
the four collaborators — classifier, tool-use agent, retriever, RAG
LLM client — and dispatches per the router's decision:

* ``mode == "rag"`` → retrieve top-k chunks, generate a cited answer
  (Week 2 path, unchanged).
* ``mode == "structured"`` → run the Day 3 tool-use loop. The agent
  picks one of the four typed tools, executes, writes the final sentence.
* ``mode == "both"`` → run *both* engines on the decomposed
  sub-questions and feed both pieces of evidence to a single
  composition LLM call that writes the final answer.

The orchestrator does **not** itself talk to Gemini — it takes already-
constructed collaborators as constructor arguments. That makes the
whole thing trivially testable: pass stubs, get deterministic answers.
The CLI factory ([cli.py](../cli.py)) is the place that wires the
real Gemini clients in.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from job_application_insights.agents.router import RouterDecision
from job_application_insights.agents.tool_use import ToolCall, ToolUseAgent
from job_application_insights.evals.runner import RetrievalFn
from job_application_insights.generate import (
    Answer,
    Citation,
    LLMClient,
    generate_answer,
)
from job_application_insights.ingest.chunk import Chunk
from job_application_insights.retrieval.vector_store import RetrievalResult

DEFAULT_RETRIEVAL_K: int = 8

COMPOSE_SYSTEM_PROMPT: str = (
    "You compose final answers to questions about Alhasan's job "
    "application emails. You will be given the question plus evidence "
    "gathered from two sources: a structured-data tool (returning "
    "counts / aggregations / typed rows) and a retrieval system "
    "(returning relevant email chunks with bracketed [source: ...] IDs).\n\n"
    "Write one short factual answer using both pieces of evidence. Cite "
    "email sources by their bracketed IDs when you reference their "
    "contents. If a piece of evidence is empty or doesn't answer the "
    "question, say so honestly — do not invent."
)


# ────────────────────────── data model ──────────────────────────


class AgenticAnswer(BaseModel):
    """Typed output of one :meth:`AgenticAgent.answer` round-trip."""

    model_config = ConfigDict(frozen=True)

    question: str
    decision: RouterDecision
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    stopped_reason: Literal["final_answer", "max_steps", "unknown_tool"] = "final_answer"


# ────────────────────────── orchestrator ──────────────────────────


class AgenticAgent:
    """Holds the four collaborators and dispatches per router decision.

    All four are injected via the constructor. Tests pass stubs; the
    CLI passes real Gemini-backed clients. The orchestrator itself has
    no Gemini knowledge — it just knows how to compose the pieces.
    """

    def __init__(
        self,
        *,
        classifier: object,  # Callable[[str], RouterDecision]
        tool_agent: ToolUseAgent,
        retriever: RetrievalFn,
        chunks_by_id: dict[str, Chunk],
        rag_client: LLMClient,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
    ) -> None:
        self._classifier = classifier
        self._tool_agent = tool_agent
        self._retriever = retriever
        self._chunks_by_id = chunks_by_id
        self._rag_client = rag_client
        self._k = retrieval_k

    def answer(self, question: str) -> AgenticAnswer:
        """Classify the question, dispatch to the right engine(s), return."""
        if not question.strip():
            raise ValueError("question must be non-empty")

        decision: RouterDecision = self._classifier(question)  # type: ignore[operator]

        if decision.mode == "rag":
            return self._answer_rag(question, decision, question)

        if decision.mode == "structured":
            return self._answer_structured(question, decision, question)

        # decision.mode == "both" — validator guarantees the sub-questions exist.
        assert decision.structured_question is not None
        assert decision.rag_question is not None
        return self._answer_both(
            question,
            decision,
            structured_q=decision.structured_question,
            rag_q=decision.rag_question,
        )

    # ─── individual dispatch arms ─────────────────────────────────────

    def _answer_rag(
        self,
        question: str,
        decision: RouterDecision,
        rag_query: str,
    ) -> AgenticAnswer:
        results = self._retrieve(rag_query)
        ans: Answer = generate_answer(rag_query, results, self._rag_client)
        return AgenticAnswer(
            question=question,
            decision=decision,
            text=ans.text,
            citations=list(ans.citations),
            stopped_reason="final_answer",
        )

    def _answer_structured(
        self,
        question: str,
        decision: RouterDecision,
        structured_query: str,
    ) -> AgenticAnswer:
        result = self._tool_agent.answer(structured_query)
        return AgenticAnswer(
            question=question,
            decision=decision,
            text=result.text,
            tool_calls=list(result.tool_calls),
            stopped_reason=result.stopped_reason,
        )

    def _answer_both(
        self,
        question: str,
        decision: RouterDecision,
        *,
        structured_q: str,
        rag_q: str,
    ) -> AgenticAnswer:
        tool_result = self._tool_agent.answer(structured_q)
        rag_results = self._retrieve(rag_q)
        citations = [
            Citation(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                score=r.score,
                snippet=r.chunk.text[:200],
            )
            for r in rag_results
        ]

        # One LLM call composes the final answer from both pieces of
        # evidence. We reuse the RAG LLM client — it satisfies the
        # LLMClient Protocol, so any provider works.
        user_prompt = self._compose_prompt(question, tool_result, rag_results)
        text = self._rag_client.complete(
            system=COMPOSE_SYSTEM_PROMPT,
            user=user_prompt,
        )
        return AgenticAnswer(
            question=question,
            decision=decision,
            text=text.strip(),
            tool_calls=list(tool_result.tool_calls),
            citations=citations,
            stopped_reason=tool_result.stopped_reason,
        )

    # ─── helpers ─────────────────────────────────────────────────────

    def _retrieve(self, query: str) -> list[RetrievalResult]:
        """Run the retriever and rebuild :class:`RetrievalResult` objects.

        Mirrors :func:`cli._ask`'s rank-derived synthetic score — keeps
        citations visually ordered without lying about the literal score.
        """
        retrieved_ids = self._retriever(query, self._k)
        return [
            RetrievalResult(
                chunk=self._chunks_by_id[cid],
                score=1.0 - i / max(len(retrieved_ids), 1),
            )
            for i, cid in enumerate(retrieved_ids)
            if cid in self._chunks_by_id
        ]

    def _compose_prompt(
        self,
        question: str,
        tool_result: object,  # ToolUseResult, kept untyped to avoid circular import
        rag_results: list[RetrievalResult],
    ) -> str:
        """Build the user prompt for the "both" composition LLM call."""
        parts: list[str] = [f"<question>\n{question}\n</question>", ""]

        parts.append("<structured_evidence>")
        tool_calls = getattr(tool_result, "tool_calls", [])
        if not tool_calls:
            parts.append("(no tool calls — structured engine returned text directly)")
            structured_text = getattr(tool_result, "text", "")
            if structured_text:
                parts.append(f"Structured engine said: {structured_text}")
        else:
            for tc in tool_calls:
                parts.append(f"Called {tc.name}({tc.arguments}) → {tc.output}")
        parts.append("</structured_evidence>")
        parts.append("")

        parts.append("<rag_evidence>")
        if not rag_results:
            parts.append("(no chunks retrieved)")
        else:
            for r in rag_results:
                parts.append(f"[source: {r.chunk.chunk_id}]")
                parts.append(r.chunk.text)
                parts.append("")
        parts.append("</rag_evidence>")
        parts.append("")
        parts.append(
            "Compose a single short factual answer using both pieces of "
            "evidence. Cite email sources by their bracketed IDs."
        )
        return "\n".join(parts)
