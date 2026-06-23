"""LangGraph-based orchestrator — the agent expressed as a StateGraph.

Same semantics as :class:`AgenticAgent` (Week 3 day 4) — router picks
``rag`` / ``structured`` / ``both``, the right engine(s) run, an
:class:`AgenticAnswer` comes out — but the *shape* of the agent is now
declared explicitly as nodes + conditional edges instead of an
if/elif chain.

The graph::

                            START
                              │
                              ▼
                          ┌──────┐
                          │route │   classifier call
                          └──┬───┘
                ┌────────────┼────────────┐
            (rag)     (structured)     (both)
                │            │            │
                ▼            ▼            ▼
            ┌─────┐      ┌──────┐    ┌──────┐
            │ rag │      │tools │    │tools │  same node, reused
            └──┬──┘      └──┬───┘    └──┬───┘
               │            │           │
               ▼            ▼           ▼
              END          END      ┌────────┐
                                    │retrieve│
                                    └───┬────┘
                                        ▼
                                    ┌────────┐
                                    │compose │
                                    └───┬────┘
                                        ▼
                                       END

Why this shape and not the Week-3 if/elif:

* **The graph is documentation.** A reader sees the agent's behaviour
  at a glance without reading any node body.
* **State updates are atomic.** Each node returns a *partial* dict of
  state updates; LangGraph merges them. No bugs from one branch
  forgetting to set a field.
* **Free upgrades.** Streaming, checkpointing, retry, parallel
  branches — all available without re-architecting. We don't use them
  today; they're available when scope grows.

This class inherits :class:`AgenticAgent` so all the per-step helper
methods (``_retrieve``, ``_answer_rag``, ``_compose_prompt``) are
reused unchanged — only :meth:`answer` is overridden to dispatch
through the compiled graph.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from job_application_insights.agents.orchestrator import (
    COMPOSE_SYSTEM_PROMPT,
    AgenticAgent,
    AgenticAnswer,
)
from job_application_insights.agents.router import RouterDecision
from job_application_insights.agents.tool_use import ToolCall, ToolUseResult
from job_application_insights.generate import Citation, generate_answer
from job_application_insights.retrieval.parent_docs import expand_to_parent_documents
from job_application_insights.retrieval.vector_store import RetrievalResult

# ────────────────────────── graph state ──────────────────────────


class AgentState(TypedDict, total=False):
    """State that flows through the graph.

    Every node receives the full state and returns a *partial* dict of
    fields to merge in. ``total=False`` lets each node populate only
    what it produces — the routing node doesn't need to know about
    ``citations``, the RAG node doesn't need to know about ``tool_result``.

    Required entries set at invocation time (i.e. ``answer()``):

    * ``question`` — the user's question.

    Everything else is populated by node(s) as the graph runs.
    """

    question: str
    decision: RouterDecision
    tool_result: ToolUseResult
    rag_results: list[RetrievalResult]
    text: str
    citations: list[Citation]
    stopped_reason: str


# ────────────────────────── agent ──────────────────────────


class LangGraphAgent(AgenticAgent):
    """:class:`AgenticAgent` rewritten as a LangGraph ``StateGraph``.

    Same constructor signature, same :meth:`answer` return type, same
    behaviour — only the *mechanics* of dispatch changed.
    """

    def __init__(
        self,
        *,
        classifier: object,  # Callable[[str], RouterDecision]
        tool_agent: object,
        retriever: object,
        chunks_by_id: dict[str, Any],
        rag_client: object,
        retrieval_k: int = 8,
        expand_parents: bool = False,
    ) -> None:
        super().__init__(
            classifier=classifier,
            tool_agent=tool_agent,  # type: ignore[arg-type]
            retriever=retriever,  # type: ignore[arg-type]
            chunks_by_id=chunks_by_id,
            rag_client=rag_client,  # type: ignore[arg-type]
            retrieval_k=retrieval_k,
            expand_parents=expand_parents,
        )
        # Compile once at construction. The same compiled graph handles
        # every question — only the state is per-invocation.
        self._graph = self._build_graph()

    # ─── public entry point ─────────────────────────────────────────

    def answer(self, question: str) -> AgenticAnswer:
        if not question.strip():
            raise ValueError("question must be non-empty")

        initial: AgentState = {"question": question}
        final: AgentState = self._graph.invoke(initial)

        return AgenticAnswer(
            question=question,
            decision=final["decision"],
            text=final.get("text", ""),
            tool_calls=list(_tool_calls_from(final)),
            citations=list(final.get("citations", [])),
            stopped_reason=final.get("stopped_reason", "final_answer"),  # type: ignore[arg-type]
        )

    # ─── graph construction ─────────────────────────────────────────

    def _build_graph(self) -> Any:
        """Wire the four nodes + conditional edges, then compile."""
        graph = StateGraph(AgentState)
        graph.add_node("route", self._route_node)
        graph.add_node("rag", self._rag_node)
        graph.add_node("tools", self._tools_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("compose", self._compose_node)

        graph.add_edge(START, "route")

        # After routing — pick the engine by the classifier's decision.
        graph.add_conditional_edges(
            "route",
            _branch_after_route,
            {
                "rag": "rag",
                "tools": "tools",
            },
        )

        # RAG terminates immediately.
        graph.add_edge("rag", END)

        # Tools terminates if pure structured; continues to retrieve+
        # compose if the original question was "both".
        graph.add_conditional_edges(
            "tools",
            _branch_after_tools,
            {
                "end": END,
                "retrieve": "retrieve",
            },
        )
        graph.add_edge("retrieve", "compose")
        graph.add_edge("compose", END)

        return graph.compile()

    # ─── individual nodes ───────────────────────────────────────────

    def _route_node(self, state: AgentState) -> dict[str, Any]:
        """Run the classifier; write the decision into state."""
        decision: RouterDecision = self._classifier(state["question"])  # type: ignore[operator]
        return {"decision": decision}

    def _rag_node(self, state: AgentState) -> dict[str, Any]:
        """RAG-only path — retrieve, generate, return final answer pieces."""
        results = self._retrieve(state["question"])
        ans = generate_answer(state["question"], results, self._rag_client)
        return {
            "rag_results": results,
            "text": ans.text,
            "citations": list(ans.citations),
            "stopped_reason": "final_answer",
        }

    def _tools_node(self, state: AgentState) -> dict[str, Any]:
        """Tool-use loop. Used by both ``structured`` and ``both`` paths.

        In ``structured`` mode the tool agent's text *is* the answer.
        In ``both`` mode the text is discarded — the compose node
        rewrites it using the tool calls + retrieved chunks together.
        """
        decision = state["decision"]
        if decision.mode == "both":
            assert decision.structured_question is not None
            question_for_tools = decision.structured_question
        else:
            question_for_tools = state["question"]

        result = self._tool_agent.answer(question_for_tools)
        # For structured mode this is the final text; for both mode
        # the compose node overwrites it.
        return {
            "tool_result": result,
            "text": result.text,
            "stopped_reason": result.stopped_reason,
        }

    def _retrieve_node(self, state: AgentState) -> dict[str, Any]:
        """RAG retrieval for the ``both`` path.

        Uses the rag sub-question (the router's decomposition), not the
        original compound question. Builds citations now so the compose
        node only has to write text.
        """
        decision = state["decision"]
        assert decision.rag_question is not None
        results = self._retrieve(decision.rag_question)
        citations = [
            Citation(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                score=r.score,
                snippet=r.chunk.text[:200],
            )
            for r in results
        ]
        return {"rag_results": results, "citations": citations}

    def _compose_node(self, state: AgentState) -> dict[str, Any]:
        """Final LLM call that uses both pieces of evidence."""
        tool_result = state["tool_result"]
        rag_results = state.get("rag_results", [])
        user_prompt = self._compose_prompt(state["question"], tool_result, rag_results)
        text = self._rag_client.complete(
            system=COMPOSE_SYSTEM_PROMPT,
            user=user_prompt,
        )
        return {"text": text.strip()}

    # ─── retrieve override (parent-doc expansion, mirrors parent class) ───

    def _retrieve(self, query: str) -> list[RetrievalResult]:
        # Same logic as AgenticAgent._retrieve but exposed here so the
        # graph nodes can call it via self.
        retrieved_ids = self._retriever(query, self._k)
        results = [
            RetrievalResult(
                chunk=self._chunks_by_id[cid],
                score=1.0 - i / max(len(retrieved_ids), 1),
            )
            for i, cid in enumerate(retrieved_ids)
            if cid in self._chunks_by_id
        ]
        if self._expand_parents:
            results = expand_to_parent_documents(results, self._chunks_by_id)
        return results


# ────────────────────────── conditional-edge functions ──────────────────────────


def _branch_after_route(state: AgentState) -> str:
    """``route → rag`` or ``route → tools`` (covers both 'structured' and 'both')."""
    mode = state["decision"].mode
    if mode == "rag":
        return "rag"
    return "tools"  # structured *and* both go through the tools node first


def _branch_after_tools(state: AgentState) -> str:
    """``tools → END`` for structured-only, ``tools → retrieve`` for both."""
    mode = state["decision"].mode
    if mode == "both":
        return "retrieve"
    return "end"


# ────────────────────────── helpers ──────────────────────────


def _tool_calls_from(state: AgentState) -> list[ToolCall]:
    """Pull tool calls out of the state, defensively (may be absent on RAG path)."""
    tool_result = state.get("tool_result")
    if tool_result is None:
        return []
    return list(tool_result.tool_calls)
