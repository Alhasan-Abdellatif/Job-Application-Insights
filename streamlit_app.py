"""Streamlit demo UI for the Job Application Insights agent.

Talks to the FastAPI service over HTTP — *not* importing the agent
directly. That separation keeps this file simple (no heavy state
management, no embedder load) and matches the prod topology where the
UI and API run in separate containers.

Run locally::

    # 1. Start the API
    uv run uvicorn job_application_insights.api.main:app --reload

    # 2. In another terminal, start the UI
    uv run streamlit run streamlit_app.py

    # Open http://localhost:8501 in a browser.

Via docker compose (Day 4)::

    docker compose up qdrant api ui
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_API_URL = os.environ.get("JAI_API_URL", "http://localhost:8000")
DEMO_MODE = os.environ.get("JAI_DEMO_MODE") == "1"

MODE_HELP = {
    "rag": "Retrieve relevant email chunks and let the LLM answer from them.",
    "tools": "Use the structured DuckDB engine + LLM function calling. Best for counts.",
    "auto": (
        "Run the question router and dispatch to RAG / structured / hybrid "
        "as appropriate. Recommended for unfamiliar questions."
    ),
}


# ────────────────────────── page setup ──────────────────────────


st.set_page_config(
    page_title="Job Application Insights",
    page_icon="📧",
    layout="wide",
)

st.title("📧 Job Application Insights")
st.caption(
    "Hybrid RAG + agentic-routing demo over a job-application email corpus. "
    "Built across weeks 1-4. See [GitHub](#) for code."
)

if DEMO_MODE:
    st.info(
        "🧪 **Public demo** — running on **synthetic** application data "
        "(100 templated ACKs across 15 fictional companies). No real "
        "emails are shipped. The default LLM provider is `echo` "
        "(deterministic, no API key); switch in the sidebar if you "
        "want a real model response."
    )


# ────────────────────────── sidebar (config) ──────────────────────────


with st.sidebar:
    st.header("Configuration")

    api_url = st.text_input(
        "API URL",
        value=DEFAULT_API_URL,
        help=(
            "FastAPI service endpoint. Defaults to http://localhost:8000; "
            "override via JAI_API_URL env var or here."
        ),
    )

    mode = st.selectbox(
        "Mode",
        ["rag", "tools", "auto"],
        index=2,
        help="\n".join(f"• `{k}` — {v}" for k, v in MODE_HELP.items()),
    )

    st.subheader("LLM providers")
    provider = st.selectbox(
        "RAG / compose provider",
        ["anthropic", "openai", "gemini", "echo"],
        index=3,  # default echo so the demo doesn't require API keys
        help=(
            "Provider for the RAG generation step (and the compose step "
            "in `auto` mode). 'echo' is a deterministic test double that "
            "needs no API key — use for the demo."
        ),
    )
    agent_provider = st.selectbox(
        "Agent (tools + router) provider",
        ["gemini", "anthropic", "openai"],
        index=0,
        help=(
            "Provider for the tool-use agent and the question router "
            "(used by `tools` and `auto` modes). Needs an API key for "
            "the chosen provider."
        ),
    )

    st.subheader("Retrieval")
    retriever = st.selectbox(
        "Retriever",
        ["dense", "bm25", "hybrid", "rerank"],
        index=2,
    )
    k = st.slider("Top-K chunks", 1, 30, 8)
    expand_parents = st.checkbox(
        "Expand to parent documents",
        value=False,
        help="Show the LLM the full email of each retrieved chunk (3-5x token cost).",
    )

    orchestrator = st.selectbox(
        "Orchestrator (auto mode only)",
        ["direct", "langgraph"],
        index=0,
        help="`direct` uses if/elif dispatch. `langgraph` uses a StateGraph.",
    )


# ────────────────────────── main UI ──────────────────────────


example_questions = (
    [
        "Did I apply to Aurora Robotics?",
        "How many applications did I send in 2025?",
        "Top 5 companies I applied to most.",
        "What role did I apply for at Granite Robotics?",
    ]
    if DEMO_MODE
    else [
        "Did I apply to GSK?",
        "How many applications did I send in 2025?",
        "Top 5 companies I applied to most.",
        "How many GSK applications and what role did I apply for?",
    ]
)
with st.expander("Example questions", expanded=False):
    for q in example_questions:
        st.code(q, language="text")

_default_q = (
    "How many applications did I send to Granite Robotics?"
    if DEMO_MODE
    else "How many applications did I send in 2025?"
)
question = st.text_area(
    "Ask a question",
    value=st.session_state.get("question", _default_q),
    height=80,
)

submit = st.button("Ask", type="primary", use_container_width=True)


# ────────────────────────── request + render ──────────────────────────


def _call_api(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST to /ask. Returns (status_code, body). Body is the JSON dict or
    {"detail": "..."} on errors."""
    with httpx.Client(timeout=60.0) as client:
        try:
            resp = client.post(f"{api_url.rstrip('/')}/ask", json=payload)
        except httpx.HTTPError as exc:
            return 0, {"detail": f"Network error: {exc}"}
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, {"detail": resp.text}


if submit and question.strip():
    st.session_state["question"] = question
    payload = {
        "question": question,
        "mode": mode,
        "retriever": retriever,
        "provider": provider,
        "agent_provider": agent_provider,
        "orchestrator": orchestrator,
        "k": k,
        "expand_parents": expand_parents,
    }
    with st.spinner("Asking the agent…"):
        status, body = _call_api(payload)

    if status == 200:
        # ── final answer ──────────────────────────────────────────
        st.subheader("Answer")
        st.info(body.get("text", "(empty)"))

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Mode used", body.get("mode_used", "?"))
        col_b.metric("Tool calls", len(body.get("tool_calls", [])))
        col_c.metric("Citations", len(body.get("citations", [])))

        # ── tool calls ────────────────────────────────────────────
        tool_calls = body.get("tool_calls", [])
        if tool_calls:
            with st.expander(f"Tool calls ({len(tool_calls)})", expanded=False):
                for tc in tool_calls:
                    st.markdown(f"**`{tc['name']}`**")
                    st.json({"arguments": tc["arguments"], "output": tc["output"]})

        # ── citations ─────────────────────────────────────────────
        citations = body.get("citations", [])
        if citations:
            with st.expander(f"Citations ({len(citations)})", expanded=False):
                for cit in citations:
                    st.markdown(
                        f"**`[{cit['chunk_id']}]`** "
                        f"score={cit['score']:.3f}  doc=`{cit['doc_id']}`"
                    )
                    st.code(cit["snippet"][:300], language="text")

        # ── stopped reason — only when interesting ────────────────
        stopped_reason = body.get("stopped_reason", "final_answer")
        if stopped_reason != "final_answer":
            st.warning(f"Stopped reason: `{stopped_reason}`")
    else:
        st.error(
            f"API returned {status}: {body.get('detail', body)}"
            + "\n\nIs the API running? `uv run uvicorn "
            + "job_application_insights.api.main:app --reload`"
        )

elif submit and not question.strip():
    st.warning("Please type a question first.")
