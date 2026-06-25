"""FastAPI service exposing the agent over HTTP.

Endpoints:

* ``GET  /health`` — liveness check. Returns immediately; doesn't touch
  the vector store or the LLM. Used by docker-compose's healthcheck.
* ``GET  /``       — service info (version, default config).
* ``POST /ask``    — the main endpoint. Take a question + mode, return
  a typed :class:`AskResponse` with text, citations, tool calls.

Design:

* **Typed boundary.** :class:`AskRequest` and :class:`AskResponse` are
  Pydantic models, so FastAPI parses/serialises automatically and
  generates OpenAPI docs at ``/docs``. The same boundary discipline as
  Week 1's ``Document``: outside is hostile JSON, inside is typed
  Python.
* **Heavy state loaded once.** The embedder, vector store, DuckDB
  connection, and corpus chunk-lookup are built during the FastAPI
  ``lifespan`` (startup) and live in ``app.state`` for the lifetime of
  the process. Per-request handlers reuse them.
* **Per-request agent construction.** LLM clients and agents depend on
  the request's ``provider`` / ``mode`` / ``agent_provider``, so we
  build them per-request — cheap (no model loading, just SDK clients).
* **`create_app(state=None)` factory.** Tests inject pre-built state
  to skip the slow real-data startup; production calls ``create_app()``
  which triggers the lifespan.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import duckdb
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from job_application_insights import __version__
from job_application_insights.agents.backends import AGENT_PROVIDER_NAMES
from job_application_insights.agents.langgraph_orchestrator import LangGraphAgent
from job_application_insights.agents.orchestrator import AgenticAgent
from job_application_insights.agents.router import RouterDecision, classify
from job_application_insights.agents.tool_use import LiveToolUseAgent
from job_application_insights.evals.runner import (
    make_bm25_retriever,
    make_dense_retriever,
)
from job_application_insights.generate import (
    PROVIDER_NAMES,
    generate_answer,
    make_llm_client,
)
from job_application_insights.ingest.chunk import Chunk
from job_application_insights.ingest.embed import Embedder
from job_application_insights.retrieval.bm25 import BM25Index
from job_application_insights.retrieval.hybrid import make_hybrid_retriever
from job_application_insights.retrieval.parent_docs import expand_to_parent_documents
from job_application_insights.retrieval.rerank import (
    CrossEncoderReranker,
    make_reranked_retriever,
)
from job_application_insights.retrieval.vector_store import (
    RetrievalResult,
    make_vector_store,
)
from job_application_insights.structured.table import load_applications_table

# ────────────────────────── config (env-driven) ──────────────────────────

logger = logging.getLogger(__name__)

ENV_STORE_BACKEND = "JAI_STORE_BACKEND"
ENV_STORE_PATH = "JAI_STORE_PATH"
ENV_QDRANT_URL = "JAI_QDRANT_URL"
ENV_QDRANT_PATH = "JAI_QDRANT_PATH"
ENV_STRUCTURED_CSV = "JAI_STRUCTURED_CSV"

DEFAULT_STORE_BACKEND = "chroma"
DEFAULT_STORE_PATH = "./data/chroma"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_STRUCTURED_CSV = "./data/synthetic/ack_only.csv"

API_DEFAULT_K = 8


# ────────────────────────── request / response ──────────────────────────


class AskRequest(BaseModel):
    """Incoming question + agent configuration knobs."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(..., min_length=1)
    mode: Literal["rag", "tools", "auto"] = "rag"
    retriever: Literal["dense", "bm25", "hybrid", "rerank"] = "dense"
    provider: Literal["anthropic", "openai", "gemini", "echo"] = "echo"
    agent_provider: Literal["gemini", "anthropic", "openai"] = "gemini"
    orchestrator: Literal["direct", "langgraph"] = "direct"
    k: int = Field(default=API_DEFAULT_K, ge=1, le=50)
    expand_parents: bool = False


class CitationOut(BaseModel):
    """One retrieved chunk reference in the response."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    score: float
    snippet: str


class ToolCallOut(BaseModel):
    """One tool-call record in the response."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]


class AskResponse(BaseModel):
    """The full agent response, typed."""

    model_config = ConfigDict(frozen=True)

    question: str
    mode_used: str
    text: str
    tool_calls: list[ToolCallOut] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    stopped_reason: str = "final_answer"


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str = __version__


# ────────────────────────── lifespan / state ──────────────────────────


def _load_state() -> dict[str, Any]:
    """Build the heavy app state from environment-configured paths.

    Returns a dict that gets stuffed into ``app.state`` for the
    lifetime of the process. Tests bypass this by passing
    ``state=`` to :func:`create_app` directly.
    """
    store_backend = os.environ.get(ENV_STORE_BACKEND, DEFAULT_STORE_BACKEND)
    store_path = os.environ.get(ENV_STORE_PATH, DEFAULT_STORE_PATH)
    qdrant_url = os.environ.get(ENV_QDRANT_URL, DEFAULT_QDRANT_URL)
    qdrant_path = os.environ.get(ENV_QDRANT_PATH) or None
    structured_csv = os.environ.get(ENV_STRUCTURED_CSV, DEFAULT_STRUCTURED_CSV)

    logger.info("Loading embedder…")
    embedder = Embedder()

    logger.info(
        "Opening vector store (%s, %s)…",
        store_backend,
        f"path={qdrant_path}" if qdrant_path else f"url={qdrant_url}",
    )
    store = make_vector_store(
        store_backend,
        persist_path=store_path,
        qdrant_url=qdrant_url,
        qdrant_path=qdrant_path,
        vector_size=embedder.dimension,
    )

    if store.n_chunks == 0:
        logger.warning("Vector store is empty. Run `jai ingest <csv>` before asking questions.")

    logger.info("Reading chunks for in-memory lookup…")
    chunks_by_id: dict[str, Chunk] = {c.chunk_id: c for c in store.iter_chunks()}

    structured_path = Path(structured_csv)
    if structured_path.exists():
        logger.info("Loading structured DuckDB table from %s…", structured_path)
        duckdb_con: duckdb.DuckDBPyConnection | None = load_applications_table(structured_path)
    else:
        logger.warning(
            "Structured CSV %s not found; --mode tools/auto will reject requests until present.",
            structured_path,
        )
        duckdb_con = None

    return {
        "embedder": embedder,
        "store": store,
        "chunks_by_id": chunks_by_id,
        "duckdb_con": duckdb_con,
        # Retriever cache, keyed by retriever name. We build on first
        # use to keep startup fast — loading the cross-encoder reranker
        # is ~2s and not everyone wants it.
        "retrievers": {},
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build heavy state at startup, expose it via ``app.state``."""
    state = _load_state()
    for key, value in state.items():
        setattr(app.state, key, value)
    try:
        yield
    finally:
        # Nothing to clean up — DuckDB connection / store close on GC.
        pass


# ────────────────────────── app factory ──────────────────────────


def create_app(state: dict[str, Any] | None = None) -> FastAPI:
    """Build the FastAPI app.

    Parameters
    ----------
    state
        Optional pre-built state dict. When provided, the lifespan is
        skipped and these values become ``app.state``. Used by tests
        to inject stubs; production calls ``create_app()`` and lets
        the lifespan load the real components.
    """
    if state is None:
        app = FastAPI(
            title="Job Application Insights API",
            version=__version__,
            description=(
                "RAG + agentic-routing over job-application emails. "
                "POST a question to /ask; see /docs for the full schema."
            ),
            lifespan=lifespan,
        )
    else:
        # Test path — no lifespan, state is provided.
        app = FastAPI(
            title="Job Application Insights API (test)",
            version=__version__,
        )
        for key, value in state.items():
            setattr(app.state, key, value)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    _register_routes(app)
    return app


# ────────────────────────── routes ──────────────────────────


def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "job-application-insights",
            "version": __version__,
            "endpoints": ["GET /health", "POST /ask", "GET /docs"],
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request) -> AskResponse:
        st = request.app.state
        store = getattr(st, "store", None)
        if store is None or store.n_chunks == 0:
            raise HTTPException(
                status_code=503,
                detail=("Vector store is empty. Run `jai ingest <csv>` before asking questions."),
            )

        if req.mode in ("tools", "auto") and getattr(st, "duckdb_con", None) is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Structured CSV not loaded. --mode tools/auto requires "
                    "a populated ack-only CSV (see JAI_STRUCTURED_CSV)."
                ),
            )

        try:
            if req.mode == "tools":
                return _serve_tools(req, st)
            if req.mode == "auto":
                return _serve_auto(req, st)
            return _serve_rag(req, st)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Error handling /ask")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


# ────────────────────────── handlers ──────────────────────────


def _serve_rag(req: AskRequest, st: Any) -> AskResponse:
    """RAG path — retrieve + generate, possibly with parent-doc expansion."""
    retriever = _get_or_build_retriever(req.retriever, st)
    chunks_by_id: dict[str, Chunk] = st.chunks_by_id
    retrieved_ids = retriever(req.question, req.k)
    results = [
        RetrievalResult(
            chunk=chunks_by_id[cid],
            score=1.0 - i / max(len(retrieved_ids), 1),
        )
        for i, cid in enumerate(retrieved_ids)
        if cid in chunks_by_id
    ]
    if req.expand_parents:
        results = expand_to_parent_documents(results, chunks_by_id)

    client = make_llm_client(req.provider)
    answer = generate_answer(req.question, results, client)
    return AskResponse(
        question=req.question,
        mode_used="rag",
        text=answer.text,
        citations=[_to_citation_out(c) for c in answer.citations],
    )


def _serve_tools(req: AskRequest, st: Any) -> AskResponse:
    """Structured-engine-only path — tool-use loop, no retrieval."""
    agent = LiveToolUseAgent(st.duckdb_con, provider=req.agent_provider)
    result = agent.answer(req.question)
    return AskResponse(
        question=req.question,
        mode_used="tools",
        text=result.text,
        tool_calls=[_to_tool_call_out(tc) for tc in result.tool_calls],
        stopped_reason=result.stopped_reason,
    )


def _serve_auto(req: AskRequest, st: Any) -> AskResponse:
    """Full agentic loop — router → engine(s) → typed answer."""
    retriever = _get_or_build_retriever(req.retriever, st)
    chunks_by_id: dict[str, Chunk] = st.chunks_by_id

    router_client = make_llm_client(req.agent_provider)

    def _router(question: str) -> RouterDecision:
        return classify(question, llm_client=router_client)

    agent_cls = LangGraphAgent if req.orchestrator == "langgraph" else AgenticAgent
    agent = agent_cls(
        classifier=_router,
        tool_agent=LiveToolUseAgent(st.duckdb_con, provider=req.agent_provider),
        retriever=retriever,
        chunks_by_id=chunks_by_id,
        rag_client=make_llm_client(req.provider),
        retrieval_k=req.k,
        expand_parents=req.expand_parents,
    )
    result = agent.answer(req.question)
    return AskResponse(
        question=req.question,
        mode_used=result.decision.mode,
        text=result.text,
        tool_calls=[_to_tool_call_out(tc) for tc in result.tool_calls],
        citations=[_to_citation_out(c) for c in result.citations],
        stopped_reason=result.stopped_reason,
    )


# ────────────────────────── retriever cache + serialisation helpers ──────────────


def _get_or_build_retriever(name: str, st: Any) -> Any:
    """Cache retrievers by name in app.state.retrievers — built once per process."""
    cache: dict[str, Any] = st.retrievers
    if name in cache:
        return cache[name]
    retriever = _build_retriever(name, st.embedder, st.store)
    cache[name] = retriever
    return retriever


def _build_retriever(name: str, embedder: Embedder, store: Any) -> Any:
    """Concrete retriever construction — mirrors `cli._build_retriever`."""
    if name == "dense":
        return make_dense_retriever(store, embedder)
    if name == "bm25":
        chunks = store.iter_chunks()
        index = BM25Index(chunks)
        return make_bm25_retriever(index)
    if name == "hybrid":
        dense = make_dense_retriever(store, embedder)
        index = BM25Index(store.iter_chunks())
        bm25 = make_bm25_retriever(index)
        return make_hybrid_retriever([dense, bm25])
    if name == "rerank":
        dense = make_dense_retriever(store, embedder)
        chunks = store.iter_chunks()
        index = BM25Index(chunks)
        bm25 = make_bm25_retriever(index)
        hybrid = make_hybrid_retriever([dense, bm25])
        chunk_text_by_id = {c.chunk_id: c.text for c in chunks}
        reranker = CrossEncoderReranker()
        return make_reranked_retriever(hybrid, reranker, chunk_text_by_id.__getitem__)
    raise ValueError(f"unknown retriever {name!r}")


def _to_citation_out(citation: Any) -> CitationOut:
    return CitationOut(
        chunk_id=citation.chunk_id,
        doc_id=citation.doc_id,
        score=float(citation.score),
        snippet=citation.snippet,
    )


def _to_tool_call_out(tc: Any) -> ToolCallOut:
    return ToolCallOut(
        name=tc.name,
        arguments=dict(tc.arguments),
        output=dict(tc.output),
    )


# Module-level app for production (`uvicorn job_application_insights.api.main:app`).
# This is what Dockerfile's CMD references. Tests build their own app via
# create_app(state=...).
app = create_app()

# Suppress unused-import / unused-import-warning for type-only imports.
__all__ = [
    "AGENT_PROVIDER_NAMES",
    "PROVIDER_NAMES",
    "AskRequest",
    "AskResponse",
    "CitationOut",
    "HealthResponse",
    "ToolCallOut",
    "app",
    "create_app",
]
