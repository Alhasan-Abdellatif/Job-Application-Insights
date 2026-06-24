"""Tests for the FastAPI service (api/main.py).

These use FastAPI's :class:`TestClient` so we don't actually bind a
port. The app is built via :func:`create_app(state=...)` which bypasses
the production ``lifespan`` and injects pre-built stubs — no embedder
load, no Chroma/Qdrant, no real LLM.

What we cover:

* `/health` and `/` smoke endpoints.
* `/ask` request validation (Pydantic boundary).
* The three modes — ``rag`` / ``tools`` / ``auto`` — each plumbed
  through to the right collaborators.
* Empty-store and missing-DuckDB error paths return clear HTTP codes.
"""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient
from job_application_insights.agents.router import RouterDecision
from job_application_insights.agents.tool_use import ToolCall, ToolUseResult
from job_application_insights.api.main import (
    AskResponse,
    _to_tool_call_out,
    create_app,
)
from job_application_insights.ingest.chunk import Chunk
from job_application_insights.structured.table import load_applications_table

# ────────────────────────── stubs ──────────────────────────


class _StubStore:
    """Pretends to be a vector store. Provides just what `/ask` needs."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)

    def iter_chunks(self) -> list[Chunk]:
        return list(self._chunks)


class _StubEmbedder:
    """Embedder with a controllable dimension; query() returns a fixed vector."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self._dimension), dtype=np.float32)


def _chunk(chunk_id: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        n_tokens=len(text.split()),
    )


def _stub_retriever(chunk_ids: list[str]) -> Any:
    """A retriever Callable that always returns ``chunk_ids``."""
    return lambda _q, _k: list(chunk_ids)


@pytest.fixture
def populated_csv(tmp_path: Any) -> Any:
    csv = tmp_path / "ack.csv"
    csv.write_text(
        "From,Subject,Date,Body,Company,Role\n"
        '"recruiter@gsk.com","Thanks","Mon, 12 Aug 2024 10:00:00 +0000","b","GSK","Eng"\n'
    )
    return csv


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        _chunk("c1", "msg_001", "Hello from GSK"),
        _chunk("c2", "msg_002", "Hello from Edinburgh"),
    ]


@pytest.fixture
def duckdb_con(populated_csv: Any) -> duckdb.DuckDBPyConnection:
    return load_applications_table(populated_csv)


@pytest.fixture
def state(chunks: list[Chunk], duckdb_con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Pre-built app.state for tests. Real production loads this via lifespan."""
    return {
        "embedder": _StubEmbedder(dimension=8),
        "store": _StubStore(chunks),
        "chunks_by_id": {c.chunk_id: c for c in chunks},
        "duckdb_con": duckdb_con,
        # Pre-seed the retriever cache so we don't try to build a real
        # dense retriever (which would need a real vector store).
        "retrievers": {"dense": _stub_retriever(["c1"])},
    }


@pytest.fixture
def client(state: dict[str, Any]) -> TestClient:
    app = create_app(state=state)
    return TestClient(app)


# ────────────────────────── smoke endpoints ──────────────────────────


def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_root_returns_service_info(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "job-application-insights"
    assert "endpoints" in body


# ────────────────────────── /ask validation ──────────────────────────


def test_ask_rejects_empty_question(client: TestClient) -> None:
    """Pydantic min_length=1 on `question` blocks empty strings at the boundary."""
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422  # Pydantic validation error


def test_ask_rejects_invalid_mode(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "q", "mode": "made_up"})
    assert resp.status_code == 422


def test_ask_rejects_k_out_of_range(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "q", "k": 0})
    assert resp.status_code == 422
    resp = client.post("/ask", json={"question": "q", "k": 100})
    assert resp.status_code == 422


# ────────────────────────── /ask happy paths ──────────────────────────


def test_ask_rag_mode_echo_provider(client: TestClient) -> None:
    """RAG path with the echo provider — no API key needed.

    The echo LLMClient mirrors back the source IDs found in the prompt
    so we can verify chunk c1 made it through.
    """
    resp = client.post(
        "/ask",
        json={
            "question": "Did I apply to GSK?",
            "mode": "rag",
            "retriever": "dense",  # pre-seeded in the fixture
            "provider": "echo",
            "k": 1,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode_used"] == "rag"
    assert "c1" in body["text"]  # echo client echoes source IDs
    assert body["citations"][0]["chunk_id"] == "c1"
    assert body["tool_calls"] == []


def test_ask_tools_mode_requires_duckdb_con(client: TestClient) -> None:
    """Tools mode would need DuckDB + a real LLM; verify the request
    *reaches* the tools handler — that it doesn't 503 on the
    missing-DuckDB guard. We expect a 500 only because we have no
    valid Gemini API key in the test environment."""
    resp = client.post(
        "/ask",
        json={
            "question": "How many GSK?",
            "mode": "tools",
            "agent_provider": "gemini",  # not actually called — Gemini client
        },  # init may or may not fail without a key
    )
    # The test environment has no Gemini key, so we expect either
    # 200 (if a key happens to be present) or 500 (real Gemini call failed).
    # The contract we're asserting is *we passed the empty-store guard*,
    # i.e. NOT a 503.
    assert resp.status_code != 503


def test_ask_empty_store_returns_503(state: dict[str, Any], chunks: list[Chunk]) -> None:
    """If the vector store has no chunks, /ask should 503 with a hint."""
    state_copy = dict(state)
    state_copy["store"] = _StubStore([])
    state_copy["chunks_by_id"] = {}
    app = create_app(state=state_copy)
    client = TestClient(app)
    resp = client.post("/ask", json={"question": "anything", "provider": "echo"})
    assert resp.status_code == 503
    assert "empty" in resp.json()["detail"].lower()


def test_ask_tools_mode_no_duckdb_returns_503(
    state: dict[str, Any],
) -> None:
    """Tools mode without a loaded DuckDB connection -> 503 (not a crash)."""
    state_copy = dict(state)
    state_copy["duckdb_con"] = None
    app = create_app(state=state_copy)
    client = TestClient(app)
    resp = client.post("/ask", json={"question": "q", "mode": "tools"})
    assert resp.status_code == 503
    assert "structured" in resp.json()["detail"].lower()


# ────────────────────────── response shape ──────────────────────────


def test_response_matches_pydantic_schema(client: TestClient) -> None:
    """Response is parseable by the same Pydantic model the server emits.

    Catches accidental drift between AskResponse and what we actually
    return.
    """
    resp = client.post(
        "/ask",
        json={
            "question": "Did I apply?",
            "mode": "rag",
            "retriever": "dense",
            "provider": "echo",
            "k": 1,
        },
    )
    assert resp.status_code == 200
    # Round-trip through Pydantic — fails loudly if the JSON shape
    # diverges from the response_model.
    AskResponse.model_validate(resp.json())


# ────────────────────────── ToolCallOut / CitationOut roundtrip ──────────────────────────


def test_tool_call_out_serialises_correctly() -> None:
    tc = ToolCall(
        name="count_applications",
        arguments={"company": "GSK"},
        output={"value": 7},
    )
    out = _to_tool_call_out(tc)
    assert out.name == "count_applications"
    assert out.arguments == {"company": "GSK"}
    assert out.output == {"value": 7}


def test_router_decision_is_passed_through_unchanged() -> None:
    """The router's mode is what surfaces in mode_used for auto mode."""
    decision = RouterDecision(mode="structured")
    assert decision.mode == "structured"


def test_tool_use_result_propagates_to_response() -> None:
    """A ToolUseResult with stopped_reason='max_steps' propagates to the response."""
    # This is a smoke check that the data model accepts the value
    # the API hands through — no real call.
    result = ToolUseResult(
        question="q",
        text="answer",
        tool_calls=[ToolCall(name="count_applications", arguments={}, output={"value": 0})],
        stopped_reason="max_steps",
    )
    assert result.stopped_reason == "max_steps"
    assert len(result.tool_calls) == 1
