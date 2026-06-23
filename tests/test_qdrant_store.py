"""Tests for the Qdrant-backed vector store.

We use Qdrant's ``:memory:`` mode so these run with no docker, no network,
no port. The API surface is identical to the live HTTP client, so any
test that passes here would also pass against a real Qdrant service.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from job_application_insights.ingest.chunk import Chunk
from job_application_insights.retrieval.qdrant_store import (
    QdrantVectorStore,
    _chunk_id_to_point_id,
)
from job_application_insights.retrieval.vector_store import (
    RetrievalResult,
    make_vector_store,
)


def _chunk(chunk_id: str, doc_id: str, text: str, *, n_tokens: int = 5) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        n_tokens=n_tokens,
        sender="test@example.com",
        subject="Subject",
        date=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _embedding(seed: float, dim: int = 8) -> np.ndarray:
    """Build a deterministic unit vector for tests.

    Cosine similarity between two such vectors with the same seed is
    1.0; vectors with different seeds get progressively smaller
    similarity. Tiny dim (8) keeps test storage cheap.
    """
    rng = np.random.default_rng(int(seed * 1000))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


@pytest.fixture
def store() -> QdrantVectorStore:
    """Fresh in-memory Qdrant for each test."""
    return QdrantVectorStore(
        url=":memory:",
        collection_name="test_chunks",
        vector_size=8,
    )


# ────────────────────────── deterministic point IDs ──────────────────────────


def test_chunk_id_to_point_id_is_deterministic() -> None:
    """Same chunk_id always maps to the same UUID — that's how upserts
    stay idempotent under the hood."""
    a = _chunk_id_to_point_id("msg_000123__c000")
    b = _chunk_id_to_point_id("msg_000123__c000")
    assert a == b


def test_chunk_id_to_point_id_collides_only_on_identical_input() -> None:
    a = _chunk_id_to_point_id("msg_000123__c000")
    b = _chunk_id_to_point_id("msg_000123__c001")
    assert a != b


# ────────────────────────── lifecycle ──────────────────────────


def test_empty_store_reports_zero_chunks(store: QdrantVectorStore) -> None:
    assert store.n_chunks == 0
    assert store.iter_chunks() == []


def test_query_on_empty_store_returns_empty(store: QdrantVectorStore) -> None:
    """Don't crash if asked to retrieve before anything was upserted."""
    results = store.query(_embedding(0.1), k=5)
    assert results == []


def test_upsert_with_empty_input_is_noop(store: QdrantVectorStore) -> None:
    store.upsert([], np.zeros((0, 8), dtype=np.float32))
    assert store.n_chunks == 0


# ────────────────────────── upsert + query ──────────────────────────


def test_upsert_then_count(store: QdrantVectorStore) -> None:
    chunks = [_chunk(f"msg_001__c{i:03d}", "msg_001", f"text {i}") for i in range(3)]
    embeddings = np.vstack([_embedding(0.1 * (i + 1)) for i in range(3)])
    store.upsert(chunks, embeddings)
    assert store.n_chunks == 3


def test_query_returns_results_sorted_by_similarity(store: QdrantVectorStore) -> None:
    """The chunk whose embedding equals the query should come first."""
    chunks = [
        _chunk("msg_001__c000", "msg_001", "A"),
        _chunk("msg_002__c000", "msg_002", "B"),
        _chunk("msg_003__c000", "msg_003", "C"),
    ]
    embeddings = np.vstack([_embedding(0.1), _embedding(0.5), _embedding(0.9)])
    store.upsert(chunks, embeddings)

    # Query exactly the second vector — it should rank top.
    results = store.query(_embedding(0.5), k=3)
    assert len(results) == 3
    assert results[0].chunk.chunk_id == "msg_002__c000"
    # Scores descend.
    assert results[0].score >= results[1].score >= results[2].score


def test_query_returns_correct_chunk_metadata(store: QdrantVectorStore) -> None:
    """Round-trip the chunk through Qdrant's payload — every field survives."""
    original = _chunk("msg_001__c000", "msg_001", "Body text", n_tokens=42)
    store.upsert([original], np.vstack([_embedding(0.1)]))
    results = store.query(_embedding(0.1), k=1)
    got = results[0].chunk
    assert got.chunk_id == original.chunk_id
    assert got.doc_id == original.doc_id
    assert got.text == original.text
    assert got.n_tokens == 42
    assert got.sender == original.sender
    assert got.subject == original.subject
    assert got.date == original.date


def test_query_respects_k(store: QdrantVectorStore) -> None:
    chunks = [_chunk(f"msg_001__c{i:03d}", "msg_001", "x") for i in range(5)]
    embeddings = np.vstack([_embedding(0.1 * (i + 1)) for i in range(5)])
    store.upsert(chunks, embeddings)
    assert len(store.query(_embedding(0.5), k=3)) == 3
    assert len(store.query(_embedding(0.5), k=10)) == 5  # capped at n_chunks


def test_query_returns_retrieval_result_type(store: QdrantVectorStore) -> None:
    """The contract is RetrievalResult — same as the Chroma store."""
    store.upsert([_chunk("msg_001__c000", "msg_001", "x")], np.vstack([_embedding(0.1)]))
    results = store.query(_embedding(0.1), k=1)
    assert isinstance(results[0], RetrievalResult)


# ────────────────────────── idempotent upsert ──────────────────────────


def test_re_upserting_same_chunk_id_replaces_not_duplicates(
    store: QdrantVectorStore,
) -> None:
    """The whole point of UUID5 from chunk_id: idempotent upserts."""
    chunk = _chunk("msg_001__c000", "msg_001", "original")
    store.upsert([chunk], np.vstack([_embedding(0.1)]))
    assert store.n_chunks == 1

    # Re-upsert with same chunk_id but different text.
    updated = _chunk("msg_001__c000", "msg_001", "replaced text")
    store.upsert([updated], np.vstack([_embedding(0.2)]))
    assert store.n_chunks == 1  # still one row

    # The newest text won.
    results = store.query(_embedding(0.2), k=1)
    assert results[0].chunk.text == "replaced text"


# ────────────────────────── iter_chunks ──────────────────────────


def test_iter_chunks_returns_all(store: QdrantVectorStore) -> None:
    chunks = [_chunk(f"msg_001__c{i:03d}", "msg_001", f"text {i}") for i in range(5)]
    store.upsert(chunks, np.vstack([_embedding(0.1 * (i + 1)) for i in range(5)]))
    iterated = store.iter_chunks()
    assert {c.chunk_id for c in iterated} == {c.chunk_id for c in chunks}


# ────────────────────────── clear ──────────────────────────


def test_clear_removes_all_chunks(store: QdrantVectorStore) -> None:
    store.upsert(
        [_chunk("msg_001__c000", "msg_001", "x")],
        np.vstack([_embedding(0.1)]),
    )
    assert store.n_chunks == 1
    store.clear()
    assert store.n_chunks == 0


# ────────────────────────── factory ──────────────────────────


def test_factory_returns_qdrant_store_for_qdrant_backend() -> None:
    store = make_vector_store("qdrant", qdrant_url=":memory:", vector_size=8)
    assert isinstance(store, QdrantVectorStore)


def test_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown vector-store backend"):
        make_vector_store("pinecone")  # not implemented (yet)


# ────────────────────────── shape-compatibility ──────────────────────────


def test_shape_compatible_with_chroma_interface() -> None:
    """Public method names + signatures match the Chroma VectorStore.

    Catches accidental drift between the two backends at the test layer
    rather than at the first surprising-error-in-production layer.
    """
    qdrant_methods = {"n_chunks", "iter_chunks", "upsert", "query", "clear"}
    actual = {m for m in dir(QdrantVectorStore) if not m.startswith("_")}
    assert qdrant_methods.issubset(actual)
