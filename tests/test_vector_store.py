"""Tests for :mod:`job_application_insights.retrieval.vector_store`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from job_application_insights.ingest.chunk import Chunk
from job_application_insights.retrieval.vector_store import (
    DEFAULT_COLLECTION_NAME,
    RetrievalResult,
    VectorStore,
)
from pydantic import ValidationError

# ───── helpers ─────


def _chunk(idx: int = 0, text: str = "alpha", doc: str = "msg_001") -> Chunk:
    return Chunk(
        chunk_id=f"{doc}__c{idx:03d}",
        doc_id=doc,
        chunk_index=idx,
        text=text,
        n_tokens=5,
        sender=f"sender_{idx}@x.com",
        subject=f"Subject {idx}",
        date=datetime(2026, 6, 17, 12, 0, idx, tzinfo=UTC),
    )


def _unit(vec: list[float]) -> np.ndarray:
    """Return ``vec`` as a unit-norm float32 vector — matches Embedder output."""
    arr = np.array(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr /= norm
    return arr


def _embeddings(*vecs: list[float]) -> np.ndarray:
    return np.stack([_unit(v) for v in vecs])


# ───── init & basics ─────


def test_init_creates_persist_path(tmp_path: Path):
    persist = tmp_path / "chroma_db"
    assert not persist.exists()
    VectorStore(persist)
    assert persist.exists()


def test_default_collection_name():
    assert DEFAULT_COLLECTION_NAME == "chunks"


def test_n_chunks_starts_at_zero(tmp_path: Path):
    store = VectorStore(tmp_path / "store")
    assert store.n_chunks == 0


def test_query_empty_store_returns_empty(tmp_path: Path):
    store = VectorStore(tmp_path / "store")
    results = store.query(_unit([1.0, 0.0, 0.0]), k=5)
    assert results == []


# ───── upsert ─────


def test_upsert_increases_count(tmp_path: Path):
    store = VectorStore(tmp_path / "store")
    chunks = [_chunk(0), _chunk(1)]
    embeddings = _embeddings([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    store.upsert(chunks, embeddings)
    assert store.n_chunks == 2


def test_upsert_empty_input_is_noop(tmp_path: Path):
    store = VectorStore(tmp_path / "store")
    store.upsert([], np.zeros((0, 3), dtype=np.float32))
    assert store.n_chunks == 0


def test_upsert_is_idempotent(tmp_path: Path):
    """Re-upserting the same chunk_ids should replace, not duplicate."""
    store = VectorStore(tmp_path / "store")
    chunks = [_chunk(0), _chunk(1)]
    emb = _embeddings([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    store.upsert(chunks, emb)
    store.upsert(chunks, emb)
    assert store.n_chunks == 2


def test_upsert_rejects_misaligned_inputs(tmp_path: Path):
    store = VectorStore(tmp_path / "store")
    chunks = [_chunk(0), _chunk(1)]
    embeddings = _embeddings([1.0, 0.0, 0.0])  # only 1 vector, 2 chunks
    with pytest.raises(ValueError, match="length mismatch"):
        store.upsert(chunks, embeddings)


# ───── query ─────


def test_query_returns_most_similar_chunk_first(tmp_path: Path):
    store = VectorStore(tmp_path / "store")
    chunks = [
        _chunk(0, text="alpha"),
        _chunk(1, text="beta"),
        _chunk(2, text="gamma"),
    ]
    # Three orthogonal unit vectors
    embeddings = _embeddings(
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )
    store.upsert(chunks, embeddings)

    # Query close to chunk[1] (beta)
    results = store.query(_unit([0.1, 0.9, 0.0]), k=3)
    assert len(results) == 3
    assert results[0].chunk.text == "beta"
    # Scores in descending order
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_query_score_is_one_for_identical_vector(tmp_path: Path):
    store = VectorStore(tmp_path / "store")
    v = _unit([1.0, 2.0, 3.0])
    store.upsert([_chunk(0)], np.expand_dims(v, 0))
    results = store.query(v, k=1)
    assert len(results) == 1
    # Identical normalised vectors → similarity ~1.0 (tiny float drift OK)
    assert abs(results[0].score - 1.0) < 1e-5


def test_query_caps_k_at_n_chunks(tmp_path: Path):
    store = VectorStore(tmp_path / "store")
    chunks = [_chunk(0), _chunk(1)]
    embeddings = _embeddings([1.0, 0.0], [0.0, 1.0])
    store.upsert(chunks, embeddings)
    results = store.query(_unit([1.0, 0.0]), k=100)
    assert len(results) == 2


def test_query_rejects_2d_input(tmp_path: Path):
    store = VectorStore(tmp_path / "store")
    store.upsert([_chunk(0)], _embeddings([1.0, 0.0]))
    with pytest.raises(ValueError, match="1-D"):
        store.query(np.zeros((1, 2), dtype=np.float32), k=1)


def test_query_rejects_non_positive_k(tmp_path: Path):
    store = VectorStore(tmp_path / "store")
    store.upsert([_chunk(0)], _embeddings([1.0, 0.0]))
    with pytest.raises(ValueError, match="k must be positive"):
        store.query(_unit([1.0, 0.0]), k=0)


# ───── metadata round-trip ─────


def test_query_preserves_chunk_metadata(tmp_path: Path):
    store = VectorStore(tmp_path / "store")
    when = datetime(2026, 6, 17, 12, 34, 56, tzinfo=UTC)
    chunk = Chunk(
        chunk_id="msg_001__c000",
        doc_id="msg_001",
        chunk_index=7,
        text="thank you for applying",
        n_tokens=42,
        sender="ats@acme.com",
        subject="Re: ML Engineer role",
        date=when,
    )
    store.upsert([chunk], _embeddings([1.0, 0.0]))
    [result] = store.query(_unit([1.0, 0.0]), k=1)
    rt = result.chunk
    assert rt.chunk_id == chunk.chunk_id
    assert rt.doc_id == chunk.doc_id
    assert rt.chunk_index == chunk.chunk_index
    assert rt.text == chunk.text
    assert rt.n_tokens == chunk.n_tokens
    assert rt.sender == chunk.sender
    assert rt.subject == chunk.subject
    assert rt.date == chunk.date


def test_query_handles_chunk_without_date(tmp_path: Path):
    store = VectorStore(tmp_path / "store")
    chunk = Chunk(
        chunk_id="x__c000",
        doc_id="x",
        chunk_index=0,
        text="t",
        n_tokens=1,
        date=None,
    )
    store.upsert([chunk], _embeddings([1.0, 0.0]))
    [result] = store.query(_unit([1.0, 0.0]), k=1)
    assert result.chunk.date is None


# ───── persistence ─────


def test_store_survives_reopen(tmp_path: Path):
    """Write, close (let it go out of scope), re-open, query — data intact."""
    persist = tmp_path / "store"
    chunks = [_chunk(0, text="persisted_chunk")]
    embeddings = _embeddings([1.0, 2.0, 3.0])

    store1 = VectorStore(persist)
    store1.upsert(chunks, embeddings)
    assert store1.n_chunks == 1
    del store1

    store2 = VectorStore(persist)
    assert store2.n_chunks == 1
    results = store2.query(_unit([1.0, 2.0, 3.0]), k=1)
    assert len(results) == 1
    assert results[0].chunk.text == "persisted_chunk"


# ───── clear ─────


def test_clear_removes_everything(tmp_path: Path):
    store = VectorStore(tmp_path / "store")
    store.upsert(
        [_chunk(0), _chunk(1)],
        _embeddings([1.0, 0.0], [0.0, 1.0]),
    )
    assert store.n_chunks == 2
    store.clear()
    assert store.n_chunks == 0


def test_clear_on_empty_store_is_safe(tmp_path: Path):
    store = VectorStore(tmp_path / "store")
    store.clear()
    assert store.n_chunks == 0


# ───── RetrievalResult invariants ─────


def test_retrieval_result_is_frozen():
    chunk = _chunk(0)
    result = RetrievalResult(chunk=chunk, score=0.5)
    with pytest.raises(ValidationError):
        result.score = 0.9  # type: ignore[misc]


def test_retrieval_result_accepts_unbounded_scores():
    """Score is unbounded — different retrievers (dense, BM25, reranker)
    use incomparable scales. Hybrid fusion uses ranks, so the model
    doesn't constrain the score field."""
    chunk = _chunk(0)
    RetrievalResult(chunk=chunk, score=-1.0)
    RetrievalResult(chunk=chunk, score=1.0)
    RetrievalResult(chunk=chunk, score=42.5)  # BM25-style positive
    RetrievalResult(chunk=chunk, score=-0.27)  # rank_bm25 epsilon edge case
