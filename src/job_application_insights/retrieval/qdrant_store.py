"""Qdrant-backed vector store — the Week-4 production swap for Chroma.

Shape-compatible with :class:`job_application_insights.retrieval.vector_store.VectorStore`:
same public method names, same return types, same idempotent upsert
contract. Anything that consumed the Chroma store will consume this
one with no other code changes.

What's different from Chroma:

* **Out-of-process service.** Qdrant runs in its own container (see
  ``docker-compose.yml``); this class talks to it over HTTP. The
  ``:memory:`` mode is identical-API for tests and CI — no docker.
* **HNSW with payload filtering**. Qdrant filters *during* the ANN
  search, not after, so metadata filters don't hurt recall.
* **Point IDs are int or UUID, not arbitrary strings.** Our chunk_ids
  (``msg_000123__c000``) get a deterministic UUID5 — same chunk_id
  always maps to the same point ID, so upserts stay idempotent. The
  original chunk_id is preserved in the payload for citation.
* **Score sign convention is the opposite of Chroma's.** Qdrant
  returns *similarity* (higher = more similar) directly; Chroma
  returns *distance* (lower = closer). We expose similarity to match
  the existing :class:`RetrievalResult` contract.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from job_application_insights.ingest.chunk import Chunk
from job_application_insights.ingest.embed import assert_aligned
from job_application_insights.retrieval.vector_store import RetrievalResult

DEFAULT_COLLECTION_NAME: str = "chunks"
DEFAULT_QDRANT_URL: str = "http://localhost:6333"

_UPSERT_BATCH_SIZE: int = 512
"""Qdrant accepts much larger batches than Chroma but we keep batches
modest for visible progress over the network."""

# Stable namespace UUID — never change this. Together with a chunk_id
# string it produces a deterministic UUID5, so the same chunk maps to
# the same Qdrant point ID across runs.
_CHUNK_ID_NAMESPACE = uuid.UUID("c1a4e1a8-3c2b-4b1f-9e1a-7c2b4b1f9e1a")


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Map our string chunk_id to a Qdrant-acceptable UUID."""
    return str(uuid.uuid5(_CHUNK_ID_NAMESPACE, chunk_id))


def _chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    """Serialise a :class:`Chunk` for Qdrant's payload field.

    Mirrors :func:`vector_store._chunk_to_metadata` but adds the
    original ``chunk_id`` (Qdrant's point ID is a UUID, so we keep
    the human-readable id in payload for citation).
    """
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "n_tokens": chunk.n_tokens,
        "sender": chunk.sender,
        "subject": chunk.subject,
        "date": chunk.date.isoformat() if chunk.date else "",
    }


def _payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    """Rebuild a :class:`Chunk` from a Qdrant payload."""
    date_str = str(payload.get("date") or "")
    date = datetime.fromisoformat(date_str) if date_str else None
    return Chunk(
        chunk_id=str(payload["chunk_id"]),
        doc_id=str(payload["doc_id"]),
        chunk_index=int(payload["chunk_index"]),
        text=str(payload.get("text") or ""),
        n_tokens=int(payload["n_tokens"]),
        sender=str(payload.get("sender") or ""),
        subject=str(payload.get("subject") or ""),
        date=date,
    )


class QdrantVectorStore:
    """Typed wrapper around a Qdrant collection.

    Parameters
    ----------
    url
        HTTP endpoint of the running Qdrant service, or the literal
        string ``":memory:"`` for an in-process store (used by tests).
    collection_name
        Name of the collection. Created on first ``upsert`` if missing.
    vector_size
        Dimensionality of the vectors. Must match the embedder. Default
        384 matches :class:`Embedder` with ``BAAI/bge-small-en-v1.5``.
    api_key
        Optional API key for cloud Qdrant. Unset for local docker.
    """

    def __init__(
        self,
        url: str = DEFAULT_QDRANT_URL,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = 384,
        api_key: str | None = None,
    ) -> None:
        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size

        if url == ":memory:":
            # In-memory mode — no server, no network. Perfect for tests.
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(url=url, api_key=api_key)

        # Lazy collection creation: only on first upsert, so just
        # instantiating the store has no side effects.
        self._ensured = False

    # ─── public API (shape-compatible with VectorStore) ───────────────

    @property
    def n_chunks(self) -> int:
        """Total number of points (chunks) in the collection."""
        if not self._collection_exists():
            return 0
        return int(self._client.count(self.collection_name, exact=True).count)

    def iter_chunks(self) -> list[Chunk]:
        """Reconstruct every stored :class:`Chunk` from Qdrant payloads.

        Uses Qdrant's ``scroll`` API to page through the whole
        collection. Order is whatever Qdrant returns (no insertion-order
        guarantee); callers needing stable ordering sort by ``chunk_id``.
        """
        if not self._collection_exists():
            return []
        chunks: list[Chunk] = []
        next_offset: Any = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                limit=512,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
            )
            for p in points:
                if p.payload is not None:
                    chunks.append(_payload_to_chunk(dict(p.payload)))
            if next_offset is None:
                break
        return chunks

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Insert or replace ``chunks`` with their ``embeddings``.

        Idempotent: re-upserting the same ``chunk_id`` replaces the
        existing point (same deterministic UUID maps to the same row).
        Empty input is a no-op. The collection is created on the first
        upsert if it doesn't exist yet.
        """
        if not chunks:
            return
        assert_aligned(chunks, embeddings)
        self._ensure_collection()

        for start in range(0, len(chunks), _UPSERT_BATCH_SIZE):
            end = start + _UPSERT_BATCH_SIZE
            batch = chunks[start:end]
            points = [
                PointStruct(
                    id=_chunk_id_to_point_id(c.chunk_id),
                    vector=embeddings[start + i].tolist(),
                    payload=_chunk_to_payload(c),
                )
                for i, c in enumerate(batch)
            ]
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def query(
        self,
        query_embedding: np.ndarray,
        *,
        k: int = 8,
    ) -> list[RetrievalResult]:
        """Return the top-``k`` chunks most similar to ``query_embedding``."""
        if query_embedding.ndim != 1:
            raise ValueError(f"query_embedding must be 1-D, got shape {query_embedding.shape}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if self.n_chunks == 0:
            return []

        k = min(k, self.n_chunks)
        # Qdrant's `query_points` (newer API) returns similarity scores
        # directly — for COSINE distance, higher is more similar.
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        out: list[RetrievalResult] = []
        for hit in response.points:
            if hit.payload is None:
                continue
            chunk = _payload_to_chunk(dict(hit.payload))
            # Qdrant returns the cosine similarity in [-1, 1] directly.
            out.append(RetrievalResult(chunk=chunk, score=float(hit.score)))
        return out

    def clear(self) -> None:
        """Drop and recreate the collection. Cheaper than deleting all points."""
        if self._collection_exists():
            self._client.delete_collection(self.collection_name)
        self._ensured = False

    # ─── internals ────────────────────────────────────────────────────

    def _collection_exists(self) -> bool:
        """Check the server, not the local ``_ensured`` flag.

        ``_ensured`` is a fast-path bypass; this method is the source of
        truth (e.g. someone else deleted the collection while we held a
        client).
        """
        return bool(self._client.collection_exists(self.collection_name))

    def _ensure_collection(self) -> None:
        """Create the collection with the right vector params if missing."""
        if self._ensured and self._collection_exists():
            return
        if not self._collection_exists():
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
        self._ensured = True
