"""Persistent vector store over Chroma — upsert chunks, query by embedding.

Why a vector store at all (vs. just a numpy array):

* **Persistence**: load/save the index without re-embedding. Embedding the
  corpus takes minutes; loading from disk takes milliseconds.
* **Metadata + filtering**: every chunk carries ``doc_id``, ``sender``,
  ``subject``, ``date``. The store can return them alongside the chunk and
  (Week 2+) filter by them at query time.
* **ANN at scale**: Chroma uses HNSW under the hood. We don't need it for
  300 chunks, but the same API scales to 10M without code changes.
* **Single source of truth**: one place that knows where vectors live.

Storage layout:

* One Chroma "document" per :class:`Chunk`, keyed by ``chunk_id``.
* The chunk text lives in Chroma's ``documents`` field.
* Everything else (``doc_id``, ``chunk_index``, ``n_tokens``, ``sender``,
  ``subject``, ``date``) lives in ``metadatas``.
* The embedding lives in Chroma's vector index.
* The collection is configured for cosine distance (``hnsw:space=cosine``),
  which matches the L2-normalised vectors our :class:`Embedder` produces.

Design choices:

* **Explicit embeddings, not implicit**. We never let Chroma pick its own
  embedding model — we pass vectors we computed ourselves. Cleaner contract,
  reproducible, swappable.
* **Upsert is idempotent**. Re-running ingestion on the same chunks updates
  existing rows; it never raises ``DuplicateError``. Makes the pipeline safe
  to retry.
* **A typed RetrievalResult**. Queries return a list of ``RetrievalResult``
  (Pydantic, frozen) — never a raw Chroma response dict. This keeps the
  retrieval contract stable when we swap Chroma for Qdrant in Week 2.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from job_application_insights.ingest.chunk import Chunk
from job_application_insights.ingest.embed import assert_aligned

# ────────────────────────────── constants ──────────────────────────────

DEFAULT_COLLECTION_NAME: str = "chunks"
"""Single collection name we use across the project. Multiple collections
would be a Week 2+ concern (e.g. separate index per data source)."""


# ────────────────────────────── data model ──────────────────────────────


class RetrievalResult(BaseModel):
    """One hit from a vector-store query.

    Attributes
    ----------
    chunk
        The reconstructed :class:`Chunk` — text + metadata.
    score
        Cosine similarity in ``[-1.0, 1.0]``. Higher = more similar.
        Chroma natively returns cosine *distance* (``1 - similarity``); we
        convert at the boundary so downstream code never has to remember.
    """

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = Field(..., ge=-1.0, le=1.0)


# ────────────────────────────── helpers ──────────────────────────────


def _chunk_to_metadata(chunk: Chunk) -> dict[str, str | int]:
    """Serialise a :class:`Chunk` for Chroma's metadata field.

    Chroma metadata values must be primitives (``str``, ``int``, ``float``,
    ``bool``). We store the date as an ISO-8601 string and the missing-date
    sentinel as the empty string.
    """
    return {
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "n_tokens": chunk.n_tokens,
        "sender": chunk.sender,
        "subject": chunk.subject,
        "date": chunk.date.isoformat() if chunk.date else "",
    }


def _metadata_to_chunk(chunk_id: str, text: str, meta: dict[str, Any]) -> Chunk:
    """Rebuild a :class:`Chunk` from a Chroma metadata dict."""
    date_str = str(meta.get("date") or "")
    date = datetime.fromisoformat(date_str) if date_str else None
    return Chunk(
        chunk_id=chunk_id,
        doc_id=str(meta["doc_id"]),
        chunk_index=int(meta["chunk_index"]),
        text=text,
        n_tokens=int(meta["n_tokens"]),
        sender=str(meta.get("sender") or ""),
        subject=str(meta.get("subject") or ""),
        date=date,
    )


# ────────────────────────────── public API ──────────────────────────────


class VectorStore:
    """Typed wrapper around a Chroma persistent collection.

    Parameters
    ----------
    persist_path
        Directory where Chroma writes its SQLite + HNSW index. Created if
        missing. Use a ``tmp_path`` in tests; a stable folder for production.
    collection_name
        Name of the collection within the Chroma database. Default is
        :data:`DEFAULT_COLLECTION_NAME`.

    Notes
    -----
    The store is opened lazily — instantiating the class is cheap; the index
    is loaded on first read/write.
    """

    def __init__(
        self,
        persist_path: Path | str,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(path=str(self.persist_path))
        # `hnsw:space=cosine` matches our L2-normalised embeddings. Switching
        # to `ip` (inner product) or `l2` would silently break similarity math.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def n_chunks(self) -> int:
        """Total number of chunks currently stored."""
        return int(self._collection.count())

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Insert or replace ``chunks`` with their ``embeddings``.

        Idempotent: re-upserting the same ``chunk_id`` replaces the existing
        row. Empty input is a no-op.
        """
        if not chunks:
            return
        assert_aligned(chunks, embeddings)

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[_chunk_to_metadata(c) for c in chunks],
        )

    def query(
        self,
        query_embedding: np.ndarray,
        *,
        k: int = 8,
    ) -> list[RetrievalResult]:
        """Return the top-``k`` chunks most similar to ``query_embedding``.

        Parameters
        ----------
        query_embedding
            1-D ndarray of shape ``(dimension,)`` — typically produced by
            ``Embedder.embed([query_text])[0]``.
        k
            Maximum number of results to return. Silently capped at
            ``n_chunks`` if larger.

        Returns
        -------
        A list of :class:`RetrievalResult`, sorted descending by ``score``.
        Empty list if the store has no chunks.
        """
        if query_embedding.ndim != 1:
            raise ValueError(f"query_embedding must be 1-D, got shape {query_embedding.shape}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if self.n_chunks == 0:
            return []

        k = min(k, self.n_chunks)
        raw = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )

        # Chroma returns lists-of-lists keyed by the batched query.
        # We sent one query → results live in index 0 of each list.
        # Each include= field comes back as `list[list[T]] | None`; narrow it.
        ids = raw["ids"][0]
        raw_docs = raw.get("documents")
        raw_metas = raw.get("metadatas")
        raw_dists = raw.get("distances")
        texts = raw_docs[0] if raw_docs is not None else [""] * len(ids)
        metas = raw_metas[0] if raw_metas is not None else [{}] * len(ids)
        distances = raw_dists[0] if raw_dists is not None else [0.0] * len(ids)

        out: list[RetrievalResult] = []
        for chunk_id, text, meta, dist in zip(ids, texts, metas, distances, strict=True):
            chunk = _metadata_to_chunk(chunk_id, text or "", dict(meta))
            # cosine distance ∈ [0, 2]; similarity = 1 - distance ∈ [-1, 1]
            similarity = max(-1.0, min(1.0, 1.0 - float(dist)))
            out.append(RetrievalResult(chunk=chunk, score=similarity))
        return out

    def clear(self) -> None:
        """Delete every chunk in the collection. Useful for fresh re-ingest."""
        existing = self._collection.get()
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
