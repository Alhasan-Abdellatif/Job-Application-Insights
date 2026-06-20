"""Cross-encoder reranking — the precision-fixing stage.

Yesterday's hybrid + RRF retriever traded a clean dense win on the
Amazon question (R@8 = 1.000) for a noise-diluted 0.250. The reason
was structural: RRF gives both retrievers equal vote, so BM25's
top results for "Amazon" (AWS marketing, billing alerts, LinkedIn
suggestions) competed on equal footing with dense's correct
application acks.

A cross-encoder fixes that by scoring each ``(query, chunk)`` pair
*jointly* — encoder attention runs over both at once — and producing
a calibrated relevance score per pair. Anything that doesn't actually
answer the query gets pushed down regardless of which retriever
surfaced it. We rerank only the top-N fused candidates so the cost
stays bounded (50 cross-encoder calls per query is fast on CPU).

Architecture
------------
Bi-encoder (Week 1 BGE-small):

    query  ──► encode ──► q_vec ─┐
                                  ├── cosine
    chunk  ──► encode ──► c_vec ─┘
    (encoded independently, indexed offline, microsecond query)

Cross-encoder (today's BGE-reranker):

    [query] [SEP] [chunk]  ──► encode  ──► score
    (encoded together, no pre-computation possible, ~10ms per pair)

The cost difference is exactly the point: bi-encoder for the wide
funnel, cross-encoder for the tight filter.

What this module provides
-------------------------
* :class:`Reranker` Protocol — minimum interface: ``rerank(query,
  items) -> sorted list of (chunk_id, score)``.
* :class:`CrossEncoderReranker` — sentence-transformers backed
  implementation using ``BAAI/bge-reranker-base`` by default.
* :func:`make_reranked_retriever` — wraps any base :data:`RetrievalFn`
  with a reranker. Casts a net of ``fetch_k`` candidates from the base,
  reranks all of them, returns the top-K.

The Protocol seam means tests can plug in a deterministic stub
reranker without loading a 280MB cross-encoder, and tomorrow's
reranker swap (say `bge-reranker-v2-m3` for higher quality) is a
one-line change in the CLI.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

from sentence_transformers import CrossEncoder

if TYPE_CHECKING:
    from job_application_insights.evals.runner import RetrievalFn

# ────────────────────────────── constants ──────────────────────────────


DEFAULT_RERANKER_MODEL: str = "BAAI/bge-reranker-base"
"""Same lineage as the BGE embedder, ~280 MB, runs on CPU at ~10ms/pair.
Swap to ``BAAI/bge-reranker-v2-m3`` (bigger, multilingual, better quality)
or ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (~90 MB, faster) without
touching anything else in the pipeline."""

DEFAULT_FETCH_K: int = 50
"""How many candidates to pull from the base retriever before reranking.
50 is the field standard — gives the cross-encoder enough range to
surface buried gems without making per-query cost balloon. Smaller
values risk under-recall (relevant chunk never made the shortlist),
larger values cost linearly more cross-encoder calls."""


# ────────────────────────────── public API ──────────────────────────────


class Reranker(Protocol):
    """Minimum interface a reranker must implement.

    Implementations score ``(query, chunk_text)`` pairs and return them
    in descending order of relevance. Score scale is implementation-
    defined (cross-encoders typically emit raw logits in ``[-10, 10]``,
    others may normalize); downstream code only relies on the *order*,
    not the absolute values.
    """

    def rerank(
        self,
        query: str,
        items: Sequence[tuple[str, str]],
    ) -> list[tuple[str, float]]: ...


class CrossEncoderReranker:
    """``sentence_transformers.CrossEncoder``-backed :class:`Reranker`."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(
        self,
        query: str,
        items: Sequence[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        """Score every ``(chunk_id, text)`` pair against ``query`` and sort.

        Empty input is a no-op (returns ``[]``).
        """
        if not items:
            return []
        pairs: list[list[str]] = [[query, text] for _, text in items]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        ranked = [
            (chunk_id, float(score)) for (chunk_id, _), score in zip(items, scores, strict=False)
        ]
        ranked.sort(key=lambda kv: kv[1], reverse=True)
        return ranked


def make_reranked_retriever(
    base: RetrievalFn,
    reranker: Reranker,
    chunk_text_lookup: Callable[[str], str],
    *,
    fetch_k: int = DEFAULT_FETCH_K,
) -> RetrievalFn:
    """Wrap ``base`` with a reranking stage.

    Pipeline at query time:

    1. ``base(query, fetch_k)`` — wide cheap funnel.
    2. ``chunk_text_lookup(chunk_id)`` — text reconstitution for each
       candidate. The lookup function is your seam for where the chunks
       live (in-memory dict, VectorStore, etc.).
    3. ``reranker.rerank(query, items)`` — joint cross-encoder scoring.
    4. Slice top-``k`` (whatever the caller asked for) off the result.

    Parameters
    ----------
    base
        Any :data:`RetrievalFn` — dense, BM25, hybrid, or even another
        reranked retriever (though that's wasteful).
    reranker
        Anything satisfying the :class:`Reranker` Protocol.
    chunk_text_lookup
        ``chunk_id → text``. Raised exceptions propagate, so a missing
        ID is loud (probably means the base retriever returned an ID
        that's no longer in the corpus).
    fetch_k
        Candidates pulled from ``base`` before reranking. Default 50.

    Returns
    -------
    A :data:`RetrievalFn` ready to slot into the eval harness or CLI.
    """
    if fetch_k <= 0:
        raise ValueError(f"fetch_k must be positive, got {fetch_k}")

    def retrieve(query: str, k: int) -> list[str]:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        candidate_ids = base(query, fetch_k)
        if not candidate_ids:
            return []
        items: list[tuple[str, str]] = [(cid, chunk_text_lookup(cid)) for cid in candidate_ids]
        ranked = reranker.rerank(query, items)
        return [chunk_id for chunk_id, _ in ranked[:k]]

    return retrieve
