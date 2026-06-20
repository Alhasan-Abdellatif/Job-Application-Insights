"""BM25 lexical retriever — the in-memory sparse counterpart to Chroma.

BM25 ("Best Matching 25") scores how well a document matches a query by
literal token overlap, with smart adjustments for term frequency,
inverse document frequency, and document length. It's the same family
of algorithms that powers Lucene and Elasticsearch. The math itself is
sixty lines of NumPy, but we use :mod:`rank_bm25` so we don't reinvent
that wheel.

Why we bother
-------------
Dense embeddings (Week 1) are great at *concepts*. They lose at
*specific tokens*. ``BAAI/bge-small-en-v1.5`` does not have a strong
notion of "Qualcomm" — it shares a region of vector space with every
other "thank-you-for-applying" email regardless of company. BM25
treats "Qualcomm" as an irreducible token that either matches or
doesn't, and IDF makes a rare match worth a lot.

What this module provides
-------------------------
* :class:`BM25Index` — wraps a corpus of :class:`Chunk` objects and
  exposes :meth:`BM25Index.query`. The interface mirrors
  :class:`~job_application_insights.retrieval.vector_store.VectorStore`
  so callers see one shape regardless of which retriever they hold.
* :func:`tokenize` — the tokenization step, exposed because consistent
  tokenization between index and query is the single most important
  BM25 correctness invariant. Doing it once in this module ensures the
  index and the queries never drift apart.

Persistence
-----------
The index is in-memory only. Rebuilding for ~7k chunks takes a fraction
of a second, so persistence is a non-feature. A disk-backed BM25 (or
SQLite FTS5) is a perfectly reasonable next step if the corpus grows;
the interface stays the same.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from job_application_insights.ingest.chunk import Chunk
from job_application_insights.retrieval.vector_store import RetrievalResult

# ────────────────────────────── constants ──────────────────────────────


_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[a-z0-9]+")
"""Bag-of-words tokenizer: lowercase ASCII alphanumerics. Punctuation
and case are discarded; multi-script content is dropped. Coarse, but
right for the BM25-as-second-signal role: this index is meant to catch
exact-token wins (``qualcomm``, ``gsk``, ``ETH``) that the dense model
misses, not to nail morphology. Stopwords are *not* stripped — they
don't hurt BM25 much (IDF will down-weight them naturally) and keeping
them avoids matching ``"a"`` in ``"a/b testing"`` to nothing."""


# ────────────────────────────── public API ──────────────────────────────


def tokenize(text: str) -> list[str]:
    """Split ``text`` into bag-of-words tokens for BM25.

    Same function is used to tokenize both the corpus at index time and
    the query at search time — the BM25 contract requires this. If you
    change the tokenization, rebuild the index.
    """
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """In-memory BM25 index over a list of :class:`Chunk` objects.

    Construction tokenizes every chunk and builds the rank_bm25 index.
    Query tokenizes the question and asks BM25Okapi for scores against
    every doc, then returns the top-K.

    Parameters
    ----------
    chunks
        The full corpus to index. **Order is preserved** — score
        positions returned by rank_bm25 map back to this list. Empty
        input is permitted; queries will return ``[]``.
    """

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self._chunks: list[Chunk] = list(chunks)
        # rank_bm25 requires a non-empty corpus to compute IDF.
        # We tolerate the empty case to make plumbing easy; queries
        # just short-circuit to an empty list.
        if self._chunks:
            self._tokenized: list[list[str]] = [tokenize(c.text) for c in self._chunks]
            self._token_sets: list[set[str]] = [set(toks) for toks in self._tokenized]
            self._bm25: BM25Okapi | None = BM25Okapi(self._tokenized)
        else:
            self._tokenized = []
            self._token_sets = []
            self._bm25 = None

    @property
    def n_chunks(self) -> int:
        """Number of chunks in the index."""
        return len(self._chunks)

    def query(self, query: str, *, k: int = 8) -> list[RetrievalResult]:
        """Return the top-``k`` chunks by BM25 score (descending).

        Parameters
        ----------
        query
            Natural-language question. Tokenized with :func:`tokenize`.
        k
            Maximum number of results to return. Capped at corpus size.

        Notes
        -----
        BM25 scores are unbounded (typically in ``[0, 30]`` for short
        English text). We return the raw score on the :class:`RetrievalResult`
        — *do not* compare it against a cosine similarity from the dense
        store directly. Use RRF (see :mod:`.hybrid`) for fusion.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not self._chunks or self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        query_set = set(tokens)
        scores: np.ndarray = np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)
        # argsort + slice is O(N log N); for small K an `argpartition`
        # would be O(N) — but our corpora are <100k chunks so the
        # simplicity wins.
        top_idx = np.argsort(scores)[::-1][:k]
        # Filter on literal token overlap (not score > 0). When a query
        # token has IDF=0 in a small corpus, BM25 returns score 0 even
        # though there IS overlap; the score filter would wrongly drop
        # those. The set-intersection check distinguishes "no signal"
        # from "weak signal".
        return [
            RetrievalResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_idx
            if query_set & self._token_sets[i]
        ]

    def chunk_ids(self) -> Sequence[str]:
        """The full ordered list of indexed chunk IDs (handy for tests)."""
        return [c.chunk_id for c in self._chunks]
