"""Hybrid retrieval via Reciprocal Rank Fusion.

Yesterday's measurement showed dense and BM25 failing on *different*
questions: BM25 wins on proper-noun lookups (Imperial, Ellison), dense
wins on vocabulary-mismatch and high-frequency-token cases (Universities,
Amazon, Research Engineer). A hybrid retriever runs both and combines
their rankings so the *union* of strengths shows up in the top-K.

The combination algorithm is **Reciprocal Rank Fusion (RRF)** — Cormack,
Clarke & Büttcher, 2009. For each document, sum ``1 / (k + rank)`` over
every retriever that returned it:

.. math::

    RRF(d) = \\sum_{r \\in retrievers}  \\frac{1}{k + \\text{rank}_r(d)}

where ``k`` is a small constant (the paper recommends 60). Three
properties make this the right default:

* **Scale-invariant.** It compares *ranks*, not scores, so a cosine
  similarity in ``[-1, 1]`` and a BM25 score in ``[0, 30]`` can be
  fused without any calibration.
* **Rewards consensus.** A doc that both retrievers rank in the top-K
  gets credit from both sums and beats a doc only one retriever saw.
* **Parameter-free in practice.** One constant (``k=60``) that's been
  battle-tested for fifteen years and never seriously beaten on
  comparable budgets.

The function and the factory are intentionally generic: anything that
respects :data:`~job_application_insights.evals.runner.RetrievalFn`
plugs in. That means today it's dense + BM25; tomorrow we can add a
cross-encoder reranker as a third (or fourth) input with no change to
``reciprocal_rank_fusion`` or ``make_hybrid_retriever``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from job_application_insights.evals.runner import RetrievalFn

# ────────────────────────────── constants ──────────────────────────────


DEFAULT_RRF_K: int = 60
"""Cormack et al. 2009 constant. Damps the influence of top ranks just
enough that consensus across retrievers matters more than any one
retriever's first choice. Don't change without an eval to justify it."""

DEFAULT_FETCH_K: int = 50
"""How many candidates to pull from each individual retriever before
fusion. The motto is *cast a wide net cheap, then re-narrow*. 50 is
plenty for our corpus size; bump it if you start adding a reranker
that needs more candidates downstream."""


# ────────────────────────────── public API ──────────────────────────────


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of chunk IDs into one list with scores.

    Parameters
    ----------
    rankings
        One ranked list per retriever, **best-first**. Lists may have
        different lengths; missing entries simply don't contribute to
        a doc's RRF sum.
    rrf_k
        The RRF damping constant. Default 60 (the paper's recommendation).

    Returns
    -------
    A list of ``(chunk_id, rrf_score)`` pairs sorted descending by
    score. Each chunk ID appears exactly once even if it was returned
    by multiple retrievers.

    Pure function — no side effects, no I/O. Easy to unit-test against
    hand-calculated values.
    """
    if rrf_k < 0:
        raise ValueError(f"rrf_k must be non-negative, got {rrf_k}")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def make_hybrid_retriever(
    retrievers: Sequence[RetrievalFn],
    *,
    fetch_k: int = DEFAULT_FETCH_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> RetrievalFn:
    """Build a ``RetrievalFn`` that fuses several retrievers with RRF.

    Each retriever is asked for the top ``fetch_k`` candidates. Their
    rankings are fused with :func:`reciprocal_rank_fusion`. The
    returned function exposes the usual ``(query, k)`` signature.

    Parameters
    ----------
    retrievers
        Two or more retrievers conforming to
        :data:`~job_application_insights.evals.runner.RetrievalFn`.
        With one retriever the call still works, but you're just
        re-ranking by RRF over its own output (no gain).
    fetch_k
        How many candidates to pull from each retriever. Bigger ``fetch_k``
        = more chances for one retriever to surface a doc the other
        missed, at modest extra cost.
    rrf_k
        Forwarded to :func:`reciprocal_rank_fusion`.

    Raises
    ------
    ValueError
        If ``retrievers`` is empty, or ``fetch_k <= 0``.
    """
    retrievers_list = list(retrievers)
    if not retrievers_list:
        raise ValueError("at least one retriever is required")
    if fetch_k <= 0:
        raise ValueError(f"fetch_k must be positive, got {fetch_k}")

    def retrieve(query: str, k: int) -> list[str]:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        rankings = [r(query, fetch_k) for r in retrievers_list]
        fused = reciprocal_rank_fusion(rankings, rrf_k=rrf_k)
        return [chunk_id for chunk_id, _ in fused[:k]]

    return retrieve
