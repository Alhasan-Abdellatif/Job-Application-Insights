"""Retrieval metrics — pure functions over (retrieved, relevant, k).

Every function in this module has the same signature shape:

* ``retrieved`` — a ranked list of chunk IDs (best first) returned by the
  retriever you're evaluating.
* ``relevant`` — a *set* of chunk IDs that are known to be relevant for
  the query (from a hand-curated golden set).
* ``k`` — cutoff position. Only ``retrieved[:k]`` is scored.

Why this shape? Because it cleanly separates *what the math is* from
*where the data came from*. The runner module is responsible for talking
to the vector store and the golden set; this module only knows how to
score one list against one set. That makes the metrics trivial to test
and impossible to misuse.

Metrics implemented
-------------------
* :func:`precision_at_k` — fraction of the top-K that are relevant.
* :func:`recall_at_k` — fraction of relevant items that appear in top-K.
  The single most important metric for RAG: if relevant chunks aren't
  retrieved, the LLM literally cannot answer correctly.
* :func:`reciprocal_rank` — ``1 / rank`` of the first relevant hit. ``0``
  if nothing relevant was retrieved. Averaged over a dataset this is
  Mean Reciprocal Rank (MRR).
* :func:`dcg_at_k` / :func:`ndcg_at_k` — Discounted Cumulative Gain with
  binary relevance. nDCG normalises by the best-possible ranking so it
  lives in ``[0, 1]`` and is comparable across queries.

Edge cases
----------
* Empty ``relevant`` raises ``ValueError`` — an eval row with no
  ground-truth answer is a data bug, not something to silently score 0.
* Empty ``retrieved`` returns 0.0 for every metric.
* ``k`` larger than ``len(retrieved)`` is fine — we just score what we
  have.
* ``k <= 0`` raises ``ValueError``.

Binary vs graded relevance
--------------------------
We use binary relevance (a chunk is either relevant or it isn't). This
matches how the golden set is built: each (question, list-of-chunk-IDs)
pair is enumerated once, no grades. Graded relevance (e.g. "perfectly
relevant", "partially relevant") would let nDCG be more discriminative,
but it doubles the curation cost and is overkill for a single-person
eval set. If we ever want it, the signature change is small: swap
``relevant: set[str]`` for ``relevant: dict[str, float]``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# ────────────────────────────── public API ──────────────────────────────


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-K retrieved IDs that are in ``relevant``.

    ``precision@k = |retrieved[:k] ∩ relevant| / k``

    Useful when you care about not wasting the LLM's attention on
    irrelevant chunks. Less important than recall in most RAG settings
    because a missed relevant chunk hurts more than an extra noisy one.
    """
    _validate(relevant, k)
    if not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant IDs that appear in the top-K.

    ``recall@k = |retrieved[:k] ∩ relevant| / |relevant|``

    The headline metric for RAG retrieval. If recall@K is low, no
    prompt engineering can save the answer — the LLM is being asked
    to reason about evidence that isn't in front of it.
    """
    _validate(relevant, k)
    if not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """``1 / rank`` of the first relevant hit (1-indexed), else ``0``.

    Averaged over a dataset this is **Mean Reciprocal Rank (MRR)** —
    a measure of how high up the list the *first* useful result lands.
    Loved by chatbots and search engines because the top result
    disproportionately matters.

    There's no ``k`` argument: MRR is computed over the entire ranking.
    (You could add a cutoff — e.g. ``mrr@10`` — and it's a one-line
    change, but for a few-thousand-chunk store it adds no information.)
    """
    if not relevant:
        raise ValueError("`relevant` must be non-empty; empty golden-set entries are data bugs")
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Discounted Cumulative Gain at K, with binary relevance.

    ``DCG@k = Σ_{i=1..k}  rel_i / log2(i + 1)``

    where ``rel_i = 1`` if the i-th retrieved ID is in ``relevant``
    and ``0`` otherwise. The logarithm "discounts" gains the further
    down the list they appear: a relevant hit at rank 1 is worth
    ``1 / log2(2) = 1``, at rank 2 worth ``1 / log2(3) ≈ 0.6309``,
    at rank 10 worth only ``≈ 0.2890``.

    DCG is unbounded above and depends on how many relevant items
    exist — use :func:`ndcg_at_k` for a normalised, comparable
    version.
    """
    _validate(relevant, k)
    if not retrieved:
        return 0.0
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved[:k], start=1)
        if cid in relevant
    )


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K. Lives in ``[0, 1]``.

    ``nDCG@k = DCG@k / IDCG@k``

    where ``IDCG@k`` is the DCG of the *ideal* ranking — all relevant
    items packed at the top of the list. With binary relevance and
    ``R = |relevant|`` items, that's::

        IDCG@k = Σ_{i=1..min(R, k)}  1 / log2(i + 1)

    nDCG = 1.0 means the retriever ranked every relevant item exactly
    at the top. nDCG = 0.0 means it found nothing relevant in the
    top-K. Because of normalisation it's safe to average across
    queries with different numbers of relevant items.
    """
    _validate(relevant, k)
    if not retrieved:
        return 0.0
    actual = dcg_at_k(retrieved, relevant, k)
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return actual / ideal  # ideal > 0 because len(relevant) >= 1 (validated)


# ────────────────────────────── internals ──────────────────────────────


def _validate(relevant: set[str], k: int) -> None:
    if not relevant:
        raise ValueError("`relevant` must be non-empty; empty golden-set entries are data bugs")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
