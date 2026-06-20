"""Tests for :mod:`job_application_insights.retrieval.hybrid`."""

from __future__ import annotations

import pytest
from job_application_insights.evals.runner import RetrievalFn
from job_application_insights.retrieval.hybrid import (
    DEFAULT_RRF_K,
    make_hybrid_retriever,
    reciprocal_rank_fusion,
)

# ───── reciprocal_rank_fusion: math ─────


def test_rrf_single_ranking_returns_descending_scores():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    # Scores: a=1/(60+1), b=1/(60+2), c=1/(60+3) → already sorted descending
    assert [cid for cid, _ in fused] == ["a", "b", "c"]
    assert fused[0][1] == pytest.approx(1.0 / 61)
    assert fused[1][1] == pytest.approx(1.0 / 62)
    assert fused[2][1] == pytest.approx(1.0 / 63)


def test_rrf_full_overlap_sums_scores():
    """A doc ranked #1 by both retrievers gets 2 * 1/61."""
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "b"]])
    score_a, score_b = dict(fused)["a"], dict(fused)["b"]
    assert score_a == pytest.approx(2.0 / 61)
    assert score_b == pytest.approx(2.0 / 62)


def test_rrf_partial_overlap_rewards_consensus():
    """A doc seen by both retrievers beats one only one retriever saw,
    even when the latter is at rank 1 alone."""
    ranking_dense = ["x", "a", "b"]
    ranking_bm25 = ["a", "y", "b"]
    fused = dict(reciprocal_rank_fusion([ranking_dense, ranking_bm25]))
    # 'a' is at rank 2 in dense and rank 1 in bm25 → 1/62 + 1/61
    # 'x' is at rank 1 in dense only → 1/61
    # 'a' beats 'x' because consensus matters
    assert fused["a"] > fused["x"]


def test_rrf_doc_in_one_ranking_only_still_scored():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["c"]]))
    assert "a" in fused
    assert "b" in fused
    assert "c" in fused
    # Each rank-1 doc gets 1/61
    assert fused["a"] == pytest.approx(1.0 / 61)
    assert fused["c"] == pytest.approx(1.0 / 61)


def test_rrf_dedupes_within_a_single_ranking():
    """If a retriever produces a duplicate (shouldn't happen, but…) the
    last occurrence wins via dict semantics. Caller-visible behaviour:
    the doc appears once in the output."""
    fused = reciprocal_rank_fusion([["a", "b", "a"]])
    chunk_ids = [cid for cid, _ in fused]
    assert chunk_ids.count("a") == 1


def test_rrf_empty_rankings_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_constant_default_is_60():
    """Cormack 2009 recommendation. Don't change without a citation."""
    assert DEFAULT_RRF_K == 60


def test_rrf_constant_affects_score_decay():
    """Bigger rrf_k = flatter score curve = consensus matters more.
    Smaller rrf_k = top ranks dominate more."""
    flat = dict(reciprocal_rank_fusion([["a", "b", "c"]], rrf_k=100))
    steep = dict(reciprocal_rank_fusion([["a", "b", "c"]], rrf_k=1))
    # Ratio of score[a]/score[c] is larger when rrf_k is smaller.
    assert (steep["a"] / steep["c"]) > (flat["a"] / flat["c"])


def test_rrf_rejects_negative_constant():
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion([["a"]], rrf_k=-1)


# ───── make_hybrid_retriever: behaviour ─────


def _const_retriever(ranking: list[str]) -> RetrievalFn:
    """Return a fake retriever that always returns ``ranking[:k]``."""

    def retrieve(query: str, k: int) -> list[str]:
        return ranking[:k]

    return retrieve


def test_hybrid_returns_top_k_chunks():
    dense = _const_retriever(["a", "b", "c", "d", "e"])
    bm25 = _const_retriever(["c", "f", "g", "h", "i"])
    hybrid = make_hybrid_retriever([dense, bm25], fetch_k=5)
    result = hybrid("any query", k=3)
    assert len(result) == 3
    # 'c' appears in both → should be near the top
    assert "c" in result


def test_hybrid_promotes_consensus_doc_to_top():
    """A doc both retrievers rank at #1 should be #1 in the fused output."""
    dense = _const_retriever(["consensus", "x", "y"])
    bm25 = _const_retriever(["consensus", "p", "q"])
    hybrid = make_hybrid_retriever([dense, bm25], fetch_k=3)
    result = hybrid("q", k=3)
    assert result[0] == "consensus"


def test_hybrid_with_one_retriever_works_but_is_pointless():
    """A single retriever wrapped in hybrid just re-ranks itself by RRF."""
    only = _const_retriever(["a", "b", "c"])
    hybrid = make_hybrid_retriever([only], fetch_k=3)
    assert hybrid("q", k=3) == ["a", "b", "c"]


def test_hybrid_rejects_empty_retrievers():
    with pytest.raises(ValueError, match="at least one"):
        make_hybrid_retriever([])


def test_hybrid_rejects_non_positive_fetch_k():
    only = _const_retriever(["a"])
    with pytest.raises(ValueError, match="fetch_k"):
        make_hybrid_retriever([only], fetch_k=0)


def test_hybrid_rejects_non_positive_query_k():
    only = _const_retriever(["a", "b"])
    hybrid = make_hybrid_retriever([only])
    with pytest.raises(ValueError, match="positive"):
        hybrid("q", k=0)


def test_hybrid_fetch_k_caps_input_to_each_retriever():
    """fetch_k limits how many chunks each retriever sees, regardless of k."""
    calls: list[int] = []

    def spy(query: str, k: int) -> list[str]:
        calls.append(k)
        return ["a", "b", "c"]

    hybrid = make_hybrid_retriever([spy, spy], fetch_k=2)
    hybrid("q", k=1)
    assert calls == [2, 2]


def test_hybrid_returns_unique_chunk_ids():
    dense = _const_retriever(["a", "b", "c"])
    bm25 = _const_retriever(["a", "b", "d"])  # 'a' and 'b' overlap
    hybrid = make_hybrid_retriever([dense, bm25], fetch_k=3)
    result = hybrid("q", k=4)
    assert len(result) == len(set(result))
