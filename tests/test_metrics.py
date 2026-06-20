"""Tests for :mod:`job_application_insights.evals.metrics`."""

from __future__ import annotations

import math

import pytest
from job_application_insights.evals.metrics import (
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ───── precision_at_k ─────


def test_precision_at_k_perfect():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert precision_at_k(retrieved, relevant, 3) == 1.0


def test_precision_at_k_half():
    retrieved = ["a", "x", "b", "y"]
    relevant = {"a", "b"}
    # top-4: a, x, b, y → 2 hits / 4 = 0.5
    assert precision_at_k(retrieved, relevant, 4) == 0.5


def test_precision_at_k_truncates_to_k():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "b"}
    # top-2: a, b → 2/2 = 1.0
    assert precision_at_k(retrieved, relevant, 2) == 1.0


def test_precision_at_k_no_hits():
    retrieved = ["x", "y", "z"]
    relevant = {"a"}
    assert precision_at_k(retrieved, relevant, 3) == 0.0


def test_precision_at_k_empty_retrieved():
    assert precision_at_k([], {"a"}, 5) == 0.0


def test_precision_at_k_k_larger_than_list():
    retrieved = ["a", "b"]
    relevant = {"a"}
    # top-10 is just ["a", "b"] → 1 hit / 10 = 0.1
    assert precision_at_k(retrieved, relevant, 10) == 0.1


# ───── recall_at_k ─────


def test_recall_at_k_perfect():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_recall_at_k_half():
    retrieved = ["a", "x", "y", "z"]
    relevant = {"a", "b"}
    # top-4 includes "a" but not "b" → 1/2 = 0.5
    assert recall_at_k(retrieved, relevant, 4) == 0.5


def test_recall_at_k_truncated_misses_relevant():
    retrieved = ["x", "y", "z", "a"]
    relevant = {"a"}
    # top-2: x, y → 0 hits
    assert recall_at_k(retrieved, relevant, 2) == 0.0
    # top-4: x, y, z, a → 1/1
    assert recall_at_k(retrieved, relevant, 4) == 1.0


def test_recall_at_k_empty_retrieved():
    assert recall_at_k([], {"a"}, 5) == 0.0


# ───── reciprocal_rank ─────


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5


def test_reciprocal_rank_finds_first_relevant_only():
    # Even though both "a" and "b" are relevant, we only count the
    # rank of the first relevant hit.
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5


def test_reciprocal_rank_no_relevant_in_list():
    assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0


def test_reciprocal_rank_empty_retrieved():
    assert reciprocal_rank([], {"a"}) == 0.0


# ───── dcg_at_k ─────


def test_dcg_at_k_rank_one_is_worth_one():
    # Single relevant hit at rank 1 → 1/log2(2) = 1.0
    assert dcg_at_k(["a"], {"a"}, 1) == 1.0


def test_dcg_at_k_rank_two_uses_log2_3():
    # Single relevant hit at rank 2 → 1/log2(3)
    expected = 1.0 / math.log2(3)
    assert dcg_at_k(["x", "a"], {"a"}, 5) == pytest.approx(expected)


def test_dcg_at_k_sums_over_hits():
    # Hits at ranks 1 and 3 → 1/log2(2) + 1/log2(4) = 1.0 + 0.5
    result = dcg_at_k(["a", "x", "b"], {"a", "b"}, 3)
    assert result == pytest.approx(1.0 + 0.5)


def test_dcg_at_k_truncates_at_k():
    # Hit at rank 5 is ignored when k=3
    assert dcg_at_k(["x", "y", "z", "w", "a"], {"a"}, 3) == 0.0


# ───── ndcg_at_k ─────


def test_ndcg_at_k_perfect_ranking_is_one():
    # All relevant items packed at the top
    retrieved = ["a", "b", "c", "x", "y"]
    relevant = {"a", "b", "c"}
    assert ndcg_at_k(retrieved, relevant, 5) == pytest.approx(1.0)


def test_ndcg_at_k_no_relevant_in_top_k_is_zero():
    retrieved = ["x", "y", "z"]
    relevant = {"a"}
    assert ndcg_at_k(retrieved, relevant, 3) == 0.0


def test_ndcg_at_k_lies_between_zero_and_one():
    # Hits at ranks 2 and 4 — non-trivial, not perfect, not zero
    retrieved = ["x", "a", "y", "b", "z"]
    relevant = {"a", "b"}
    score = ndcg_at_k(retrieved, relevant, 5)
    assert 0.0 < score < 1.0


def test_ndcg_at_k_handles_more_relevant_than_k():
    # 5 relevant items, k=3 — ideal is 3 perfect hits in top-3
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "b", "c", "d", "e"}
    # All 5 relevant — top-3 are a, b, c → perfect for k=3
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)


def test_ndcg_at_k_partial_credit():
    # Single relevant item at rank 2, k=3
    # DCG = 1/log2(3), IDCG = 1/log2(2) = 1.0
    # nDCG = 1/log2(3) ≈ 0.6309
    retrieved = ["x", "a", "y"]
    relevant = {"a"}
    expected = (1.0 / math.log2(3)) / 1.0
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(expected)


# ───── shared validation ─────


@pytest.mark.parametrize(
    "fn",
    [
        lambda r, s, k: precision_at_k(r, s, k),
        lambda r, s, k: recall_at_k(r, s, k),
        lambda r, s, k: dcg_at_k(r, s, k),
        lambda r, s, k: ndcg_at_k(r, s, k),
    ],
)
def test_empty_relevant_raises(fn):
    with pytest.raises(ValueError, match="non-empty"):
        fn(["a"], set(), 5)


def test_reciprocal_rank_empty_relevant_raises():
    with pytest.raises(ValueError, match="non-empty"):
        reciprocal_rank(["a"], set())


@pytest.mark.parametrize(
    "fn",
    [
        lambda r, s, k: precision_at_k(r, s, k),
        lambda r, s, k: recall_at_k(r, s, k),
        lambda r, s, k: dcg_at_k(r, s, k),
        lambda r, s, k: ndcg_at_k(r, s, k),
    ],
)
@pytest.mark.parametrize("bad_k", [0, -1, -100])
def test_non_positive_k_raises(fn, bad_k):
    with pytest.raises(ValueError, match="positive"):
        fn(["a"], {"a"}, bad_k)
