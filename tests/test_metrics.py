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


# ───── group-aware metrics ─────


from job_application_insights.evals.metrics import (  # noqa: E402
    dcg_at_k_groups,
    ndcg_at_k_groups,
    precision_at_k_groups,
    recall_at_k_groups,
    reciprocal_rank_groups,
)


def test_recall_groups_one_hit_per_group_is_perfect():
    """Each of 3 groups has a chunk in the top-3 → recall = 1.0."""
    groups = [{"a", "b"}, {"c"}, {"d", "e"}]
    retrieved = ["a", "c", "d"]
    assert recall_at_k_groups(retrieved, groups, 3) == pytest.approx(1.0)


def test_recall_groups_finding_two_chunks_from_same_group_counts_once():
    """The key property: extra chunks of an already-hit group give no credit."""
    groups = [{"a", "b", "c"}, {"d"}]
    # Retrieved 'a' and 'b' (same group) but missed group 2
    retrieved = ["a", "b", "x"]
    # Only 1 of 2 groups hit → recall = 0.5 (not 2/3 like the flat metric)
    assert recall_at_k_groups(retrieved, groups, 3) == pytest.approx(0.5)


def test_recall_groups_reduces_to_chunk_recall_with_singleton_groups():
    """Backwards-compat: groups of size 1 should behave like flat recall."""
    flat_relevant = {"a", "b", "c"}
    groups = [{"a"}, {"b"}, {"c"}]
    retrieved = ["a", "x", "b"]
    assert recall_at_k_groups(retrieved, groups, 3) == recall_at_k(retrieved, flat_relevant, 3)


def test_recall_groups_no_hits_is_zero():
    assert recall_at_k_groups(["x"], [{"a"}, {"b"}], 3) == 0.0


def test_recall_groups_caps_at_one_even_with_many_relevant_chunks():
    """Even if every retrieved chunk is relevant, recall can't exceed 1.0."""
    # One big group with 10 chunks. Top-3 are all in it.
    groups = [set("abcdefghij")]
    retrieved = ["a", "b", "c"]
    assert recall_at_k_groups(retrieved, groups, 3) == pytest.approx(1.0)


# ───── precision_at_k_groups ─────


def test_precision_groups_counts_any_chunk_in_any_group():
    groups = [{"a", "b"}, {"c"}]
    retrieved = ["a", "x", "c", "y"]
    # Top-4: a (hit), x (miss), c (hit), y (miss) → 2/4 = 0.5
    assert precision_at_k_groups(retrieved, groups, 4) == pytest.approx(0.5)


# ───── reciprocal_rank_groups ─────


def test_mrr_groups_finds_first_chunk_in_any_group():
    groups = [{"a"}, {"b", "c"}]
    retrieved = ["x", "b", "a"]
    # 'b' is in group 2 at rank 2 → 1/2 = 0.5
    assert reciprocal_rank_groups(retrieved, groups) == pytest.approx(0.5)


def test_mrr_groups_no_relevant_returns_zero():
    assert reciprocal_rank_groups(["x", "y"], [{"a"}, {"b"}]) == 0.0


# ───── ndcg_at_k_groups ─────


def test_ndcg_groups_perfect_ranking_is_one():
    groups = [{"a"}, {"b"}, {"c"}]
    retrieved = ["a", "b", "c", "x", "y"]
    assert ndcg_at_k_groups(retrieved, groups, 5) == pytest.approx(1.0)


def test_ndcg_groups_subsequent_chunks_of_same_group_dont_help():
    """Set-cover DCG: only the FIRST hit of each group accrues gain."""
    # Two groups. Group1 = {a, b}, Group2 = {c}. Retrieved hits group1
    # twice and never finds group2.
    groups = [{"a", "b"}, {"c"}]
    retrieved = ["a", "b", "x"]  # rank 1 hits group1; rank 2 is "wasted"
    # DCG = 1/log2(2) = 1.0. IDCG@3 = 1/log2(2) + 1/log2(3) ≈ 1.6309
    expected = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k_groups(retrieved, groups, 3) == pytest.approx(expected)


def test_ndcg_groups_zero_for_no_hits():
    assert ndcg_at_k_groups(["x"], [{"a"}], 1) == 0.0


# ───── validation parity with chunk-level ─────


@pytest.mark.parametrize(
    "fn",
    [
        precision_at_k_groups,
        recall_at_k_groups,
        dcg_at_k_groups,
        ndcg_at_k_groups,
    ],
)
def test_groups_metrics_reject_empty_groups(fn):
    with pytest.raises(ValueError, match="non-empty|empty"):
        fn(["a"], [], 5)


@pytest.mark.parametrize(
    "fn",
    [
        precision_at_k_groups,
        recall_at_k_groups,
        dcg_at_k_groups,
        ndcg_at_k_groups,
    ],
)
def test_groups_metrics_reject_empty_inner_group(fn):
    with pytest.raises(ValueError, match="≥1 chunk"):
        fn(["a"], [{"a"}, set()], 3)
