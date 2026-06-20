"""Tests for :mod:`job_application_insights.evals.runner`."""

from __future__ import annotations

from pathlib import Path

import pytest
from job_application_insights.evals.golden_set import GoldenEntry, save_golden_set
from job_application_insights.evals.runner import (
    EvalReport,
    EvalRow,
    evaluate,
    evaluate_path,
    format_report,
)


def _golden(question: str, relevant: list[str], tags: list[str] | None = None) -> GoldenEntry:
    return GoldenEntry(question=question, relevant_chunk_ids=relevant, tags=tags or [])


# ───── evaluate(): happy path ─────


def test_evaluate_perfect_retriever_gets_all_ones():
    """Retriever returns relevant chunks at the very top → all metrics = 1.0."""
    golden = [
        _golden("q1", ["a", "b"]),
        _golden("q2", ["x"]),
    ]

    # Perfect: returns the ground-truth IDs in order, top-of-list
    def retrieve(query: str, k: int) -> list[str]:
        relevant_map = {"q1": ["a", "b"], "q2": ["x"]}
        return relevant_map[query][:k]

    report = evaluate(golden, retrieve, k=5)
    assert report.mean_recall == pytest.approx(1.0)
    assert report.mean_mrr == pytest.approx(1.0)
    assert report.mean_ndcg == pytest.approx(1.0)
    # Precision: q1 = 2/5, q2 = 1/5 → mean = 0.3
    assert report.mean_precision == pytest.approx(0.3)


def test_evaluate_useless_retriever_gets_all_zeros():
    """Retriever returns nothing relevant → all metrics = 0.0."""
    golden = [_golden("q1", ["a"])]
    report = evaluate(golden, lambda q, k: ["x", "y", "z"], k=3)
    assert report.mean_recall == 0.0
    assert report.mean_mrr == 0.0
    assert report.mean_ndcg == 0.0
    assert report.mean_precision == 0.0


def test_evaluate_partial_recall():
    """One of two relevant chunks found at rank 2."""
    golden = [_golden("q1", ["a", "b"])]
    # Retriever returns ['x', 'a', 'y']
    report = evaluate(golden, lambda q, k: ["x", "a", "y"][:k], k=3)
    # Recall@3 = 1/2 = 0.5
    assert report.mean_recall == pytest.approx(0.5)
    # MRR: first relevant ('a') at rank 2 → 0.5
    assert report.mean_mrr == pytest.approx(0.5)


def test_evaluate_returns_correct_per_question_rows():
    golden = [_golden("q1", ["a"], tags=["t1"]), _golden("q2", ["b"], tags=["t2"])]
    report = evaluate(golden, lambda q, k: [q.replace("q", "")[0]], k=3)
    # NB: above lambda maps "q1" → ["1"] and "q2" → ["2"] — both miss
    assert len(report.rows) == 2
    assert report.rows[0].question == "q1"
    assert report.rows[0].tags == ["t1"]
    assert report.rows[1].question == "q2"
    assert report.rows[1].tags == ["t2"]


# ───── evaluate(): validation ─────


def test_evaluate_rejects_empty_golden_set():
    with pytest.raises(ValueError, match="empty"):
        evaluate([], lambda q, k: [], k=5)


@pytest.mark.parametrize("bad_k", [0, -1, -100])
def test_evaluate_rejects_non_positive_k(bad_k):
    golden = [_golden("q1", ["a"])]
    with pytest.raises(ValueError, match="positive"):
        evaluate(golden, lambda q, k: [], k=bad_k)


# ───── EvalRow / EvalReport invariants ─────


def test_eval_row_is_frozen():
    row = EvalRow(
        question="q",
        n_relevant=1,
        n_retrieved=1,
        precision_at_k=1.0,
        recall_at_k=1.0,
        reciprocal_rank=1.0,
        ndcg_at_k=1.0,
    )
    with pytest.raises(Exception):  # noqa: B017
        row.recall_at_k = 0.0  # type: ignore[misc]


def test_eval_report_holds_rows():
    golden = [_golden("q1", ["a"])]
    report = evaluate(golden, lambda q, k: ["a"], k=1)
    assert isinstance(report, EvalReport)
    assert len(report.rows) == 1
    assert isinstance(report.rows[0], EvalRow)


# ───── evaluate_path: file-based variant ─────


def test_evaluate_path_round_trips_through_disk(tmp_path: Path):
    golden = [_golden("q1", ["a"]), _golden("q2", ["b"])]
    path = tmp_path / "g.jsonl"
    save_golden_set(golden, path)
    report = evaluate_path(path, lambda q, k: ["a", "b"][:k], k=2)
    # Both questions: relevant in top-2 → recall = 1.0 each → mean = 1.0
    # Wait: q1 expects "a", retriever returns ["a","b"], recall = 1/1.
    #       q2 expects "b", retriever returns ["a","b"], recall = 1/1.
    assert report.mean_recall == pytest.approx(1.0)


# ───── format_report ─────


def test_format_report_includes_aggregates():
    golden = [_golden("q1", ["a"])]
    report = evaluate(golden, lambda q, k: ["a"], k=1)
    text = format_report(report)
    assert "MEAN" in text
    assert "R=1.000" in text
    assert "MRR=1.000" in text


def test_format_report_can_hide_rows():
    golden = [_golden("q1", ["a"])]
    report = evaluate(golden, lambda q, k: ["a"], k=1)
    text = format_report(report, show_rows=False)
    assert "MEAN" in text
    # The question text shouldn't appear when rows are hidden
    assert "q1" not in text
