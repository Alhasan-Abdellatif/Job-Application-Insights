"""Tests for :mod:`job_application_insights.retrieval.rerank`.

These tests use a deterministic stub reranker so the 280 MB cross-encoder
weights are not required at test time. The Protocol design lets the stub
swap in seamlessly.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from job_application_insights.evals.runner import RetrievalFn
from job_application_insights.retrieval.rerank import (
    DEFAULT_FETCH_K,
    DEFAULT_RERANKER_MODEL,
    Reranker,
    make_reranked_retriever,
)


class StubReranker:
    """Reranker that scores each chunk by a hand-mapped (query, id) lookup.

    Lets tests assert "the reranker rescued this chunk" without loading
    a model. Conforms to the :class:`Reranker` Protocol by virtue of the
    method signature alone — no inheritance needed.
    """

    def __init__(self, scores_by_id: dict[str, float]) -> None:
        self._scores_by_id = scores_by_id

    def rerank(
        self,
        query: str,
        items: Sequence[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        del query  # the stub doesn't read the query — scores are id-keyed
        ranked = [(cid, self._scores_by_id.get(cid, 0.0)) for cid, _ in items]
        ranked.sort(key=lambda kv: kv[1], reverse=True)
        return ranked


def _base_retriever(ranking: list[str]) -> RetrievalFn:
    def retrieve(query: str, k: int) -> list[str]:
        return ranking[:k]

    return retrieve


def _lookup(texts_by_id: dict[str, str]):
    return lambda cid: texts_by_id[cid]


# ───── make_reranked_retriever: behaviour ─────


def test_reranker_can_promote_a_buried_candidate():
    """The base puts the right doc at rank 3. The reranker should promote it."""
    base = _base_retriever(["wrong1", "wrong2", "right", "wrong3"])
    texts = _lookup({"wrong1": "x", "wrong2": "y", "right": "z", "wrong3": "w"})
    stub = StubReranker({"right": 0.99, "wrong1": 0.10, "wrong2": 0.05, "wrong3": 0.01})
    retriever = make_reranked_retriever(base, stub, texts, fetch_k=4)
    assert retriever("any query", k=1) == ["right"]


def test_reranker_can_demote_a_top_candidate():
    """Base puts an irrelevant doc at #1. Reranker demotes it."""
    base = _base_retriever(["noise", "real", "other"])
    texts = _lookup({"noise": "irrelevant text", "real": "answer text", "other": "z"})
    stub = StubReranker({"noise": 0.01, "real": 0.95, "other": 0.50})
    retriever = make_reranked_retriever(base, stub, texts, fetch_k=3)
    out = retriever("q", k=3)
    assert out[0] == "real"
    assert out[1] == "other"
    assert out[2] == "noise"


def test_reranker_returns_top_k_chunks():
    base = _base_retriever([f"d{i}" for i in range(10)])
    texts = _lookup({f"d{i}": f"text {i}" for i in range(10)})
    stub = StubReranker({f"d{i}": float(10 - i) for i in range(10)})
    retriever = make_reranked_retriever(base, stub, texts, fetch_k=10)
    out = retriever("q", k=3)
    assert out == ["d0", "d1", "d2"]


def test_reranker_uses_fetch_k_for_base_call():
    """The base retriever should be called with fetch_k, not the user k."""
    seen_k: list[int] = []

    def spy(query: str, k: int) -> list[str]:
        seen_k.append(k)
        return ["a", "b", "c"]

    texts = _lookup({"a": "1", "b": "2", "c": "3"})
    stub = StubReranker({"a": 1.0, "b": 2.0, "c": 3.0})
    retriever = make_reranked_retriever(spy, stub, texts, fetch_k=42)
    retriever("q", k=1)
    assert seen_k == [42]


def test_reranker_handles_empty_base_output():
    base = _base_retriever([])
    texts = _lookup({})
    stub = StubReranker({})
    retriever = make_reranked_retriever(base, stub, texts, fetch_k=10)
    assert retriever("q", k=5) == []


def test_reranker_propagates_missing_chunk_text_error():
    """If the base returns an ID the lookup doesn't know, raise loudly.
    Silently dropping would hide corpus-drift bugs."""
    base = _base_retriever(["known", "unknown"])
    texts = _lookup({"known": "text"})  # 'unknown' is missing
    stub = StubReranker({"known": 1.0})
    retriever = make_reranked_retriever(base, stub, texts, fetch_k=2)
    with pytest.raises(KeyError):
        retriever("q", k=1)


# ───── factory validation ─────


@pytest.mark.parametrize("bad_fetch_k", [0, -1, -100])
def test_make_reranked_rejects_non_positive_fetch_k(bad_fetch_k: int):
    base = _base_retriever(["a"])
    stub = StubReranker({"a": 1.0})
    with pytest.raises(ValueError, match="fetch_k"):
        make_reranked_retriever(base, stub, lambda x: "t", fetch_k=bad_fetch_k)


@pytest.mark.parametrize("bad_k", [0, -1])
def test_reranked_retriever_rejects_non_positive_query_k(bad_k: int):
    base = _base_retriever(["a", "b"])
    texts = _lookup({"a": "1", "b": "2"})
    stub = StubReranker({"a": 1.0, "b": 0.5})
    retriever = make_reranked_retriever(base, stub, texts, fetch_k=2)
    with pytest.raises(ValueError, match="positive"):
        retriever("q", k=bad_k)


# ───── constants exist ─────


def test_default_reranker_model_is_bge():
    assert "bge" in DEFAULT_RERANKER_MODEL.lower()


def test_default_fetch_k_is_fifty():
    assert DEFAULT_FETCH_K == 50


# ───── StubReranker satisfies Reranker Protocol ─────


def test_stub_reranker_satisfies_protocol():
    """Static-typing sanity: a structurally-typed stub works as a Reranker."""
    r: Reranker = StubReranker({"x": 1.0})
    result = r.rerank("q", [("x", "text")])
    assert result == [("x", 1.0)]
