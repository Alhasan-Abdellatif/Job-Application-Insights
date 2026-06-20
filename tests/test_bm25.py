"""Tests for :mod:`job_application_insights.retrieval.bm25`."""

from __future__ import annotations

import pytest
from job_application_insights.ingest.chunk import Chunk
from job_application_insights.retrieval.bm25 import BM25Index, tokenize


def _chunk(chunk_id: str, text: str, **kw: object) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("__", 1)[0],
        chunk_index=0,
        text=text,
        n_tokens=len(text.split()),
        sender=kw.get("sender", "x@y.com"),
        subject=kw.get("subject", "subj"),
    )


# ───── tokenize ─────


def test_tokenize_lowercases():
    assert tokenize("Hello WORLD") == ["hello", "world"]


def test_tokenize_drops_punctuation():
    assert tokenize("hello, world!") == ["hello", "world"]


def test_tokenize_splits_on_whitespace_and_punctuation():
    assert tokenize("a/b c-d") == ["a", "b", "c", "d"]


def test_tokenize_keeps_alphanumerics():
    assert tokenize("gpt4 release notes 2026") == ["gpt4", "release", "notes", "2026"]


def test_tokenize_empty_string_yields_empty_list():
    assert tokenize("") == []


def test_tokenize_collapses_repeated_punctuation():
    assert tokenize("..a..b..") == ["a", "b"]


# ───── BM25Index: construction ─────


def test_bm25_empty_index_is_legal():
    idx = BM25Index([])
    assert idx.n_chunks == 0
    assert idx.query("anything", k=5) == []


def test_bm25_records_chunk_count():
    idx = BM25Index([_chunk("m1__c0", "hello"), _chunk("m2__c0", "world")])
    assert idx.n_chunks == 2


def test_bm25_chunk_ids_preserve_order():
    idx = BM25Index([_chunk("m1__c0", "x"), _chunk("m2__c0", "y"), _chunk("m3__c0", "z")])
    assert list(idx.chunk_ids()) == ["m1__c0", "m2__c0", "m3__c0"]


# ───── BM25Index: query semantics ─────


def test_bm25_finds_exact_token_match():
    chunks = [
        _chunk("m1__c0", "thank you for applying to anthropic"),
        _chunk("m2__c0", "thank you for applying to qualcomm"),
        _chunk("m3__c0", "thank you for applying to google"),
    ]
    idx = BM25Index(chunks)
    results = idx.query("qualcomm", k=3)
    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "m2__c0"


def test_bm25_orders_by_relevance():
    chunks = [
        _chunk("m1__c0", "qualcomm qualcomm qualcomm interview"),
        _chunk("m2__c0", "thank you for applying to qualcomm"),
        _chunk("m3__c0", "totally unrelated banking email"),
    ]
    idx = BM25Index(chunks)
    results = idx.query("qualcomm", k=3)
    # Higher TF document wins
    assert results[0].chunk.chunk_id == "m1__c0"
    # The unrelated email is filtered (zero score)
    assert all(r.chunk.chunk_id != "m3__c0" for r in results)


def test_bm25_filters_zero_score_results():
    chunks = [
        _chunk("m1__c0", "qualcomm interview"),
        _chunk("m2__c0", "completely unrelated"),
    ]
    idx = BM25Index(chunks)
    results = idx.query("qualcomm", k=5)
    # Only the match comes back; zero-score is dropped
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "m1__c0"


def test_bm25_respects_k_cutoff():
    chunks = [_chunk(f"m{i}__c0", "qualcomm signal") for i in range(10)]
    idx = BM25Index(chunks)
    results = idx.query("qualcomm", k=3)
    assert len(results) == 3


def test_bm25_query_is_case_insensitive():
    chunks = [_chunk("m1__c0", "Thank you for applying to QUALCOMM")]
    idx = BM25Index(chunks)
    results = idx.query("qualcomm", k=1)
    assert len(results) == 1


def test_bm25_no_match_returns_empty():
    chunks = [_chunk("m1__c0", "thank you for applying to anthropic")]
    idx = BM25Index(chunks)
    results = idx.query("qualcomm", k=5)
    assert results == []


def test_bm25_handles_empty_query():
    chunks = [_chunk("m1__c0", "anything goes here")]
    idx = BM25Index(chunks)
    assert idx.query("", k=5) == []
    assert idx.query("   ", k=5) == []
    assert idx.query("!!!", k=5) == []  # punctuation-only tokenises to nothing


def test_bm25_rare_token_outweighs_common_token():
    chunks = [
        _chunk("m1__c0", "thank you for applying " * 10),  # generic, common tokens
        _chunk("m2__c0", "your application to imperial college london"),
    ]
    idx = BM25Index(chunks)
    # "imperial" appears in only one doc → high IDF, wins despite shorter doc
    results = idx.query("imperial", k=2)
    assert results[0].chunk.chunk_id == "m2__c0"


# ───── BM25Index: validation ─────


@pytest.mark.parametrize("bad_k", [0, -1, -100])
def test_bm25_rejects_non_positive_k(bad_k: int):
    idx = BM25Index([_chunk("m1__c0", "x")])
    with pytest.raises(ValueError, match="positive"):
        idx.query("x", k=bad_k)


# ───── BM25Index: scores are float on RetrievalResult ─────


def test_bm25_result_scores_are_finite_positive():
    # Need a non-trivial corpus so IDF for the rare token is positive.
    # In a 1-doc corpus, every token has df=N=1 → negative IDF → BM25
    # falls back on the rank_bm25 epsilon (also negative). Real corpora
    # don't hit this; tests should reflect realistic sizes.
    chunks = [
        _chunk("m1__c0", "qualcomm interview"),
        _chunk("m2__c0", "thank you for applying to google"),
        _chunk("m3__c0", "your application to meta has been received"),
        _chunk("m4__c0", "thank you for applying to amazon"),
        _chunk("m5__c0", "application received apple"),
    ]
    idx = BM25Index(chunks)
    results = idx.query("qualcomm", k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "m1__c0"
    assert results[0].score > 0.0
    assert isinstance(results[0].score, float)
