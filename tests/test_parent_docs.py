"""Tests for parent-document retrieval (retrieval/parent_docs.py)."""

from __future__ import annotations

from job_application_insights.ingest.chunk import Chunk
from job_application_insights.retrieval.parent_docs import expand_to_parent_documents
from job_application_insights.retrieval.vector_store import RetrievalResult


def _chunk(chunk_id: str, doc_id: str, chunk_index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_index=chunk_index,
        text=text,
        n_tokens=max(1, len(text.split())),
    )


def _r(chunk: Chunk, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(chunk=chunk, score=score)


# Three emails:
#   msg_001 has 3 chunks (c0/c1/c2)
#   msg_002 has 2 chunks (c0/c1)
#   msg_003 has 1 chunk  (c0)
M1C0 = _chunk("msg_001__c000", "msg_001", 0, "Hello.")
M1C1 = _chunk("msg_001__c001", "msg_001", 1, "Middle part.")
M1C2 = _chunk("msg_001__c002", "msg_001", 2, "Goodbye.")
M2C0 = _chunk("msg_002__c000", "msg_002", 0, "Second email start.")
M2C1 = _chunk("msg_002__c001", "msg_002", 1, "Second email end.")
M3C0 = _chunk("msg_003__c000", "msg_003", 0, "Third single-chunk email.")

CHUNKS_BY_ID: dict[str, Chunk] = {c.chunk_id: c for c in [M1C0, M1C1, M1C2, M2C0, M2C1, M3C0]}


# ────────────────────────── basic expansion ──────────────────────────


def test_single_chunk_expands_to_full_parent() -> None:
    """One chunk in → one parent result with all sibling chunks concatenated."""
    out = expand_to_parent_documents([_r(M1C1)], CHUNKS_BY_ID)
    assert len(out) == 1
    assert out[0].chunk.chunk_id == "msg_001"  # citation = email id
    assert out[0].chunk.doc_id == "msg_001"
    assert out[0].chunk.text == "Hello.\n\nMiddle part.\n\nGoodbye."


def test_chunks_are_concatenated_in_chunk_index_order() -> None:
    """Order in the input doesn't matter; output uses chunk_index order."""
    # Retrieved c2 before c0 — but parent text must be in c0..c2 order.
    out = expand_to_parent_documents([_r(M1C2), _r(M1C0)], CHUNKS_BY_ID)
    assert len(out) == 1  # deduplicated to one parent
    assert out[0].chunk.text == "Hello.\n\nMiddle part.\n\nGoodbye."


def test_n_tokens_is_sum_of_siblings() -> None:
    """Parent's n_tokens reflects the LLM-budget reality of seeing the whole email."""
    out = expand_to_parent_documents([_r(M1C0)], CHUNKS_BY_ID)
    assert out[0].chunk.n_tokens == M1C0.n_tokens + M1C1.n_tokens + M1C2.n_tokens


def test_score_kept_from_first_hit_in_group() -> None:
    """The parent inherits the *retrieval rank* of the first hit for that doc."""
    out = expand_to_parent_documents(
        [_r(M1C2, score=0.4), _r(M1C0, score=0.9)],
        CHUNKS_BY_ID,
    )
    # First retrieved chunk for msg_001 was c2 (score 0.4), so the parent gets 0.4.
    assert out[0].score == 0.4


# ────────────────────────── deduplication ──────────────────────────


def test_multiple_chunks_same_doc_collapse_to_one_parent() -> None:
    """Three chunks from msg_001 in → one parent out."""
    out = expand_to_parent_documents([_r(M1C0), _r(M1C1), _r(M1C2)], CHUNKS_BY_ID)
    assert len(out) == 1
    assert out[0].chunk.chunk_id == "msg_001"


def test_multiple_docs_preserve_first_hit_order() -> None:
    """Order of parents follows first appearance in the input."""
    out = expand_to_parent_documents(
        [_r(M2C0), _r(M1C0), _r(M3C0)],
        CHUNKS_BY_ID,
    )
    assert [r.chunk.chunk_id for r in out] == ["msg_002", "msg_001", "msg_003"]


# ────────────────────────── edge cases ──────────────────────────


def test_empty_input_returns_empty() -> None:
    assert expand_to_parent_documents([], CHUNKS_BY_ID) == []


def test_single_chunk_email_passes_through() -> None:
    """If a doc has just one chunk, its parent text equals that chunk's text."""
    out = expand_to_parent_documents([_r(M3C0)], CHUNKS_BY_ID)
    assert len(out) == 1
    assert out[0].chunk.text == "Third single-chunk email."


def test_unknown_doc_id_uses_input_chunk_as_fallback() -> None:
    """If the corpus map doesn't contain the doc, fall back to just the input chunk.

    Defensive — this shouldn't happen if ``chunks_by_id`` is the full corpus,
    but the contract is that we don't crash. The fallback uses the input's
    own text as the "parent" so the LLM at least sees what the retriever found.
    """
    orphan = _chunk("msg_orphan__c000", "msg_orphan", 0, "Orphaned chunk.")
    out = expand_to_parent_documents([_r(orphan)], CHUNKS_BY_ID)
    assert len(out) == 1
    assert out[0].chunk.chunk_id == "msg_orphan"
    assert out[0].chunk.text == "Orphaned chunk."


def test_custom_separator() -> None:
    out = expand_to_parent_documents([_r(M1C0)], CHUNKS_BY_ID, separator=" | ")
    assert out[0].chunk.text == "Hello. | Middle part. | Goodbye."
