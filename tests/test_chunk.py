"""Tests for :mod:`job_application_insights.ingest.chunk`."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise

import pytest
from job_application_insights.ingest.chunk import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Chunk,
    chunk_documents,
    count_tokens,
)
from job_application_insights.ingest.parse import Document
from pydantic import ValidationError

# ───── helpers ─────


def _make_doc(
    doc_id: str = "msg_000001",
    subject: str = "Hi",
    body: str = "Hello world.",
    sender: str = "x@y.com",
    date: datetime | None = None,
) -> Document:
    return Document(
        doc_id=doc_id,
        sender=sender,
        recipient="",
        subject=subject,
        body=body,
        date=date,
    )


# ───── count_tokens ─────


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_increases_with_length():
    assert count_tokens("hi") < count_tokens("hello there friend, how are you?")


def test_count_tokens_deterministic():
    txt = "Thank you for applying to Vatic Labs"
    assert count_tokens(txt) == count_tokens(txt)


# ───── chunk_documents basic ─────


def test_chunk_documents_empty_input():
    assert chunk_documents([]) == []


def test_chunk_documents_skips_empty_text():
    doc = _make_doc(subject="", body="")
    assert chunk_documents([doc]) == []


def test_chunk_documents_short_doc_yields_one_chunk():
    doc = _make_doc(subject="Subj", body="Short body")
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "msg_000001__c000"
    assert c.doc_id == "msg_000001"
    assert c.chunk_index == 0
    assert "Subj" in c.text
    assert "Short body" in c.text
    assert c.n_tokens > 0


def test_chunk_documents_long_doc_yields_many_chunks():
    long_body = " ".join(["paragraph"] * 800)  # ≫ 512 tokens
    doc = _make_doc(body=long_body)
    chunks = chunk_documents([doc])
    assert len(chunks) > 1
    # Chunk indices are dense from 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Chunk IDs follow the documented format
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"msg_000001__c{i:03d}"


def test_chunk_documents_respects_chunk_size():
    long_body = " ".join(["word"] * 600)
    doc = _make_doc(body=long_body)
    chunks = chunk_documents([doc], chunk_size=256, chunk_overlap=32)
    # Every chunk should be at or below the budget (small overshoot is OK
    # because tiktoken counts may differ slightly from the splitter's view)
    for c in chunks:
        assert c.n_tokens <= 256 + 32, f"chunk too big: {c.n_tokens}"


def test_chunk_documents_overlap_creates_shared_text():
    """Overlap is intra-separator: chunks within the same passage share tokens.

    The splitter respects high-priority separators (e.g. the ``\\n\\n``
    between subject and body) before falling back to overlap-based splits, so
    we look for overlap among consecutive body-only chunks rather than across
    the subject/body boundary.
    """
    long_body = " ".join([f"unique{i}" for i in range(400)])
    doc = _make_doc(subject="", body=long_body)  # no subject → no paragraph break
    chunks = chunk_documents([doc], chunk_size=128, chunk_overlap=32)
    assert len(chunks) >= 2, "expected the long body to produce multiple chunks"
    found_overlap = False
    for prev, curr in pairwise(chunks):
        if set(prev.text.split()) & set(curr.text.split()):
            found_overlap = True
            break
    assert found_overlap, "expected at least one pair of adjacent chunks to share tokens"


def test_chunk_documents_carries_metadata():
    date = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
    doc = _make_doc(
        subject="Thank you for applying to Acme",
        body="body text",
        sender="ats@acme.com",
        date=date,
    )
    chunks = chunk_documents([doc])
    assert chunks, "expected at least one chunk"
    c = chunks[0]
    assert c.sender == "ats@acme.com"
    assert c.subject == "Thank you for applying to Acme"
    assert c.date == date


def test_chunk_documents_multiple_documents_get_distinct_ids():
    docs = [
        _make_doc(doc_id="msg_000001", body="alpha"),
        _make_doc(doc_id="msg_000002", body="beta"),
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) == 2
    ids = {c.chunk_id for c in chunks}
    assert ids == {"msg_000001__c000", "msg_000002__c000"}


def test_chunk_documents_deterministic():
    doc = _make_doc(body="some text to chunk")
    a = chunk_documents([doc])
    b = chunk_documents([doc])
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


# ───── validation / edge cases ─────


def test_chunk_documents_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_documents([_make_doc(body="x")], chunk_size=10, chunk_overlap=10)


def test_chunk_documents_rejects_overlap_bigger_than_size():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_documents([_make_doc(body="x")], chunk_size=10, chunk_overlap=20)


# ───── Chunk model invariants ─────


def test_chunk_is_frozen():
    chunk = Chunk(
        chunk_id="x__c000",
        doc_id="x",
        chunk_index=0,
        text="t",
        n_tokens=1,
    )
    with pytest.raises(ValidationError):
        chunk.text = "boom"  # type: ignore[misc]


def test_chunk_rejects_negative_index():
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="x__c000",
            doc_id="x",
            chunk_index=-1,
            text="t",
            n_tokens=1,
        )


def test_chunk_rejects_empty_text():
    with pytest.raises(ValidationError):
        Chunk(chunk_id="x__c000", doc_id="x", chunk_index=0, text="", n_tokens=0)


def test_default_constants_are_sensible():
    assert DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE
    assert DEFAULT_CHUNK_OVERLAP > 0
