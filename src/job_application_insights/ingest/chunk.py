"""Split :class:`Document` instances into embedding-ready :class:`Chunk` passages.

Why we chunk:

* An entire long email body might be 2,000 tokens, but only one paragraph is
  relevant to a given query. Embedding the whole body dilutes the semantic
  signal — the embedding becomes an "average" of many topics.
* Retrieval works at the chunk level, so the chunk is the smallest "unit of
  relevance" we can hand to the LLM.
* The LLM's context window is finite. If we retrieve k chunks per query, each
  chunk must be small enough that k of them fit comfortably alongside the
  system prompt and the question.

Chunking strategy:

* **RecursiveCharacterTextSplitter** (LangChain) tries to split on natural
  document boundaries first — paragraphs → lines → sentences → words — and
  only falls back to character splits when nothing better is available.
  This respects document structure far better than naive fixed-size slicing.
* **Token-aware**: chunk size and overlap are measured in tokens (via
  ``tiktoken``'s ``cl100k_base``), not characters. Tokens are what the LLM
  charges and what context windows are sized in.
* **Overlap (64 tokens, ~12%)**: chunks share a small tail/head with their
  neighbours so a sentence cut at a boundary is still captured intact in at
  least one chunk.
* **Subject + body merged**: we chunk ``doc.text``, which prepends the subject
  to the body. Subjects often carry the strongest keyword signal ("Thank you
  for applying to Acme") and we want them retrievable even when the body is
  empty (Workday image-only emails).
"""

from __future__ import annotations

from datetime import datetime

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field

from job_application_insights.ingest.parse import Document

# ────────────────────────────── constants ──────────────────────────────

DEFAULT_CHUNK_SIZE: int = 512
"""Target chunk size in tokens. 512 is a balanced choice — big enough to hold
a typical email paragraph, small enough that 8-10 chunks fit comfortably in
the LLM prompt alongside the question."""

DEFAULT_CHUNK_OVERLAP: int = 64
"""Token overlap between adjacent chunks (~12% of chunk size). Standard
recommendation — bigger overlap costs storage with diminishing recall gain."""

_TIKTOKEN_ENCODING = "cl100k_base"
"""OpenAI's tokenizer used for the text-embedding-3 and GPT-4 families.
For BGE-style embedding models the count is a small overestimate (~10%) —
fine for budgeting; we'll switch to the embedding model's own tokenizer in
Week 2 if quality demands it."""


# ────────────────────────────── data model ──────────────────────────────


class Chunk(BaseModel):
    """A passage of a Document, ready for embedding + retrieval.

    Attributes
    ----------
    chunk_id
        Globally unique within a corpus, formatted ``<doc_id>__c<NNN>``.
    doc_id
        The parent :class:`Document`'s id — preserved for citation.
    chunk_index
        Zero-based position of this chunk within the parent document.
    text
        The actual chunk content.
    n_tokens
        Token count under ``cl100k_base``. Useful for prompt budgeting.
    sender, subject, date
        Carried-over metadata. Filtering on these is much cheaper than
        re-joining against the original Document collection at query time.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0)
    text: str = Field(..., min_length=1)
    n_tokens: int = Field(..., ge=0)

    sender: str = ""
    subject: str = ""
    date: datetime | None = None


# ────────────────────────────── helpers ──────────────────────────────


def count_tokens(text: str, encoding_name: str = _TIKTOKEN_ENCODING) -> int:
    """Return the number of tokens in ``text`` under the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Construct a token-aware recursive splitter.

    Splitters are stateless except for their configuration, so we build a fresh
    one per call rather than caching a module-level singleton — keeps tests
    parallel-safe and lets callers override chunk sizing without surprises.
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=_TIKTOKEN_ENCODING,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Order matters — splitter tries each separator in turn before falling
        # back to character splits. Paragraph break → newline → sentence → word.
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )


# ────────────────────────────── public API ──────────────────────────────


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split each :class:`Document` into one or more :class:`Chunk` passages.

    Parameters
    ----------
    documents
        Validated Documents from the ingestion layer.
    chunk_size
        Target tokens per chunk (default 512).
    chunk_overlap
        Tokens shared between adjacent chunks (default 64).

    Returns
    -------
    A flat list of Chunks. Each Document contributes ``ceil(text_tokens / (chunk_size
    - chunk_overlap))`` chunks (approximately). Documents whose ``.text`` is
    empty are skipped entirely.

    Notes
    -----
    Chunk IDs are deterministic: re-running ``chunk_documents`` on the same
    Documents yields identical chunk IDs. This lets a later embedding step do
    incremental work (only embed chunk_ids it hasn't seen).
    """
    if chunk_size <= chunk_overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be greater than chunk_overlap ({chunk_overlap})"
        )

    splitter = _make_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks: list[Chunk] = []
    for doc in documents:
        text = doc.text
        if not text:
            continue

        slices = splitter.split_text(text)
        # Defensive: splitter occasionally returns [] for trivially short inputs.
        if not slices:
            slices = [text]

        for chunk_index, piece in enumerate(slices):
            piece_clean = piece.strip()
            if not piece_clean:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}__c{chunk_index:03d}",
                    doc_id=doc.doc_id,
                    chunk_index=chunk_index,
                    text=piece_clean,
                    n_tokens=count_tokens(piece_clean),
                    sender=doc.sender,
                    subject=doc.subject,
                    date=doc.date,
                )
            )

    return chunks
