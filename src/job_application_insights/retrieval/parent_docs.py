"""Parent-document retrieval: expand each retrieved chunk to its full email.

The Week 2 retriever surfaces the most relevant *chunk* of a document.
But a chunk is a 512-token slice; the LLM seeing only that slice can
miss context that lived in adjacent chunks of the same email. Parent-
document retrieval is the simple fix: for every retrieved chunk,
expand to the **full parent email** at generation time.

What changes:

* **Citation handle** — the synthetic parent result uses ``doc_id`` as
  its ``chunk_id`` (the citation now refers to the email, not the
  slice). Multiple retrieved chunks from the same email collapse into
  one parent result.
* **Text seen by the LLM** — concatenation of *all* chunks for that
  email, in ``chunk_index`` order. No gaps, no truncation.
* **Score** — kept from the highest-scoring retrieved chunk in the
  group, so the parent results stay sorted in the same order the
  retriever produced.

What does *not* change:

* **Retrieval recall/precision metrics.** Expansion happens *after*
  retrieval — the retriever still finds the same chunks; we just
  show the LLM more of each. Week 2's R@8 numbers are unaffected.

The cost is token budget — each parent typically multiplies the
chunks-into-the-prompt ratio by 3-5x. Opt in via the ``--expand-parents``
CLI flag; off by default.
"""

from __future__ import annotations

from job_application_insights.ingest.chunk import Chunk
from job_application_insights.retrieval.vector_store import RetrievalResult


def _build_parent_index(chunks_by_id: dict[str, Chunk]) -> dict[str, list[Chunk]]:
    """Group chunks by ``doc_id``, sorted by ``chunk_index``.

    Pre-computing this avoids an O(N) scan per retrieved chunk during
    expansion. For a corpus of ~11k chunks the build is sub-second.
    """
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks_by_id.values():
        grouped.setdefault(chunk.doc_id, []).append(chunk)
    for siblings in grouped.values():
        siblings.sort(key=lambda c: c.chunk_index)
    return grouped


def expand_to_parent_documents(
    results: list[RetrievalResult],
    chunks_by_id: dict[str, Chunk],
    *,
    separator: str = "\n\n",
) -> list[RetrievalResult]:
    """Expand each retrieved chunk to its full parent email.

    Returns one :class:`RetrievalResult` per *unique* ``doc_id`` in
    ``results``, in the order they first appeared (so the top-ranked
    parent stays first). The synthetic chunk inside each result has:

    * ``chunk_id = doc_id`` — citation now points at the email.
    * ``chunk_index = 0`` — there's only one parent per doc.
    * ``text`` = concatenated text of every chunk in that doc, joined
      by ``separator``, in ``chunk_index`` order.
    * ``n_tokens`` = sum of n_tokens across all chunks (the LLM-budget
      side of the story).
    * sender / subject / date carried over from the first chunk.

    Parameters
    ----------
    results
        Output of any retriever (dense / BM25 / hybrid / rerank).
    chunks_by_id
        The full corpus, keyed by chunk_id. Used to find sibling chunks
        of each retrieved doc_id.
    separator
        Joiner between concatenated chunk texts. Default ``"\\n\\n"``
        preserves paragraph-style boundaries the LLM can parse.
    """
    if not results:
        return []
    parent_index = _build_parent_index(chunks_by_id)

    seen_doc_ids: set[str] = set()
    out: list[RetrievalResult] = []
    for result in results:
        doc_id = result.chunk.doc_id
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)

        siblings = parent_index.get(doc_id, [result.chunk])
        parent_text = separator.join(c.text for c in siblings)
        first = siblings[0]
        parent_chunk = Chunk(
            chunk_id=doc_id,
            doc_id=doc_id,
            chunk_index=0,
            text=parent_text,
            n_tokens=sum(c.n_tokens for c in siblings),
            sender=first.sender,
            subject=first.subject,
            date=first.date,
        )
        out.append(RetrievalResult(chunk=parent_chunk, score=result.score))
    return out
