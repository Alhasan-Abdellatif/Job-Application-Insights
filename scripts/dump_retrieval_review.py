"""Dump every retriever's top-K per golden-set question for manual review.

For each question, runs dense / BM25 / hybrid / rerank, then emits the
full email content of every retrieved chunk annotated with:

* ``✓`` — chunk is in the golden set's relevant list (true positive)
* ``✗`` — chunk is NOT in the golden set (either a false positive, OR a
  true positive the curator missed — review-time decision)

Output goes to ``evals/retrieval_review.md`` (gitignored — raw email
content).

Run::

    uv run python scripts/dump_retrieval_review.py
"""

from __future__ import annotations

from pathlib import Path

from job_application_insights.evals.golden_set import load_golden_set
from job_application_insights.evals.runner import (
    make_bm25_retriever,
    make_dense_retriever,
)
from job_application_insights.ingest.embed import Embedder
from job_application_insights.retrieval.bm25 import BM25Index
from job_application_insights.retrieval.hybrid import make_hybrid_retriever
from job_application_insights.retrieval.rerank import (
    CrossEncoderReranker,
    make_reranked_retriever,
)
from job_application_insights.retrieval.vector_store import VectorStore

GOLDEN_PATH = Path("./evals/golden_set.jsonl")
STORE_PATH = Path("./data/chroma")
OUT_PATH = Path("./evals/retrieval_review.md")
K = 8
MAX_BODY_CHARS = 1500


def main() -> None:
    print(f"Loading golden set from {GOLDEN_PATH}…")
    entries = load_golden_set(GOLDEN_PATH)
    print(f"  {len(entries)} entries")

    print(f"Opening vector store at {STORE_PATH}…")
    store = VectorStore(STORE_PATH)
    chunks = store.iter_chunks()
    by_id = {c.chunk_id: c for c in chunks}
    print(f"  {len(by_id):,} chunks indexed")

    print("Building retrievers (this loads BGE-small + BGE-reranker)…")
    embedder = Embedder()
    dense = make_dense_retriever(store, embedder)
    bm25_index = BM25Index(chunks)
    bm25 = make_bm25_retriever(bm25_index)
    hybrid = make_hybrid_retriever([dense, bm25])
    reranker = CrossEncoderReranker()
    text_by_id = {c.chunk_id: c.text for c in chunks}
    rerank = make_reranked_retriever(hybrid, reranker, text_by_id.__getitem__)
    retrievers: list[tuple[str, object]] = [
        ("dense", dense),
        ("bm25", bm25),
        ("hybrid", hybrid),
        ("rerank", rerank),
    ]

    lines: list[str] = []
    lines.append("# Retrieval review")
    lines.append("")
    lines.append(
        f"For each question, the top-{K} from each retriever, annotated "
        f"with ✓ (in golden set) or ✗ (not in golden set)."
    )
    lines.append("")
    lines.append("**How to read this file:**")
    lines.append("")
    lines.append("- **✓** = the retriever returned a chunk that's in your golden set.")
    lines.append(
        "- **✗** = the retriever returned something not in the golden set. "
        "Either a false positive (irrelevant) OR a *missed* true positive — "
        "use these to grow the golden set."
    )
    lines.append("")
    lines.append("**Workflow:**")
    lines.append("")
    lines.append(
        "1. For each ✗ row, decide: false positive, or missed positive? "
        "If missed, add the chunk_id to the question's `relevant_chunk_ids` "
        "in `evals/golden_set.jsonl`."
    )
    lines.append(
        "2. For each ✓ row, sanity-check the chunk actually answers the question. "
        "If it doesn't, remove it from the golden set."
    )
    lines.append("3. Re-run `uv run jai eval --retriever {dense,bm25,hybrid,rerank}` after edits.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for q_idx, entry in enumerate(entries, start=1):
        groups = entry.groups
        relevant = entry.relevant
        lines.append(f"## Q{q_idx}. {entry.question}")
        lines.append("")
        lines.append(
            f"*Ground truth: {len(groups)} answer group(s), "
            f"{sum(len(g) for g in groups)} chunk(s) total*"
        )
        lines.append("")

        for name, fn in retrievers:
            ids: list[str] = fn(entry.question, K)  # type: ignore[operator]
            top_k_set = set(ids)
            groups_hit = sum(1 for g in groups if g & top_k_set)
            recall = groups_hit / len(groups) if groups else 0.0
            lines.append(
                f"### {name} — R@{K} = {recall:.3f}  "
                f"({groups_hit}/{len(groups)} answer groups hit)"
            )
            lines.append("")
            if not ids:
                lines.append("_(no results)_")
                lines.append("")
                continue
            for rank, cid in enumerate(ids, start=1):
                marker = "✓" if cid in relevant else "✗"
                chunk = by_id.get(cid)
                if chunk is None:
                    lines.append(f"**{rank}. {marker} `{cid}` — ⚠️ NOT FOUND IN STORE**")
                    lines.append("")
                    continue
                date_str = chunk.date.isoformat() if chunk.date else "(no date)"
                body = (chunk.text or "").strip()
                if len(body) > MAX_BODY_CHARS:
                    body = body[:MAX_BODY_CHARS] + " …"
                lines.append(f"**{rank}. {marker} `{cid}`**")
                lines.append("")
                lines.append(f"- From: {chunk.sender or '(none)'}")
                lines.append(f"- Subject: {chunk.subject or '(none)'}")
                lines.append(f"- Date: {date_str}")
                lines.append("")
                lines.append("```text")
                lines.append(body if body else "(empty body)")
                lines.append("```")
                lines.append("")
            lines.append("")

        lines.append("---")
        lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
