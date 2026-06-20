"""Print top-K retrieval candidates per question so a human can pick relevance.

Run with::

    uv run python scripts/curate_golden_set.py <batch_name>

where ``batch_name`` is one of: ``A``, ``B``, ``C``, ``D``, ``E``, ``all``.
Each batch loads the embedder + store once and prints candidates in a
scannable format (chunk_id, similarity, sender, subject, snippet).

This is a one-off curation aid; output is meant for the assistant + user
to triage together. Not part of the library.
"""

from __future__ import annotations

import sys
from pathlib import Path

from job_application_insights.ingest.embed import Embedder
from job_application_insights.retrieval.vector_store import VectorStore

STORE_PATH = Path("./data/chroma")

BATCHES: dict[str, list[tuple[int, str, int]]] = {
    # (question_id, question, k_for_this_question)
    "A": [
        (1, "Did I apply to Anthropic?", 15),
        (2, "What role did I apply for at Qualcomm?", 15),
        (3, "Did Imperial College acknowledge my application?", 15),
        (4, "What did the Ellison Institute say in their email?", 15),
    ],
    "B": [
        (5, "Which universities did I apply to?", 30),
        (6, "Which companies invited me to an interview?", 30),
        (7, "Which companies sent me rejection emails?", 30),
        (8, "Which postdoc positions did I apply for?", 30),
        (9, "Which research engineer roles did I apply to?", 30),
    ],
    "C": [
        (10, "Did I apply to ETH Zurich?", 15),
        (11, "Did I get an acknowledgment from GSK?", 15),
        (12, "Show me the Amazon application email", 15),
        (13, "Did I apply to ALESAYI HOLDING?", 15),
    ],
    "D": [
        (14, "Where did I apply for academic jobs?", 20),
        (15, "Which AI safety positions did I apply to?", 20),
        (16, "Did I apply for any roles working on large language models?", 20),
    ],
    "E": [
        (17, "Did any startup acknowledge my application?", 20),
        (18, "Which UK-based companies did I apply to?", 20),
    ],
}


def truncate(s: str | None, n: int) -> str:
    if not s:
        return "(none)"
    s = " ".join(s.split())  # collapse whitespace
    return s if len(s) <= n else s[: n - 1] + "…"


def run_batch(batch_name: str, embedder: Embedder, store: VectorStore) -> None:
    questions = BATCHES[batch_name]
    for qid, question, k in questions:
        print()
        print("═" * 90)
        print(f"Q{qid}: {question}     (k={k})")
        print("═" * 90)
        q_vec = embedder.embed([question])[0]
        results = store.query(q_vec, k=k)
        if not results:
            print("  (no results)")
            continue
        for idx, r in enumerate(results, start=1):
            print(
                f"  [{idx:02d}] {r.chunk.chunk_id}  sim={r.score:+.3f}"
                f"  sender={truncate(r.chunk.sender, 45)}"
            )
            print(f"       subj: {truncate(r.chunk.subject, 100)}")
            print(f"       text: {truncate(r.chunk.text, 170)}")


def find_by_entity(store: VectorStore, needle: str, limit: int = 30) -> None:
    """List chunks whose sender or subject contains ``needle`` (case-insensitive).

    Bypasses the retriever — used as an oracle to confirm the corpus
    really does (or doesn't) contain the entity we're asking about.
    """
    needle_low = needle.lower()
    collection = store._collection  # — intentional inspection
    data = collection.get(include=["metadatas"])
    metas = data.get("metadatas") or []
    ids = data.get("ids") or []
    print(f"\nLooking for chunks whose sender/subject contains {needle!r}…")
    print("─" * 90)
    hits = 0
    for cid, meta in zip(ids, metas, strict=False):
        sender = (meta or {}).get("sender", "") or ""
        subject = (meta or {}).get("subject", "") or ""
        if needle_low in sender.lower() or needle_low in subject.lower():
            hits += 1
            print(f"  [{hits:02d}] {cid}")
            print(f"       sender: {truncate(sender, 80)}")
            print(f"       subj:   {truncate(subject, 100)}")
            if hits >= limit:
                print(f"  … (limit {limit} reached)")
                return
    if hits == 0:
        print("  (no chunks matched)")
    else:
        print(f"\n  {hits} chunk(s) matched.")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(f"usage: {sys.argv[0]} {{A|B|C|D|E|all}} | find <entity>", file=sys.stderr)
        return 2

    print("Opening vector store…", file=sys.stderr)
    store = VectorStore(STORE_PATH)
    print(f"  {store.n_chunks:,} chunks indexed", file=sys.stderr)

    if args[0] == "find":
        if len(args) < 2:
            print("usage: ... find <entity-substring>", file=sys.stderr)
            return 2
        find_by_entity(store, " ".join(args[1:]))
        return 0

    if args[0] not in {*BATCHES, "all"}:
        print(f"unknown batch {args[0]!r}; expected one of A|B|C|D|E|all|find", file=sys.stderr)
        return 2

    print("Loading embedder (BGE-small-en-v1.5)…", file=sys.stderr)
    embedder = Embedder()
    print(f"  dim={embedder.dimension}", file=sys.stderr)

    batches_to_run = list(BATCHES) if args[0] == "all" else [args[0]]
    for name in batches_to_run:
        run_batch(name, embedder, store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
