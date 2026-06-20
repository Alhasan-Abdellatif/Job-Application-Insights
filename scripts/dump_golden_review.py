"""Dump the golden set as human-readable Markdown for manual review.

Each entry shows the question and the full sender/subject/date/body of
every chunk marked relevant. Output goes to evals/golden_set_review.md
(gitignored — it contains raw email content).

Run::

    uv run python scripts/dump_golden_review.py
"""

from __future__ import annotations

from pathlib import Path

from job_application_insights.evals.golden_set import load_golden_set
from job_application_insights.retrieval.vector_store import VectorStore

GOLDEN_PATH = Path("./evals/golden_set.jsonl")
STORE_PATH = Path("./data/chroma")
OUT_PATH = Path("./evals/golden_set_review.md")
MAX_BODY_CHARS = 4000  # truncate very long bodies


def main() -> None:
    print(f"Loading golden set from {GOLDEN_PATH}…")
    entries = load_golden_set(GOLDEN_PATH)
    print(f"  {len(entries)} entries")

    print(f"Opening vector store at {STORE_PATH}…")
    store = VectorStore(STORE_PATH)
    chunks = store.iter_chunks()
    by_id = {c.chunk_id: c for c in chunks}
    print(f"  {len(by_id):,} chunks indexed")

    lines: list[str] = []
    lines.append("# Golden set review")
    lines.append("")
    lines.append(f"Source: `{GOLDEN_PATH}` — {len(entries)} entries")
    lines.append("")
    lines.append(
        "For each question, every chunk marked **relevant** is dumped with its "
        "sender, subject, date, and body. Use this to spot-check whether a "
        "chunk *actually* answers the question — and whether the set is missing "
        "obvious positives."
    )
    lines.append("")
    lines.append("**Instructions for review:**")
    lines.append("")
    lines.append(
        "- For each chunk: is this *actually* a positive for the question? If no, mark it."
    )
    lines.append("- For each question: are there *other* emails you remember that should be here?")
    lines.append("- Edit `evals/golden_set.jsonl` directly to fix.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, entry in enumerate(entries, start=1):
        lines.append(f"## Q{i}. {entry.question}")
        lines.append("")
        if entry.tags:
            lines.append(f"*Tags:* {', '.join(entry.tags)}")
        if entry.notes:
            lines.append(f"*Notes:* {entry.notes}")
        lines.append("")
        groups = entry.groups
        total_chunks = sum(len(g) for g in groups)
        lines.append(f"**{len(groups)} answer group(s), {total_chunks} chunk(s) total:**")
        lines.append("")

        flat_chunk_ids: list[str] = []
        for gi, group in enumerate(groups, start=1):
            if len(groups) > 1:
                lines.append(f"#### Answer group {gi} ({len(group)} chunk(s))")
                lines.append("")
            flat_chunk_ids.extend(sorted(group))
            if len(groups) > 1:
                lines.append("")

        for j, cid in enumerate(flat_chunk_ids, start=1):
            chunk = by_id.get(cid)
            if chunk is None:
                lines.append(f"### {j}. `{cid}` — ⚠️ NOT FOUND IN STORE")
                lines.append("")
                continue
            date_str = chunk.date.isoformat() if chunk.date else "(no date)"
            body = (chunk.text or "").strip()
            truncated = ""
            if len(body) > MAX_BODY_CHARS:
                body = body[:MAX_BODY_CHARS] + " …"
                truncated = f" *(body truncated to {MAX_BODY_CHARS} chars)*"
            lines.append(f"### {j}. `{cid}`{truncated}")
            lines.append("")
            lines.append(f"- **From:** {chunk.sender or '(none)'}")
            lines.append(f"- **Subject:** {chunk.subject or '(none)'}")
            lines.append(f"- **Date:** {date_str}")
            lines.append("")
            lines.append("```text")
            lines.append(body if body else "(empty body)")
            lines.append("```")
            lines.append("")

        lines.append("---")
        lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
