"""Ground-truth dataset for retrieval evaluation.

A *golden set* is a small, hand-curated collection of
``(question, expected_chunk_ids)`` pairs. The retriever's job at eval
time is to surface those expected IDs in its top-K results; the metrics
in :mod:`.metrics` score how well it does.

File format — JSONL
-------------------
Each line is one independent JSON object::

    {"question": "Which universities did I apply to?",
     "relevant_chunk_ids": ["msg_001__c000", "msg_047__c002"],
     "tags": ["aggregation"], "notes": ""}
    {"question": "Did I hear back from DeepMind?",
     "relevant_chunk_ids": ["msg_201__c000"]}

JSONL (JSON-Lines) is the de-facto format for ML datasets because:

* You can **append** with a single new line — no whole-file rewrite.
* Streaming readers handle huge files without loading them into memory.
* Git diffs are tight: changing one entry changes exactly one line.

Blank lines are tolerated so you can group entries by hand. Comments
are *not* supported (it's still strict JSON, one per line).

What this module provides
-------------------------
* :class:`GoldenEntry` — Pydantic model for one row.
* :func:`load_golden_set` / :func:`save_golden_set` — round-trippable
  I/O at the boundary, with line-numbered error messages so a typo in
  a 50-line file doesn't send you on a hunt.

The model is the only public type the rest of the project depends on.
The runner module asks ``entry.relevant`` to get a ``set[str]`` ready to
hand straight to the metric functions. That convenience exists so the
runner never has to know the model has a list under the hood — if we
later change the storage format (graded relevance, ranked relevance),
the runner doesn't change.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# ────────────────────────────── data model ──────────────────────────────


class GoldenEntry(BaseModel):
    """One row of the eval set: a question paired with ground-truth chunks.

    Fields
    ------
    question
        The natural-language question to ask the retriever. Stripped of
        surrounding whitespace on construction.
    relevant_chunk_ids
        IDs of every chunk that *should* be retrieved for this question.
        Must be non-empty (an entry with no answer is a data bug) and
        unique (duplicates would silently inflate ``len(relevant)``).
    tags
        Optional labels for slicing eval results — e.g. ``"aggregation"``,
        ``"single-fact"``, ``"negation"``. Defaults to empty.
    notes
        Free-text for the curator. Useful when an entry is intentionally
        tricky and you want to remember why. Defaults to empty.
    """

    model_config = ConfigDict(frozen=True)

    question: str = Field(..., min_length=1)
    relevant_chunk_ids: list[str] = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    notes: str = Field(default="")

    @field_validator("question")
    @classmethod
    def _strip_question(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("question must contain non-whitespace characters")
        return stripped

    @field_validator("relevant_chunk_ids")
    @classmethod
    def _validate_chunk_ids(cls, v: list[str]) -> list[str]:
        if any(not cid or not cid.strip() for cid in v):
            raise ValueError("relevant_chunk_ids may not contain empty strings")
        if len(set(v)) != len(v):
            raise ValueError("relevant_chunk_ids must be unique within an entry")
        return v

    @property
    def relevant(self) -> set[str]:
        """``relevant_chunk_ids`` as a ``set`` — the shape metrics expect."""
        return set(self.relevant_chunk_ids)


# ────────────────────────────── I/O ──────────────────────────────


def load_golden_set(path: str | Path) -> list[GoldenEntry]:
    """Read a JSONL golden-set file into a list of :class:`GoldenEntry`.

    Blank lines are skipped. Parse or validation errors are re-raised
    with the file and line number so the curator can fix the offending
    row immediately.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist (propagated from ``Path.open``).
    ValueError
        On a malformed line, with ``file:line`` prefix and the original
        diagnostic.
    """
    path = Path(path)
    entries: list[GoldenEntry] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            try:
                entry = GoldenEntry.model_validate(obj)
            except ValidationError as exc:
                raise ValueError(f"{path}:{line_no}: invalid golden-set entry: {exc}") from exc
            entries.append(entry)
    return entries


def save_golden_set(entries: Iterable[GoldenEntry], path: str | Path) -> None:
    """Write entries to a JSONL file, overwriting any existing content.

    Parent directories are created if missing. One entry per line, with
    a trailing newline on each. ``model_dump_json`` is used so the
    serialisation is exactly what Pydantic would re-parse on load.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(entry.model_dump_json() + "\n")
