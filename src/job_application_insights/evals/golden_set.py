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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# ────────────────────────────── data model ──────────────────────────────


class GoldenEntry(BaseModel):
    """One row of the eval set: a question paired with its ground truth.

    Two complementary representations are supported:

    * ``relevant_chunk_ids`` — flat list. Every listed chunk is its own
      independent "answer unit". Recall = fraction of chunks found.
      Right for factoid questions ("Did I apply to ALESAYI?") and for
      single-entity questions where every chunk matters individually.
    * ``answer_groups`` — list of lists. Each inner list is one *answer*;
      finding any chunk from that list counts as finding the answer.
      Right for aggregation questions ("Which companies invited me to
      an interview?") where the answer is a set of entities and the
      question is whether each entity surfaces — independent of how
      many chunks per entity exist.

    Exactly one of the two is required (validator enforces it). Internally
    the :attr:`groups` property gives a single canonical view: a list of
    sets, one per answer. The runner always calls the group-aware metrics
    against ``groups``, so the user sees a unified recall semantics
    regardless of which field they used.

    Fields
    ------
    question
        The natural-language question. Stripped of surrounding whitespace.
    relevant_chunk_ids
        Flat list of chunk IDs (legacy / factoid form).
    answer_groups
        List of chunk-ID lists, one list per distinct answer.
    tags
        Optional labels for slicing eval results.
    notes
        Free-text curator notes.
    """

    model_config = ConfigDict(frozen=True)

    question: str = Field(..., min_length=1)
    relevant_chunk_ids: list[str] | None = Field(default=None)
    answer_groups: list[list[str]] | None = Field(default=None)
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
    def _validate_chunk_ids(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if not v:
            raise ValueError("relevant_chunk_ids may not be an empty list")
        if any(not cid or not cid.strip() for cid in v):
            raise ValueError("relevant_chunk_ids may not contain empty strings")
        if len(set(v)) != len(v):
            raise ValueError("relevant_chunk_ids must be unique within an entry")
        return v

    @field_validator("answer_groups")
    @classmethod
    def _validate_answer_groups(cls, v: list[list[str]] | None) -> list[list[str]] | None:
        if v is None:
            return v
        if not v:
            raise ValueError("answer_groups may not be an empty list")
        seen: set[str] = set()
        for i, group in enumerate(v):
            if not group:
                raise ValueError(f"answer_groups[{i}] is empty; every group needs ≥1 chunk")
            if any(not cid or not cid.strip() for cid in group):
                raise ValueError(f"answer_groups[{i}] contains empty chunk IDs")
            if len(set(group)) != len(group):
                raise ValueError(f"answer_groups[{i}] has duplicate chunks within the group")
            for cid in group:
                if cid in seen:
                    raise ValueError(f"chunk {cid!r} appears in multiple answer_groups")
                seen.add(cid)
        return v

    @model_validator(mode="after")
    def _require_one_form(self) -> GoldenEntry:
        if self.relevant_chunk_ids is None and self.answer_groups is None:
            raise ValueError("entry must provide either relevant_chunk_ids or answer_groups")
        if self.relevant_chunk_ids is not None and self.answer_groups is not None:
            raise ValueError(
                "entry must provide exactly one of relevant_chunk_ids OR answer_groups, not both"
            )
        return self

    @property
    def groups(self) -> list[set[str]]:
        """Canonical view: one ``set[str]`` per answer.

        - If ``answer_groups`` was provided, return it as a list of sets.
        - Otherwise convert ``relevant_chunk_ids`` to one chunk per
          group — recovering the traditional chunk-level recall semantics.
        """
        if self.answer_groups is not None:
            return [set(group) for group in self.answer_groups]
        # relevant_chunk_ids is guaranteed non-None by the model validator
        assert self.relevant_chunk_ids is not None
        return [{cid} for cid in self.relevant_chunk_ids]

    @property
    def relevant(self) -> set[str]:
        """All chunks across all answer groups (union). Kept for back-compat."""
        result: set[str] = set()
        for group in self.groups:
            result |= group
        return result


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
