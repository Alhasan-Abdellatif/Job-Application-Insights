"""Tests for :mod:`job_application_insights.evals.golden_set`."""

from __future__ import annotations

from pathlib import Path

import pytest
from job_application_insights.evals.golden_set import (
    GoldenEntry,
    load_golden_set,
    save_golden_set,
)
from pydantic import ValidationError

# ───── GoldenEntry: happy paths ─────


def test_golden_entry_basic():
    entry = GoldenEntry(
        question="Which universities did I apply to?",
        relevant_chunk_ids=["msg_001__c000", "msg_047__c002"],
    )
    assert entry.question == "Which universities did I apply to?"
    assert entry.relevant_chunk_ids == ["msg_001__c000", "msg_047__c002"]
    assert entry.tags == []
    assert entry.notes == ""


def test_golden_entry_with_tags_and_notes():
    entry = GoldenEntry(
        question="q",
        relevant_chunk_ids=["a"],
        tags=["aggregation", "list"],
        notes="tricky — answer scattered across many emails",
    )
    assert entry.tags == ["aggregation", "list"]
    assert "scattered" in entry.notes


def test_golden_entry_question_is_stripped():
    entry = GoldenEntry(question="  q  \n", relevant_chunk_ids=["a"])
    assert entry.question == "q"


def test_golden_entry_relevant_property_returns_set():
    entry = GoldenEntry(question="q", relevant_chunk_ids=["a", "b", "c"])
    assert entry.relevant == {"a", "b", "c"}
    assert isinstance(entry.relevant, set)


# ───── GoldenEntry: validation ─────


def test_golden_entry_rejects_empty_question():
    with pytest.raises(ValidationError):
        GoldenEntry(question="", relevant_chunk_ids=["a"])


def test_golden_entry_rejects_whitespace_only_question():
    with pytest.raises(ValidationError):
        GoldenEntry(question="   \n\t  ", relevant_chunk_ids=["a"])


def test_golden_entry_rejects_empty_relevant_list():
    with pytest.raises(ValidationError):
        GoldenEntry(question="q", relevant_chunk_ids=[])


def test_golden_entry_rejects_empty_chunk_id_in_list():
    with pytest.raises(ValidationError):
        GoldenEntry(question="q", relevant_chunk_ids=["a", "", "b"])


def test_golden_entry_rejects_whitespace_only_chunk_id():
    with pytest.raises(ValidationError):
        GoldenEntry(question="q", relevant_chunk_ids=["a", "   "])


def test_golden_entry_rejects_duplicate_chunk_ids():
    with pytest.raises(ValidationError, match="unique"):
        GoldenEntry(question="q", relevant_chunk_ids=["a", "b", "a"])


def test_golden_entry_is_frozen():
    entry = GoldenEntry(question="q", relevant_chunk_ids=["a"])
    with pytest.raises(ValidationError):
        entry.question = "new"  # type: ignore[misc]


# ───── I/O: round-trip ─────


def test_save_and_load_round_trip(tmp_path: Path):
    entries = [
        GoldenEntry(question="q1", relevant_chunk_ids=["a", "b"], tags=["x"]),
        GoldenEntry(question="q2", relevant_chunk_ids=["c"], notes="n"),
    ]
    path = tmp_path / "golden.jsonl"
    save_golden_set(entries, path)
    loaded = load_golden_set(path)
    assert loaded == entries


def test_save_creates_parent_directories(tmp_path: Path):
    path = tmp_path / "deeply" / "nested" / "g.jsonl"
    save_golden_set(
        [GoldenEntry(question="q", relevant_chunk_ids=["a"])],
        path,
    )
    assert path.exists()


def test_save_overwrites_existing_file(tmp_path: Path):
    path = tmp_path / "g.jsonl"
    save_golden_set([GoldenEntry(question="old", relevant_chunk_ids=["a"])], path)
    save_golden_set([GoldenEntry(question="new", relevant_chunk_ids=["b"])], path)
    loaded = load_golden_set(path)
    assert len(loaded) == 1
    assert loaded[0].question == "new"


# ───── I/O: blank-line tolerance ─────


def test_load_tolerates_blank_lines(tmp_path: Path):
    content = (
        '{"question": "q1", "relevant_chunk_ids": ["a"]}\n'
        "\n"
        "   \n"
        '{"question": "q2", "relevant_chunk_ids": ["b"]}\n'
    )
    path = tmp_path / "g.jsonl"
    path.write_text(content, encoding="utf-8")
    entries = load_golden_set(path)
    assert len(entries) == 2
    assert entries[0].question == "q1"
    assert entries[1].question == "q2"


def test_load_handles_empty_file(tmp_path: Path):
    path = tmp_path / "g.jsonl"
    path.write_text("", encoding="utf-8")
    assert load_golden_set(path) == []


# ───── I/O: error messages include line numbers ─────


def test_load_reports_line_number_on_bad_json(tmp_path: Path):
    content = (
        '{"question": "q1", "relevant_chunk_ids": ["a"]}\n'
        "{not valid json at all\n"
        '{"question": "q3", "relevant_chunk_ids": ["b"]}\n'
    )
    path = tmp_path / "g.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        load_golden_set(path)


def test_load_reports_line_number_on_bad_entry(tmp_path: Path):
    content = (
        '{"question": "q1", "relevant_chunk_ids": ["a"]}\n'
        '{"question": "", "relevant_chunk_ids": ["b"]}\n'
    )
    path = tmp_path / "g.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        load_golden_set(path)


def test_load_missing_file_raises_file_not_found(tmp_path: Path):
    path = tmp_path / "does-not-exist.jsonl"
    with pytest.raises(FileNotFoundError):
        load_golden_set(path)


# ───── I/O: accepts str or Path ─────


def test_load_and_save_accept_str_paths(tmp_path: Path):
    path_str = str(tmp_path / "g.jsonl")
    save_golden_set([GoldenEntry(question="q", relevant_chunk_ids=["a"])], path_str)
    loaded = load_golden_set(path_str)
    assert len(loaded) == 1


# ───── answer_groups ─────


def test_answer_groups_basic():
    entry = GoldenEntry(
        question="Which companies invited me to interview?",
        answer_groups=[["a", "b", "c"], ["d", "e"], ["f"]],
    )
    assert len(entry.groups) == 3
    assert entry.groups[0] == {"a", "b", "c"}
    assert entry.groups[1] == {"d", "e"}
    assert entry.groups[2] == {"f"}


def test_answer_groups_relevant_property_unions_all_groups():
    entry = GoldenEntry(question="q", answer_groups=[["a"], ["b", "c"]])
    assert entry.relevant == {"a", "b", "c"}


def test_relevant_chunk_ids_auto_converts_to_one_chunk_per_group():
    """Backwards compat: a flat list becomes one group per chunk."""
    entry = GoldenEntry(question="q", relevant_chunk_ids=["a", "b", "c"])
    assert len(entry.groups) == 3
    assert entry.groups == [{"a"}, {"b"}, {"c"}]


def test_either_one_form_required():
    with pytest.raises(ValidationError, match="either"):
        GoldenEntry(question="q")


def test_cannot_specify_both_forms():
    with pytest.raises(ValidationError, match="not both"):
        GoldenEntry(question="q", relevant_chunk_ids=["a"], answer_groups=[["b"]])


def test_answer_groups_rejects_empty_outer_list():
    with pytest.raises(ValidationError):
        GoldenEntry(question="q", answer_groups=[])


def test_answer_groups_rejects_empty_inner_list():
    with pytest.raises(ValidationError, match="empty"):
        GoldenEntry(question="q", answer_groups=[["a"], []])


def test_answer_groups_rejects_duplicate_within_group():
    with pytest.raises(ValidationError, match="duplicate"):
        GoldenEntry(question="q", answer_groups=[["a", "a"]])


def test_answer_groups_rejects_chunk_in_multiple_groups():
    """A chunk that belongs to two different answers is a curator error."""
    with pytest.raises(ValidationError, match="multiple"):
        GoldenEntry(question="q", answer_groups=[["a", "b"], ["b", "c"]])


def test_answer_groups_rejects_empty_chunk_id():
    with pytest.raises(ValidationError):
        GoldenEntry(question="q", answer_groups=[["a", ""]])


def test_answer_groups_round_trip(tmp_path: Path):
    entries = [
        GoldenEntry(
            question="agg q",
            answer_groups=[["a", "b"], ["c"]],
            tags=["agg"],
        ),
        GoldenEntry(question="factoid q", relevant_chunk_ids=["d"]),
    ]
    path = tmp_path / "g.jsonl"
    save_golden_set(entries, path)
    loaded = load_golden_set(path)
    assert loaded == entries
    # The factoid entry still works the old way
    assert loaded[1].groups == [{"d"}]
    # The aggregation entry preserves groups
    assert loaded[0].groups == [{"a", "b"}, {"c"}]
