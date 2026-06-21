"""Tests for the DuckDB-backed structured applications table."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest
from job_application_insights.ingest.parse import load_documents
from job_application_insights.structured.table import (
    TABLE_NAME,
    _naive_utc,
    load_applications_table,
)

# Two GSK rows (one duplicated), one Edinburgh, one MediaTek with no Role.
SAMPLE_CSV = (
    "From,Subject,Date,Body,Company,Role\n"
    '"recruiter@gsk.com","Thank you for applying to GSK",'
    '"Mon, 12 Aug 2024 10:00:00 +0000","GSK body","GSK","Data Scientist"\n'
    '"alumni@ed.ac.uk","Application received",'
    '"Tue, 5 Mar 2024 09:00:00 +0000","Edinburgh body",'
    '"University of Edinburgh","Research Associate"\n'
    '"recruiter@gsk.com","Thank you for applying to GSK",'
    '"Mon, 12 Aug 2024 10:00:00 +0000","GSK body","GSK","Data Scientist"\n'
    '"hr@mediatek.com","Application received",'
    '"Wed, 21 Feb 2024 08:00:00 +0000","MediaTek body","MediaTek",\n'
)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    path = tmp_path / "ack.csv"
    path.write_text(SAMPLE_CSV)
    return path


def test_loads_table_and_dedups(sample_csv: Path) -> None:
    con = load_applications_table(sample_csv)
    row = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()
    assert row is not None
    assert row[0] == 3


def test_doc_ids_are_sequential(sample_csv: Path) -> None:
    con = load_applications_table(sample_csv)
    doc_ids = [
        r[0] for r in con.execute(f"SELECT doc_id FROM {TABLE_NAME} ORDER BY doc_id").fetchall()
    ]
    assert doc_ids == ["msg_000000", "msg_000001", "msg_000002"]


def test_schema_matches_design(sample_csv: Path) -> None:
    con = load_applications_table(sample_csv)
    names = [r[0] for r in con.execute(f"DESCRIBE {TABLE_NAME}").fetchall()]
    assert names == ["doc_id", "from_addr", "subject", "date", "company", "role"]


def test_empty_role_does_not_leak_nan(sample_csv: Path) -> None:
    """Empty Role cells must be the empty string, not the literal 'nan'.

    Same data-hygiene bug class as the Week-2 NaN-string leak in the
    embeddings. A leaking 'nan' here would mean ``SELECT role FROM
    applications WHERE role = ''`` returns nothing while the user can
    see (in the source) that several roles are blank.
    """
    con = load_applications_table(sample_csv)
    rows = con.execute(f"SELECT role FROM {TABLE_NAME} WHERE company = 'MediaTek'").fetchall()
    assert rows == [("",)]


def test_count_query_works(sample_csv: Path) -> None:
    con = load_applications_table(sample_csv)
    row = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE company = 'GSK'").fetchone()
    assert row is not None
    assert row[0] == 1


def test_date_is_parsed(sample_csv: Path) -> None:
    con = load_applications_table(sample_csv)
    row = con.execute(f"SELECT date FROM {TABLE_NAME} WHERE company = 'GSK'").fetchone()
    assert row is not None
    # DuckDB returns datetime objects for TIMESTAMP columns.
    assert row[0] is not None
    assert row[0].year == 2024
    assert row[0].month == 8
    assert row[0].day == 12


def test_unparseable_date_becomes_null(tmp_path: Path) -> None:
    """Garbage in the Date column should become NULL, not raise."""
    content = (
        "From,Subject,Date,Body,Company,Role\n"
        '"x@y.com","Subj","not a real date","body","Acme","Eng"\n'
    )
    path = tmp_path / "bad_date.csv"
    path.write_text(content)
    con = load_applications_table(path)
    row = con.execute(f"SELECT date FROM {TABLE_NAME}").fetchone()
    assert row == (None,)


def test_naive_utc_helper_handles_all_branches() -> None:
    """``_naive_utc`` covers three input shapes: None, naive, tz-aware."""
    assert _naive_utc(None) is None

    naive = datetime(2024, 1, 1, 12, 0, 0)
    assert _naive_utc(naive) is naive  # passthrough

    aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    out = _naive_utc(aware)
    assert out is not None
    assert out.tzinfo is None
    assert out.hour == 12


def test_subject_is_mime_decoded(tmp_path: Path) -> None:
    """RFC-2047 encoded subjects must be decoded — the same fix that made
    the Week-2 ALESAYI question matchable in the chunked corpus.
    """
    # Base64 of "ALESAYI HOLDING" (ASCII, harmless test payload).
    content = (
        "From,Subject,Date,Body,Company,Role\n"
        '"hr@alesayi.com","=?UTF-8?B?QUxFU0FZSSBIT0xESU5H?=",'
        '"Mon, 1 Jan 2024 10:00:00 +0000","Body","ALESAYI HOLDING",\n'
    )
    path = tmp_path / "alesayi.csv"
    path.write_text(content)
    con = load_applications_table(path)
    row = con.execute(f"SELECT subject FROM {TABLE_NAME}").fetchone()
    assert row is not None
    assert row[0] == "ALESAYI HOLDING"


def test_missing_columns_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("From,Subject\nx,y\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_applications_table(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_applications_table(tmp_path / "nope.csv")


def test_existing_connection_is_reused(sample_csv: Path) -> None:
    """Passing a connection keeps its other state intact while replacing the table."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE other (x INTEGER)")
    con.execute("INSERT INTO other VALUES (42)")

    returned = load_applications_table(sample_csv, connection=con)

    assert returned is con
    other_row = con.execute("SELECT x FROM other").fetchone()
    assert other_row == (42,)
    app_row = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()
    assert app_row is not None
    assert app_row[0] == 3


def test_reload_is_idempotent(sample_csv: Path) -> None:
    """Loading twice into the same connection rebuilds, doesn't append."""
    con = load_applications_table(sample_csv)
    load_applications_table(sample_csv, connection=con)
    row = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()
    assert row is not None
    assert row[0] == 3


def test_doc_ids_match_parser_output(sample_csv: Path) -> None:
    """The load-bearing invariant: structured table row N has the same
    ``doc_id`` as the Nth Document the parser yields from the same CSV.

    This is what lets the future router cross-walk a structured hit (a
    company name + count) to a RAG citation (the underlying email chunks).
    Break this and the two engines drift silently — no exception, just
    wrong joins.
    """
    parsed = load_documents([sample_csv])
    con = load_applications_table(sample_csv)
    structured_ids = [
        r[0] for r in con.execute(f"SELECT doc_id FROM {TABLE_NAME} ORDER BY doc_id").fetchall()
    ]
    parsed_ids = sorted(d.doc_id for d in parsed)
    assert structured_ids == parsed_ids
