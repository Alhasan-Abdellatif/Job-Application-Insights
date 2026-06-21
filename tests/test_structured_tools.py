"""Tests for the typed Python tools over the structured table.

The tools wrap deterministic SQL — so we test deterministic behaviour:
right counts, right ordering, right filtering, right empty-result shape.
LLM-level behaviour (does the model pick the right tool with the right
arguments?) lives in Day 3's tool-use tests.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest
from job_application_insights.structured.table import load_applications_table
from job_application_insights.structured.tools import (
    CompanyCount,
    MonthCount,
    RoleRecord,
    applications_by_month,
    count_applications,
    find_company_role,
    top_companies,
)
from pydantic import ValidationError

# Designed to exercise: case-insensitive matching (gsk / Gsk / GSK),
# date filtering (one row in early Jan, one in late Dec, one mid-year),
# multi-row per company (GSK x3), empty role, mixed months.
SAMPLE_CSV = (
    "From,Subject,Date,Body,Company,Role\n"
    '"a@gsk.com","s1","Mon, 12 Aug 2024 10:00:00 +0000","b","Gsk","Data Scientist"\n'
    '"b@gsk.com","s2","Tue, 5 Mar 2024 09:00:00 +0000","b","Gsk","ML Engineer"\n'
    '"c@gsk.com","s3","Wed, 21 Feb 2024 08:00:00 +0000","b","Gsk",""\n'
    '"d@mediatek.com","s4","Wed, 21 Feb 2024 08:00:00 +0000","b","MediaTek","ML Engineer"\n'
    '"e@ed.ac.uk","s5","Mon, 1 Jan 2024 10:00:00 +0000","b",'
    '"University of Edinburgh","Research Associate"\n'
    '"f@ed.ac.uk","s6","Tue, 31 Dec 2024 23:00:00 +0000","b","University of Edinburgh",""\n'
    '"g@roku.com","s7","Thu, 15 Aug 2025 12:00:00 +0000","b","Roku","Software Engineer"\n'
)


@pytest.fixture
def con(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """A DuckDB connection with the sample data loaded."""
    csv = tmp_path / "sample.csv"
    csv.write_text(SAMPLE_CSV)
    return load_applications_table(csv)


# ────────────────────────── count_applications ──────────────────────────


def test_count_total(con: duckdb.DuckDBPyConnection) -> None:
    assert count_applications(con) == 7


def test_count_by_company_is_case_insensitive(con: duckdb.DuckDBPyConnection) -> None:
    # Sample data has "Gsk" (canonical). Caller passes any case.
    assert count_applications(con, company="Gsk") == 3
    assert count_applications(con, company="GSK") == 3
    assert count_applications(con, company="gsk") == 3


def test_count_by_unknown_company_is_zero(con: duckdb.DuckDBPyConnection) -> None:
    assert count_applications(con, company="Nonexistent Co.") == 0


def test_count_with_date_window(con: duckdb.DuckDBPyConnection) -> None:
    # Only the 5 rows in 2024 (Jan, Feb, Feb, Mar, Aug, Dec) — wait, 6:
    assert count_applications(con, since=date(2024, 1, 1), until=date(2024, 12, 31)) == 6
    # Excluding endpoints
    assert count_applications(con, since=date(2024, 3, 1), until=date(2024, 8, 31)) == 2


def test_count_combines_company_and_date(con: duckdb.DuckDBPyConnection) -> None:
    # GSK in March-only: just msg with date 2024-03-05
    assert (
        count_applications(
            con,
            company="Gsk",
            since=date(2024, 3, 1),
            until=date(2024, 3, 31),
        )
        == 1
    )


def test_count_with_only_since(con: duckdb.DuckDBPyConnection) -> None:
    # 2025-and-later → just the Roku row.
    assert count_applications(con, since=date(2025, 1, 1)) == 1


def test_count_with_only_until(con: duckdb.DuckDBPyConnection) -> None:
    # Up to and including Jan 2024 → just the Edinburgh row.
    assert count_applications(con, until=date(2024, 1, 31)) == 1


# ────────────────────────── top_companies ──────────────────────────


def test_top_companies_orders_by_count_desc(con: duckdb.DuckDBPyConnection) -> None:
    out = top_companies(con, n=5)
    assert out == [
        CompanyCount(company="Gsk", n=3),
        CompanyCount(company="University of Edinburgh", n=2),
        CompanyCount(company="MediaTek", n=1),
        CompanyCount(company="Roku", n=1),
    ]


def test_top_companies_alphabetical_tiebreak(con: duckdb.DuckDBPyConnection) -> None:
    # MediaTek (1) and Roku (1) tie at count=1 — ordered alphabetically.
    out = top_companies(con, n=5)
    one_rows = [r for r in out if r.n == 1]
    assert [r.company for r in one_rows] == ["MediaTek", "Roku"]


def test_top_companies_respects_n(con: duckdb.DuckDBPyConnection) -> None:
    assert len(top_companies(con, n=2)) == 2
    assert len(top_companies(con, n=10)) == 4  # only 4 companies exist
    assert top_companies(con, n=0) == []


def test_top_companies_skips_empty_company(tmp_path: Path) -> None:
    """A row with no Company label shouldn't appear in the ranking."""
    content = (
        "From,Subject,Date,Body,Company,Role\n"
        '"a@x.com","s","Mon, 1 Jan 2024 10:00:00 +0000","b","Acme","Eng"\n'
        '"b@x.com","s","Mon, 1 Jan 2024 10:00:00 +0000","b","",""\n'
    )
    csv = tmp_path / "with_empty.csv"
    csv.write_text(content)
    con = load_applications_table(csv)
    out = top_companies(con, n=10)
    assert out == [CompanyCount(company="Acme", n=1)]


# ────────────────────────── applications_by_month ──────────────────────────


def test_applications_by_month_grouped_and_ordered(con: duckdb.DuckDBPyConnection) -> None:
    out = applications_by_month(con)
    assert out == [
        MonthCount(year=2024, month=1, n=1),
        MonthCount(year=2024, month=2, n=2),
        MonthCount(year=2024, month=3, n=1),
        MonthCount(year=2024, month=8, n=1),
        MonthCount(year=2024, month=12, n=1),
        MonthCount(year=2025, month=8, n=1),
    ]


def test_applications_by_month_skips_null_dates(tmp_path: Path) -> None:
    """Rows with unparseable dates don't contaminate the time-series."""
    content = (
        "From,Subject,Date,Body,Company,Role\n"
        '"a@x.com","s","Mon, 1 Jan 2024 10:00:00 +0000","b","Acme","Eng"\n'
        '"b@x.com","s","not a real date","b","Other","Eng"\n'
    )
    csv = tmp_path / "with_bad.csv"
    csv.write_text(content)
    con = load_applications_table(csv)
    out = applications_by_month(con)
    assert out == [MonthCount(year=2024, month=1, n=1)]


# ────────────────────────── find_company_role ──────────────────────────


def test_find_company_role_returns_all_rows(con: duckdb.DuckDBPyConnection) -> None:
    out = find_company_role(con, "Gsk")
    # 3 GSK rows. doc_ids are msg_000000..msg_000002 (ordered by CSV row).
    assert out == [
        RoleRecord(doc_id="msg_000000", role="Data Scientist"),
        RoleRecord(doc_id="msg_000001", role="ML Engineer"),
        RoleRecord(doc_id="msg_000002", role=""),
    ]


def test_find_company_role_is_case_insensitive(con: duckdb.DuckDBPyConnection) -> None:
    assert len(find_company_role(con, "GSK")) == 3
    assert len(find_company_role(con, "gsk")) == 3


def test_find_company_role_empty_for_unknown(con: duckdb.DuckDBPyConnection) -> None:
    assert find_company_role(con, "Nonexistent") == []


def test_find_company_role_includes_empty_role(con: duckdb.DuckDBPyConnection) -> None:
    """An empty-role row is still a valid citation handle — don't drop it."""
    out = find_company_role(con, "Gsk")
    empty_role_rows = [r for r in out if r.role == ""]
    assert len(empty_role_rows) == 1


# ────────────────────────── return-model invariants ──────────────────────────


def test_return_models_are_frozen() -> None:
    """All three return models are immutable value objects."""
    cc = CompanyCount(company="X", n=1)
    mc = MonthCount(year=2024, month=1, n=1)
    rr = RoleRecord(doc_id="msg_x", role="Eng")
    for obj, field in [(cc, "n"), (mc, "year"), (rr, "role")]:
        with pytest.raises(ValidationError):
            setattr(obj, field, "mutated")


def test_month_validates_range() -> None:
    """``month`` must be 1..12 — Pydantic catches a typo at boundary."""
    with pytest.raises(ValidationError):
        MonthCount(year=2024, month=0, n=1)
    with pytest.raises(ValidationError):
        MonthCount(year=2024, month=13, n=1)
