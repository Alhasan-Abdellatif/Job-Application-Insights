"""Typed Python tools over the structured ``applications`` table.

These are the four functions the LLM will eventually call via tool use
(Day 3). Each is a thin, *typed* wrapper around a DuckDB query, returns
a Pydantic model (so the JSON schema the LLM sees is derivable from the
signature), and takes the ``DuckDBPyConnection`` as its first argument
so unit tests and the eventual tool-use factory both stay simple — no
hidden global state.

The four functions cover the question shapes RAG cannot answer:

* :func:`count_applications` — counts, optionally filtered by company /
  date window. Answers *"How many GSK applications in 2024?"*.
* :func:`top_companies` — top-N by ack count. Answers *"Which companies
  did I apply to most?"*.
* :func:`applications_by_month` — monthly time-series. Answers *"In
  which months was my volume highest?"*.
* :func:`find_company_role` — role labels for a given company. Answers
  *"What role did I apply for at MediaTek?"* with citation-ready
  ``doc_id`` pointers back into the chunked corpus.

All company filters are case-insensitive *exact* matches against the
canonicalised company labels from ``Applications.ipynb``. Substring /
prefix matching is intentionally not exposed: the notebook already
canonicalises (so ``"Gsk"`` is the canonical form and the LLM will see
that in :func:`top_companies` output), and substring matches would
collapse distinct entities (``"Workday Gsk"`` ≠ ``"Gsk"``) in ways the
caller wouldn't notice.
"""

from __future__ import annotations

from datetime import date

import duckdb
from pydantic import BaseModel, ConfigDict, Field

from job_application_insights.structured.table import TABLE_NAME


class CompanyCount(BaseModel):
    """One row of a top-N company ranking."""

    model_config = ConfigDict(frozen=True)

    company: str
    n: int = Field(..., ge=0)


class MonthCount(BaseModel):
    """One row of a monthly time-series."""

    model_config = ConfigDict(frozen=True)

    year: int
    month: int = Field(..., ge=1, le=12)
    n: int = Field(..., ge=0)


class RoleRecord(BaseModel):
    """One (doc_id, role) tuple from :func:`find_company_role`.

    ``doc_id`` is the citation handle into the chunked corpus — the
    router can use it to fetch the underlying email chunks for a RAG
    follow-up if the LLM needs more context than just the role label.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    role: str


def count_applications(
    con: duckdb.DuckDBPyConnection,
    *,
    company: str | None = None,
    since: date | None = None,
    until: date | None = None,
) -> int:
    """Count application acks, optionally filtered.

    Parameters
    ----------
    con
        Open DuckDB connection with the ``applications`` table loaded.
    company
        Optional exact (case-insensitive) company label.
    since
        Optional inclusive lower bound on ``date``.
    until
        Optional inclusive upper bound on ``date``.

    Returns
    -------
    The number of matching rows.
    """
    sql = f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE 1=1"
    params: list[object] = []
    if company is not None:
        sql += " AND lower(company) = lower(?)"
        params.append(company)
    # Cast to DATE for comparison so `until=date(2024, 12, 31)` includes
    # rows recorded later that same day (the stored TIMESTAMP keeps the
    # original wall-clock time; a naked `date <= ?` would compare against
    # 2024-12-31 00:00:00 and exclude the 23:00 row).
    if since is not None:
        sql += " AND date::DATE >= ?"
        params.append(since)
    if until is not None:
        sql += " AND date::DATE <= ?"
        params.append(until)
    row = con.execute(sql, params).fetchone()
    assert row is not None  # COUNT always returns a row
    return int(row[0])


def top_companies(
    con: duckdb.DuckDBPyConnection,
    *,
    n: int = 10,
) -> list[CompanyCount]:
    """Top ``n`` companies by ack count, descending.

    Ties are broken alphabetically on the company name so the ranking
    is stable across runs (important for snapshot tests and for the
    LLM seeing a deterministic answer).
    """
    if n <= 0:
        return []
    sql = (
        f"SELECT company, COUNT(*) AS n FROM {TABLE_NAME} "
        f"WHERE company <> '' "
        f"GROUP BY company "
        f"ORDER BY n DESC, company ASC "
        f"LIMIT ?"
    )
    return [CompanyCount(company=row[0], n=int(row[1])) for row in con.execute(sql, [n]).fetchall()]


def applications_by_month(
    con: duckdb.DuckDBPyConnection,
) -> list[MonthCount]:
    """Monthly counts ordered chronologically.

    Rows with ``NULL`` ``date`` are silently skipped. The caller can
    still get a true total via :func:`count_applications`; this
    function specifically answers the *time-series* shape of question.
    """
    sql = (
        f"SELECT EXTRACT(year FROM date)::INT AS y, "
        f"EXTRACT(month FROM date)::INT AS m, "
        f"COUNT(*) AS n "
        f"FROM {TABLE_NAME} "
        f"WHERE date IS NOT NULL "
        f"GROUP BY y, m "
        f"ORDER BY y, m"
    )
    return [
        MonthCount(year=int(row[0]), month=int(row[1]), n=int(row[2]))
        for row in con.execute(sql).fetchall()
    ]


def find_company_role(
    con: duckdb.DuckDBPyConnection,
    company: str,
) -> list[RoleRecord]:
    """Return ``(doc_id, role)`` for every ack matching ``company``.

    Empty-role rows are *included* — the absence of a role label is
    often itself informative (and discarding them would silently drop
    citation handles the router might want).
    """
    sql = (
        f"SELECT doc_id, role FROM {TABLE_NAME} "
        f"WHERE lower(company) = lower(?) "
        f"ORDER BY doc_id"
    )
    return [
        RoleRecord(doc_id=row[0], role=row[1]) for row in con.execute(sql, [company]).fetchall()
    ]
