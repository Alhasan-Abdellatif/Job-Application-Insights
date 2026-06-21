"""Structured query layer over the ack corpus.

The Week-3 *dual* to the RAG retriever: where RAG answers "what does this
email say?", the structured engine answers "how many, which ones, by
when, in what order?". Built on DuckDB for fast analytical queries; the
tool layer (Day 2) wraps typed Python functions over this table that the
LLM can call directly.

Public surface:

* :func:`load_applications_table` — build the in-memory ``applications``
  table from an ack CSV.
* :data:`TABLE_NAME` — the canonical name of that table.
"""

from job_application_insights.structured.table import (
    TABLE_NAME,
    load_applications_table,
)
from job_application_insights.structured.tools import (
    CompanyCount,
    MonthCount,
    RoleRecord,
    applications_by_month,
    count_applications,
    find_company_role,
    top_companies,
)

__all__ = [
    "TABLE_NAME",
    "CompanyCount",
    "MonthCount",
    "RoleRecord",
    "applications_by_month",
    "count_applications",
    "find_company_role",
    "load_applications_table",
    "top_companies",
]
