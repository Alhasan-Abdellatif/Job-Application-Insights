"""DuckDB-backed structured table over the ack-only corpus.

The Day-1 foundation of the Week-3 agentic layer. Loads ``ack_only.csv``
(produced by ``scripts/filter_acks.py``) into an in-memory DuckDB table
called ``applications``. Day 2's tool functions wrap typed Python
callables over this table so the LLM can answer count / aggregation /
top-N questions through tool use.

The hard invariant this module preserves is the *doc_id mapping*:

    row N in the structured table  ⟷  msg_{N:06d} in the chunked corpus

We achieve that by mirroring the parser exactly: same input columns,
same dedup keys, same row-index → ``doc_id`` formatting. Anything that
breaks this invariant breaks the future router's ability to cross-walk
between a structured hit and a RAG citation.

Schema::

    doc_id     VARCHAR   -- foreign key into the vector store ('msg_000123')
    from_addr  VARCHAR   -- sender, whitespace-collapsed
    subject    VARCHAR   -- MIME-decoded subject
    date       TIMESTAMP -- timezone-aware; NULL if unparseable
    company    VARCHAR   -- entity label from Applications.ipynb
    role       VARCHAR   -- role label from Applications.ipynb ("" if missing)
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from job_application_insights.ingest.parse import (
    REQUIRED_COLUMNS,
    _is_missing,
    decode_mime_subject,
    parse_email_date,
)

TABLE_NAME = "applications"
EXPECTED_EXTRA_COLUMNS: tuple[str, ...] = ("Company", "Role")


def _naive_utc(dt: datetime | None) -> datetime | None:
    """Convert tz-aware datetime to naive UTC; pass through None and naive inputs.

    DuckDB's ``TIMESTAMP`` type is timezone-naive. Storing tz-aware values
    would force ``TIMESTAMPTZ``, which requires ``pytz`` at query time —
    we avoid both by normalising to UTC and dropping the tz marker. The
    wall-clock values still answer every date filter we care about.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(UTC).replace(tzinfo=None)


def _clean_string(value: Any) -> str:
    """Map missing values (None, NaN, the literal 'nan') to empty string.

    Mirrors :func:`job_application_insights.ingest.parse._normalise` for
    the missing-value half of its job. We don't collapse whitespace here
    because the structured columns are short identifiers (sender,
    company, role) — whitespace doesn't accumulate the way it does in
    free-text bodies.
    """
    if _is_missing(value):
        return ""
    return str(value).strip()


def load_applications_table(
    csv_path: Path | str,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> duckdb.DuckDBPyConnection:
    """Load an ack CSV into a DuckDB ``applications`` table.

    Parameters
    ----------
    csv_path
        Path to ``ack_only.csv`` — or any CSV with the parser's required
        columns plus ``Company`` and ``Role``.
    connection
        Optional existing DuckDB connection. Defaults to a fresh
        in-memory one. If passed, any existing ``applications`` table is
        dropped and rebuilt — re-loading is idempotent.

    Returns
    -------
    The DuckDB connection. The caller closes when done (or lets it
    garbage-collect in tests).

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS + EXPECTED_EXTRA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    # Dedup on the *raw* CSV columns, same key the parser uses. This is
    # the load-bearing line — change it and doc_ids drift out of sync
    # with the chunked corpus.
    df = df.drop_duplicates(subset=list(REQUIRED_COLUMNS), keep="first").reset_index(drop=True)

    structured_df = pd.DataFrame(
        {
            "doc_id": [f"msg_{idx:06d}" for idx in range(len(df))],
            "from_addr": df["From"].apply(_clean_string),
            "subject": df["Subject"].apply(_clean_string).apply(decode_mime_subject),
            # Reuse the parser's RFC-2822 handler. Pandas' to_datetime
            # silently fails on email headers like "Fri, 1 May 2026
            # 16:30:08 +0000 (UTC)" — we found this on the smoke test
            # against the real ack_only.csv (2,122 / 2,173 dates NULL).
            # parse_email_date handles the trailing "(UTC)" comment, the
            # short day-of-month form, and unparseable rows (→ None).
            # We then strip the tz to keep DuckDB's TIMESTAMP type naive
            # (avoiding a pytz dependency).
            "date": [_naive_utc(parse_email_date(d)) for d in df["Date"]],
            "company": df["Company"].apply(_clean_string),
            "role": df["Role"].apply(_clean_string),
        }
    )

    con = connection if connection is not None else duckdb.connect(":memory:")
    con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    con.register("__structured_df", structured_df)
    con.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM __structured_df")
    con.unregister("__structured_df")
    return con
