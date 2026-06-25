"""Filter the raw inbox CSVs down to *only ack emails*.

Uses the output of ``Applications.ipynb`` (``applications_unique.csv``) as
a (Subject, From) whitelist. The result is a clean homogeneous corpus
of application acknowledgments — no interview invites, no rejections,
no "apply now to..." LinkedIn suggestions — plus the Company/Role
labels the notebook already computed.

Output: ``data/synthetic/ack_only.csv`` (gitignored — personal email
content) with the parser's expected schema (``From``, ``Subject``,
``Date``, ``Body``) plus two extra columns (``Company``, ``Role``)
that the parser will ignore but the golden-set builder can use as
ground-truth entity labels.

Run::

    uv run python scripts/filter_acks.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Where the raw Gmail CSV exports live on disk. Set ``JAI_GMAIL_DIR`` to point
# this somewhere else without editing the script.
GMAIL_DIR = Path(os.environ.get("JAI_GMAIL_DIR", "./data/raw/gmail"))

# inbox_mail.csv covers the multi-year window; smaller snapshots are
# fresher incremental exports. Together they span the full inbox
# Applications.ipynb labelled.
INBOX_CSVS = [
    "inbox_mail.csv",
    "May_inbox_6June.csv",
    "10_Jun.csv",
    "8_Jun.csv",
    "may_inbox.csv",
]
APPLICATIONS_CSV = "applications_unique.csv"
OUT_PATH = Path("./data/synthetic/ack_only.csv")


def main() -> None:
    print("Loading raw inboxes…")
    raw = pd.concat(
        [pd.read_csv(GMAIL_DIR / csv) for csv in INBOX_CSVS],
        ignore_index=True,
    )
    # Include Date in the dedup key — many templated ACKs (Amazon, Workday-
    # style) ship with an empty/image-only Body, so dropping Date collapses
    # dozens of distinct applications into one row.
    raw = raw.drop_duplicates(subset=["From", "Subject", "Date", "Body"], keep="first").reset_index(
        drop=True
    )
    print(f"  {len(raw):,} unique (From, Subject, Date, Body) emails across {len(INBOX_CSVS)} CSVs")

    print("Loading applications_unique.csv…")
    apps = pd.read_csv(GMAIL_DIR / APPLICATIONS_CSV)
    apps_keys = apps[["Subject", "From", "Company", "Role"]].drop_duplicates(
        subset=["Subject", "From"]
    )
    print(f"  {len(apps):,} ack records → {len(apps_keys):,} unique (Subject, From) keys")

    print("Joining…")
    joined = raw.merge(apps_keys, on=["Subject", "From"], how="inner")
    print(f"  {len(joined):,} ack emails matched")

    # Keep only the columns the parser cares about + the labels
    out_cols = ["From", "Subject", "Date", "Body", "Company", "Role"]
    out = joined[out_cols].copy()
    out["To"] = ""  # parser doesn't use it but accepts it

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes, {len(out):,} rows)")

    # Quick summary stats
    print()
    print(f"Unique Company labels: {out['Company'].nunique():,}")
    print(f"Roles with non-empty value: {out['Role'].notna().sum():,}")
    print()
    print("Top companies by ack count:")
    for company, count in out["Company"].value_counts().head(10).items():
        print(f"  {count:>3}  {company}")


if __name__ == "__main__":
    main()
