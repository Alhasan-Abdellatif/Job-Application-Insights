"""Tests for :mod:`job_application_insights.ingest.parse`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from job_application_insights.ingest.parse import (
    REQUIRED_COLUMNS,
    Document,
    decode_mime_subject,
    load_documents,
    parse_email_date,
)
from pydantic import ValidationError

# ───── helpers ─────


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ───── decode_mime_subject ─────


def test_decode_mime_subject_plain_passes_through():
    assert decode_mime_subject("Thank you for applying") == "Thank you for applying"


def test_decode_mime_subject_q_encoded():
    encoded = "=?UTF-8?Q?Thank_you_for_applying_to_Acme?="
    assert "Thank you" in decode_mime_subject(encoded)
    assert "Acme" in decode_mime_subject(encoded)


def test_decode_mime_subject_arabic():
    # Real-world ALESAYI HOLDING example we hit in the notebook
    raw = "=?UTF-8?Q?Alhasan,_your_application_was_sent_t?= =?UTF-8?Q?o_ALESAYI_HOLDING?="
    decoded = decode_mime_subject(raw)
    assert "ALESAYI HOLDING" in decoded
    assert "Alhasan" in decoded


def test_decode_mime_subject_empty():
    assert decode_mime_subject("") == ""
    assert decode_mime_subject("no mime here") == "no mime here"


# ───── parse_email_date ─────


def test_parse_email_date_with_tz_label():
    dt = parse_email_date("Mon, 10 Jun 2026 06:05:19 -0400 (EDT)")
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 6
    assert dt.day == 10
    assert dt.tzinfo is not None


def test_parse_email_date_basic():
    dt = parse_email_date("Fri, 1 May 2026 16:30:08 +0000")
    assert dt is not None
    assert dt.year == 2026


def test_parse_email_date_invalid_returns_none():
    assert parse_email_date(None) is None
    assert parse_email_date("") is None
    assert parse_email_date("not a date at all") is None
    assert parse_email_date(float("nan")) is None


def test_is_missing_treats_literal_nan_string_as_missing():
    """str(NaN) yields 'nan' — that's pandas-junk, not real content."""
    from job_application_insights.ingest.parse import _is_missing

    assert _is_missing("nan") is True
    assert _is_missing("NaN") is True
    assert _is_missing("  nan  ") is True
    assert _is_missing("NAN") is True
    # Sanity: an actual word that happens to contain 'nan' is not missing
    assert _is_missing("nancy") is False
    assert _is_missing("banana") is False


# ───── load_documents ─────


def test_load_documents_simple(tmp_path: Path):
    csv = _write_csv(
        tmp_path / "inbox.csv",
        [
            {
                "From": "ats@example.com",
                "To": "me@me.com",
                "Subject": "Hi",
                "Date": "Mon, 10 Jun 2026 12:00:00 +0000",
                "Body": "Hello world",
            },
        ],
    )
    docs = load_documents([csv])
    assert len(docs) == 1
    d = docs[0]
    assert d.doc_id == "msg_000000"
    assert d.sender == "ats@example.com"
    assert d.subject == "Hi"
    assert d.body == "Hello world"
    assert d.date is not None and d.date.year == 2026


def test_load_documents_decodes_mime_in_subject(tmp_path: Path):
    csv = _write_csv(
        tmp_path / "inbox.csv",
        [
            {
                "From": "x@y.com",
                "To": "",
                "Subject": "=?UTF-8?Q?Thanks_for_applying_to_Acme?=",
                "Date": "",
                "Body": "",
            }
        ],
    )
    docs = load_documents([csv])
    assert "Acme" in docs[0].subject
    assert "=?" not in docs[0].subject


def test_load_documents_normalises_folded_subject(tmp_path: Path):
    """Long Gmail subjects get a literal CRLF + space inserted — should collapse."""
    folded = "Alhasan, your application was sent to Ellison Institute of\r\n Technology Oxford"
    csv = _write_csv(
        tmp_path / "inbox.csv",
        [
            {
                "From": "x@y.com",
                "To": "",
                "Subject": folded,
                "Date": "",
                "Body": "",
            }
        ],
    )
    docs = load_documents([csv])
    assert "Ellison Institute of Technology Oxford" in docs[0].subject
    assert "\r" not in docs[0].subject
    assert "\n" not in docs[0].subject


def test_load_documents_strips_html(tmp_path: Path):
    csv = _write_csv(
        tmp_path / "inbox.csv",
        [
            {
                "From": "x@y.com",
                "To": "",
                "Subject": "Hi",
                "Date": "",
                "Body": "<html><body>Thank you&nbsp;for&nbsp;applying!</body></html>",
            }
        ],
    )
    docs = load_documents([csv])
    assert "Thank you" in docs[0].body
    assert "<html>" not in docs[0].body
    assert "&nbsp;" not in docs[0].body


def test_load_documents_dedups_exact_duplicates(tmp_path: Path):
    dup_row = {
        "From": "a@b.c",
        "To": "me",
        "Subject": "X",
        "Date": "Mon, 10 Jun 2026 12:00:00 +0000",
        "Body": "body",
    }
    csv = _write_csv(tmp_path / "dups.csv", [dup_row, dup_row, dup_row])
    docs = load_documents([csv])
    assert len(docs) == 1


def test_load_documents_concatenates_files(tmp_path: Path):
    csv1 = _write_csv(
        tmp_path / "a.csv",
        [
            {"From": "a@x.com", "To": "", "Subject": "A", "Date": "", "Body": "alpha"},
        ],
    )
    csv2 = _write_csv(
        tmp_path / "b.csv",
        [
            {"From": "b@x.com", "To": "", "Subject": "B", "Date": "", "Body": "beta"},
        ],
    )
    docs = load_documents([csv1, csv2])
    assert len(docs) == 2
    subjects = {d.subject for d in docs}
    assert subjects == {"A", "B"}


def test_load_documents_missing_column_raises(tmp_path: Path):
    csv = _write_csv(tmp_path / "broken.csv", [{"only": "one column"}])
    with pytest.raises(ValueError, match="missing required columns"):
        load_documents([csv])


def test_load_documents_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_documents([tmp_path / "nope.csv"])


def test_load_documents_empty_paths():
    assert load_documents([]) == []


# ───── Document model ─────


def test_document_is_frozen():
    doc = Document(doc_id="x", sender="s", subject="hi", body="hello")
    with pytest.raises(ValidationError):
        doc.subject = "boom"  # type: ignore[misc]


def test_document_text_combines_subject_and_body():
    doc = Document(doc_id="x", sender="s", subject="The subject", body="The body")
    assert "The subject" in doc.text
    assert "The body" in doc.text


def test_required_columns_constant_matches_loader():
    # Smoke: REQUIRED_COLUMNS is the public contract — break it and CI catches it.
    assert "From" in REQUIRED_COLUMNS
    assert "Subject" in REQUIRED_COLUMNS
    assert "Date" in REQUIRED_COLUMNS
    assert "Body" in REQUIRED_COLUMNS
