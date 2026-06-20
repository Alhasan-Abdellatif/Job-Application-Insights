"""Parse raw email CSV exports into typed :class:`Document` objects.

The notebook side of this project already produces a CSV per Gmail Takeout.
This module reads one or more such CSVs and returns a list of validated,
deduplicated :class:`Document` instances ready for the chunking + embedding
pipeline downstream.

Design choices worth knowing:

* **Pydantic v2** for the data model — gives us validation at boundary
  ("garbage in → exception, not silent corruption") plus free JSON
  serialisation if we ever need to checkpoint the parsed corpus.
* **Immutable Documents** (``model_config = ConfigDict(frozen=True)``) — once
  parsed, a document is a value object. Frozen instances are safe to use as
  dict keys and to share across threads without copy.
* **MIME-aware subject decoding** — Gmail wraps long / non-ASCII subjects in
  RFC 2047 ``=?UTF-8?Q?…?=`` blobs. We decode them here so every downstream
  consumer sees plain text.
* **Whitespace normalisation** — collapses ``\\r\\n``-folded headers and
  multi-line bodies into single-spaced text. Avoids surprises in chunking
  and BM25 tokenisation later.
* **One ID per row** assigned at load time (``msg_000123``) so chunks and
  retrieval citations can refer back to the parent document.
"""

from __future__ import annotations

import re
from datetime import datetime
from email.header import decode_header
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """A single normalised email, ready for chunking + embedding.

    Attributes
    ----------
    doc_id
        Stable identifier assigned by the loader (e.g. ``msg_000123``).
        Used downstream as the citation key.
    sender
        The ``From`` header, whitespace-collapsed.
    recipient
        The ``To`` header, whitespace-collapsed.
    subject
        The subject after MIME decoding and whitespace normalisation.
    body
        The plain-text body (HTML tags stripped, whitespace collapsed).
    date
        Parsed timestamp, or ``None`` when the original header could not
        be parsed. Always timezone-aware when present.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., min_length=1)
    sender: str = ""
    recipient: str = ""
    subject: str = ""
    body: str = ""
    date: datetime | None = None

    @property
    def text(self) -> str:
        """Single string used downstream for chunking + embedding.

        Concatenates subject and body. Subject is included because it often
        carries the most concentrated signal ("Thank you for applying to X")
        that we want retrieved when the body is empty.
        """
        return f"{self.subject}\n\n{self.body}".strip()


# ────────────────────────────── helpers ──────────────────────────────


_TIMEZONE_LABEL_RE = re.compile(r"\s*\([^)]*\)\s*$")
_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]+|#\d+);")


def _is_missing(value: Any) -> bool:
    """Return True for None, NaN, whitespace-only strings, or the literal 'nan'.

    The "literal 'nan'" check catches a pandas-specific gotcha: ``str(NaN)``
    yields the three-character string ``"nan"``, and when callers stringify
    a cell value before passing it through, that placeholder leaks all the
    way into chunk text and gets embedded. Treating ``"nan"`` (case-insensitive)
    as missing keeps junk out of the embeddings.
    """
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        if stripped.lower() == "nan":
            return True
    return False


def _normalise(value: Any) -> str:
    """Collapse runs of whitespace and strip; safe for NaN / None."""
    if _is_missing(value):
        return ""
    return _WHITESPACE_RE.sub(" ", str(value)).strip()


def _strip_html(value: str) -> str:
    """Crude HTML-tag and entity removal — fine for our use case.

    Full-fidelity HTML rendering is overkill for retrieval / embedding;
    we only need readable plain text. BeautifulSoup is intentionally
    avoided here to keep the dependency surface small.
    """
    if not value:
        return ""
    value = _HTML_TAG_RE.sub(" ", value)
    value = _HTML_ENTITY_RE.sub(" ", value)
    return value


def decode_mime_subject(value: str) -> str:
    """Decode an RFC 2047 MIME-encoded header (e.g. ``=?UTF-8?Q?…?=``).

    Many ATS systems use this encoding for long or non-ASCII subjects (Arabic,
    Japanese, emoji, etc.). If the value isn't MIME-encoded, it's returned
    unchanged.
    """
    if not value or "=?" not in value:
        return value
    try:
        out = ""
        for chunk, encoding in decode_header(value):
            if isinstance(chunk, bytes):
                out += chunk.decode(encoding or "utf-8", errors="replace")
            else:
                out += chunk
        return out
    except Exception:
        return value


def parse_email_date(value: Any) -> datetime | None:
    """Parse an RFC 2822 / 5322 date header. Returns ``None`` if unparseable."""
    if _is_missing(value):
        return None
    raw = _TIMEZONE_LABEL_RE.sub("", str(value).strip())
    try:
        return parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None


# ────────────────────────────── public API ──────────────────────────────


REQUIRED_COLUMNS: tuple[str, ...] = ("From", "Subject", "Date", "Body")


def load_documents(paths: list[Path | str]) -> list[Document]:
    """Load and deduplicate emails from one or more CSV exports.

    Parameters
    ----------
    paths
        One or more CSV file paths. Each CSV must contain the columns
        ``From``, ``Subject``, ``Date``, ``Body`` (the ``To`` column is
        optional and defaults to empty).

    Returns
    -------
    A list of validated :class:`Document` instances with stable
    ``doc_id`` values.

    Raises
    ------
    FileNotFoundError
        If any path does not exist.
    ValueError
        If a CSV is missing the required columns.
    """
    if not paths:
        return []

    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=list(REQUIRED_COLUMNS), keep="first").reset_index(drop=True)

    docs: list[Document] = []
    for idx, (_pd_idx, row) in enumerate(df.iterrows()):
        subject = _normalise(decode_mime_subject(_normalise(row.get("Subject"))))
        body = _normalise(_strip_html(str(row.get("Body", "") or "")))
        docs.append(
            Document(
                doc_id=f"msg_{idx:06d}",
                sender=_normalise(row.get("From")),
                recipient=_normalise(row.get("To", "")),
                subject=subject,
                body=body,
                date=parse_email_date(row.get("Date")),
            )
        )
    return docs
