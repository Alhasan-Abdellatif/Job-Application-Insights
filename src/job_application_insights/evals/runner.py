"""Evaluation runner — score any retriever against the golden set.

The key abstraction here is :data:`RetrievalFn`: a callable
``(query, k) -> list[str]`` returning the top-K chunk IDs. Every
retriever in this project — dense-only today, BM25/hybrid/reranker
later — conforms to this signature, so the runner doesn't care which
one is on the other end of the protocol.

That decoupling is the whole point of putting this here. The runner
loads the golden set, calls ``retrieve_fn`` once per question,
hands the result list to :mod:`.metrics`, and prints a report. It
knows nothing about embeddings, BM25, Chroma, or LLMs.

Public surface
--------------
* :class:`EvalRow` — per-question metric line (Pydantic, frozen).
* :class:`EvalReport` — full report: rows + aggregate stats.
* :func:`evaluate` — the orchestration call.
* :func:`make_dense_retriever` — convenience factory wrapping
  :class:`VectorStore` + :class:`Embedder` into a ``RetrievalFn``.
  This is what the CLI uses to measure the Week 1 baseline.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from job_application_insights.evals.golden_set import GoldenEntry, load_golden_set
from job_application_insights.evals.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ────────────────────────────── types ──────────────────────────────


RetrievalFn = Callable[[str, int], list[str]]
"""``(query, k) -> top-k chunk_ids``. The seam between this module and
any retriever — dense, BM25, hybrid, reranked — anything that respects
this signature plugs in."""


# ────────────────────────────── data model ──────────────────────────────


class EvalRow(BaseModel):
    """One row of the eval report — metrics for a single question."""

    model_config = ConfigDict(frozen=True)

    question: str
    n_relevant: int = Field(..., ge=1)
    n_retrieved: int = Field(..., ge=0)
    precision_at_k: float = Field(..., ge=0.0, le=1.0)
    recall_at_k: float = Field(..., ge=0.0, le=1.0)
    reciprocal_rank: float = Field(..., ge=0.0, le=1.0)
    ndcg_at_k: float = Field(..., ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Per-question rows + aggregate mean scores."""

    model_config = ConfigDict(frozen=True)

    k: int = Field(..., gt=0)
    rows: list[EvalRow]
    mean_precision: float = Field(..., ge=0.0, le=1.0)
    mean_recall: float = Field(..., ge=0.0, le=1.0)
    mean_mrr: float = Field(..., ge=0.0, le=1.0)
    mean_ndcg: float = Field(..., ge=0.0, le=1.0)


# ────────────────────────────── public API ──────────────────────────────


def evaluate(
    golden_set: list[GoldenEntry],
    retrieve_fn: RetrievalFn,
    *,
    k: int = 8,
) -> EvalReport:
    """Score ``retrieve_fn`` against ``golden_set`` and aggregate.

    Parameters
    ----------
    golden_set
        Ground-truth entries, as loaded by :func:`load_golden_set`.
        An empty list is a programmer error and raises ``ValueError``.
    retrieve_fn
        Any callable with the :data:`RetrievalFn` signature.
    k
        Cutoff for ``precision_at_k``, ``recall_at_k``, and
        ``ndcg_at_k``. ``reciprocal_rank`` is computed over the whole
        returned list (no K cutoff).

    Returns
    -------
    A :class:`EvalReport` with per-question rows and aggregate means.

    Notes
    -----
    ``retrieve_fn`` is called once per entry — no batching. For 25
    questions on a local model this is fine; if it ever isn't,
    parallelising is a one-line change at the call site.
    """
    if not golden_set:
        raise ValueError("golden_set is empty; nothing to evaluate")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    rows: list[EvalRow] = []
    for entry in golden_set:
        retrieved = retrieve_fn(entry.question, k)
        relevant = entry.relevant
        rows.append(
            EvalRow(
                question=entry.question,
                n_relevant=len(relevant),
                n_retrieved=len(retrieved),
                precision_at_k=precision_at_k(retrieved, relevant, k),
                recall_at_k=recall_at_k(retrieved, relevant, k),
                reciprocal_rank=reciprocal_rank(retrieved, relevant),
                ndcg_at_k=ndcg_at_k(retrieved, relevant, k),
                tags=entry.tags,
            )
        )

    return EvalReport(
        k=k,
        rows=rows,
        mean_precision=statistics.fmean(r.precision_at_k for r in rows),
        mean_recall=statistics.fmean(r.recall_at_k for r in rows),
        mean_mrr=statistics.fmean(r.reciprocal_rank for r in rows),
        mean_ndcg=statistics.fmean(r.ndcg_at_k for r in rows),
    )


def evaluate_path(
    golden_path: str | Path,
    retrieve_fn: RetrievalFn,
    *,
    k: int = 8,
) -> EvalReport:
    """Convenience: ``load_golden_set(path)`` + :func:`evaluate`."""
    return evaluate(load_golden_set(golden_path), retrieve_fn, k=k)


# ────────────────────────────── factories ──────────────────────────────


def make_dense_retriever(
    store: object,  # VectorStore — typed as object to keep imports cheap
    embedder: object,  # Embedder — same
) -> RetrievalFn:
    """Wrap a :class:`VectorStore` + :class:`Embedder` as a ``RetrievalFn``.

    This is the *Week 1 baseline*: query → embed → dense cosine search →
    chunk IDs. It's the row in the results table everything else gets
    compared against.

    Typed as ``object`` so importing this module is cheap (no
    ``sentence_transformers`` boot). The runtime call paths still work
    because we duck-type into ``embedder.embed(...)`` and
    ``store.query(...)``.
    """

    def retrieve(query: str, k: int) -> list[str]:
        q_vec: np.ndarray = embedder.embed([query])[0]  # type: ignore[attr-defined]
        results = store.query(q_vec, k=k)  # type: ignore[attr-defined]
        return [r.chunk.chunk_id for r in results]

    return retrieve


# ────────────────────────────── pretty printer ──────────────────────────────


def format_report(report: EvalReport, *, show_rows: bool = True) -> str:
    """Plaintext report — what the CLI prints. Pure function; no I/O."""
    lines: list[str] = []
    lines.append(f"Eval @ k={report.k}  ({len(report.rows)} questions)")
    lines.append("─" * 90)

    if show_rows:
        header = f"{'P@k':>6}  {'R@k':>6}  {'MRR':>6}  {'nDCG':>6}  {'|rel|':>5}  question"
        lines.append(header)
        lines.append("─" * 90)
        for row in report.rows:
            q = row.question if len(row.question) <= 50 else row.question[:47] + "…"
            lines.append(
                f"{row.precision_at_k:>6.3f}  "
                f"{row.recall_at_k:>6.3f}  "
                f"{row.reciprocal_rank:>6.3f}  "
                f"{row.ndcg_at_k:>6.3f}  "
                f"{row.n_relevant:>5}  {q}"
            )
        lines.append("─" * 90)

    lines.append(
        f"{'MEAN':<8}  "
        f"P={report.mean_precision:.3f}  "
        f"R={report.mean_recall:.3f}  "
        f"MRR={report.mean_mrr:.3f}  "
        f"nDCG={report.mean_ndcg:.3f}"
    )
    return "\n".join(lines)
