"""Command-line entry point.

Subcommands:

* ``jai ingest <csv> [<csv> ...]`` — parse, chunk, embed, persist.
* ``jai ask "<question>"`` — embed the query, retrieve, generate, print.
* ``jai eval`` *(Week 2)* — score the retriever against the golden set.

A persistent Chroma collection lives at ``--store`` (default
``./data/chroma``) and survives across runs, so you only ever pay the
embedding cost once.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from job_application_insights import __version__
from job_application_insights.evals.runner import (
    evaluate_path,
    format_report,
    make_dense_retriever,
)
from job_application_insights.generate import (
    PROVIDER_NAMES,
    generate_answer,
    make_llm_client,
)
from job_application_insights.ingest.chunk import chunk_documents
from job_application_insights.ingest.embed import Embedder
from job_application_insights.ingest.parse import load_documents
from job_application_insights.retrieval.vector_store import VectorStore

DEFAULT_STORE_PATH = Path("./data/chroma")
DEFAULT_GOLDEN_PATH = Path("./evals/golden_set.jsonl")
DEFAULT_K = 8


def _ingest(csvs: list[Path], store_path: Path) -> int:
    """Read CSVs → chunk → embed → upsert. Idempotent."""
    print(f"Loading {len(csvs)} CSV file(s)…")
    docs = load_documents([str(p) for p in csvs])
    print(f"  parsed {len(docs):,} documents")

    print("Chunking…")
    chunks = chunk_documents(docs)
    print(f"  produced {len(chunks):,} chunks")

    print("Loading embedder…")
    embedder = Embedder()
    print(f"  model={embedder.model_name}  dim={embedder.dimension}")

    print("Embedding chunks…")
    embeddings = embedder.embed_chunks(chunks, show_progress_bar=True)

    print(f"Upserting to {store_path}…")
    store = VectorStore(store_path)
    store.upsert(chunks, embeddings)
    print(f"Done. Store now holds {store.n_chunks:,} chunks.")
    return 0


def _ask(query: str, store_path: Path, k: int, provider: str) -> int:
    """Embed the query → retrieve top-k → generate cited answer."""
    store = VectorStore(store_path)
    if store.n_chunks == 0:
        print(
            f"Vector store at {store_path} is empty. Run `jai ingest <csv>` first.",
            file=sys.stderr,
        )
        return 2

    embedder = Embedder()
    query_vec = embedder.embed([query])[0]
    results = store.query(query_vec, k=k)

    client = make_llm_client(provider)
    answer = generate_answer(query, results, client)

    print(answer.text)
    print()
    print("─── citations ───")
    for cit in answer.citations:
        print(f"  [{cit.chunk_id}]  sim={cit.score:.3f}  {cit.snippet[:80]}…")
    return 0


def _eval(store_path: Path, golden_path: Path, k: int) -> int:
    """Score the dense-only retriever against the golden set."""
    store = VectorStore(store_path)
    if store.n_chunks == 0:
        print(
            f"Vector store at {store_path} is empty. Run `jai ingest <csv>` first.",
            file=sys.stderr,
        )
        return 2
    if not golden_path.exists():
        print(f"Golden set not found at {golden_path}.", file=sys.stderr)
        return 2

    print(f"Store: {store.n_chunks:,} chunks  |  golden: {golden_path}  |  k={k}")
    embedder = Embedder()
    retriever = make_dense_retriever(store, embedder)
    report = evaluate_path(golden_path, retriever, k=k)
    print()
    print(format_report(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser."""
    parser = argparse.ArgumentParser(
        prog="jai",
        description="Hybrid retrieval over job application data.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the version and exit.",
    )

    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Index one or more email CSV exports.")
    p_ingest.add_argument(
        "csvs",
        nargs="+",
        type=Path,
        help="Path(s) to email CSV file(s).",
    )
    p_ingest.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE_PATH,
        help=f"Where to persist the Chroma index (default {DEFAULT_STORE_PATH}).",
    )

    p_ask = sub.add_parser("ask", help="Ask a natural-language question.")
    p_ask.add_argument("query", help="The question to answer.")
    p_ask.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE_PATH,
        help=f"Chroma index path (default {DEFAULT_STORE_PATH}).",
    )
    p_ask.add_argument(
        "-k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of chunks to retrieve (default {DEFAULT_K}).",
    )
    p_ask.add_argument(
        "--provider",
        choices=PROVIDER_NAMES,
        default="anthropic",
        help=(
            "LLM provider to use. 'anthropic' (default), 'openai', 'gemini', or "
            "'echo' (deterministic test double; no API call). Each non-echo "
            "provider reads its own env var: ANTHROPIC_API_KEY / OPENAI_API_KEY "
            "/ GOOGLE_API_KEY."
        ),
    )

    p_eval = sub.add_parser(
        "eval",
        help="Measure the dense-only retriever against the golden set.",
    )
    p_eval.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE_PATH,
        help=f"Chroma index path (default {DEFAULT_STORE_PATH}).",
    )
    p_eval.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help=f"JSONL golden set path (default {DEFAULT_GOLDEN_PATH}).",
    )
    p_eval.add_argument(
        "-k",
        type=int,
        default=DEFAULT_K,
        help=f"Retrieval cutoff (default {DEFAULT_K}).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.command == "ingest":
        return _ingest(list(args.csvs), args.store)
    if args.command == "ask":
        return _ask(args.query, args.store, args.k, args.provider)
    if args.command == "eval":
        return _eval(args.store, args.golden, args.k)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
