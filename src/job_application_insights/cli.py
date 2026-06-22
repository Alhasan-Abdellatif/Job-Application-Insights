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
from job_application_insights.agents.backends import AGENT_PROVIDER_NAMES
from job_application_insights.agents.orchestrator import AgenticAgent
from job_application_insights.agents.router import RouterDecision, classify
from job_application_insights.agents.tool_use import LiveToolUseAgent
from job_application_insights.evals.runner import (
    RetrievalFn,
    evaluate_path,
    format_report,
    make_bm25_retriever,
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
from job_application_insights.retrieval.bm25 import BM25Index
from job_application_insights.retrieval.hybrid import make_hybrid_retriever
from job_application_insights.retrieval.parent_docs import expand_to_parent_documents
from job_application_insights.retrieval.rerank import (
    CrossEncoderReranker,
    make_reranked_retriever,
)
from job_application_insights.retrieval.vector_store import RetrievalResult, VectorStore
from job_application_insights.structured.table import load_applications_table

DEFAULT_STORE_PATH = Path("./data/chroma")
DEFAULT_GOLDEN_PATH = Path("./evals/golden_set.jsonl")
DEFAULT_STRUCTURED_CSV = Path("./data/synthetic/ack_only.csv")
DEFAULT_K = 8
DEFAULT_AGENT_PROVIDER = "gemini"
RETRIEVER_NAMES: tuple[str, ...] = ("dense", "bm25", "hybrid", "rerank")
ASK_MODE_NAMES: tuple[str, ...] = ("rag", "tools", "auto")


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


def _ask(
    query: str,
    store_path: Path,
    k: int,
    provider: str,
    retriever_name: str,
    expand_parents: bool = False,
) -> int:
    """Retrieve top-k via the named retriever, generate a cited answer."""
    store = VectorStore(store_path)
    if store.n_chunks == 0:
        print(
            f"Vector store at {store_path} is empty. Run `jai ingest <csv>` first.",
            file=sys.stderr,
        )
        return 2

    print(f"Building '{retriever_name}' retriever…", file=sys.stderr)
    retriever = _build_retriever(retriever_name, store)
    chunks_by_id = {c.chunk_id: c for c in store.iter_chunks()}

    retrieved_ids = retriever(query, k)
    # The Reranker, BM25, and hybrid retrievers return chunk IDs, not raw
    # scores. We rebuild RetrievalResult objects so generate_answer's
    # signature stays uniform; the placeholder score is rank-derived so
    # citations are still visually ordered. For the dense path users will
    # see the values as smoothly descending in [0, 1] — not literal
    # cosine similarities. If you need exact retriever scores, look at
    # the raw retriever output directly.
    results = [
        RetrievalResult(chunk=chunks_by_id[cid], score=1.0 - i / max(len(retrieved_ids), 1))
        for i, cid in enumerate(retrieved_ids)
        if cid in chunks_by_id
    ]
    if expand_parents:
        print(f"Expanding {len(results)} chunks to parent documents…", file=sys.stderr)
        results = expand_to_parent_documents(results, chunks_by_id)
        print(f"  -> {len(results)} parent documents", file=sys.stderr)

    client = make_llm_client(provider)
    answer = generate_answer(query, results, client)

    print(answer.text)
    print()
    print("─── citations ───")
    for cit in answer.citations:
        print(f"  [{cit.chunk_id}]  rank-score={cit.score:.3f}  {cit.snippet[:80]}…")
    return 0


def _ask_auto(
    query: str,
    store_path: Path,
    structured_csv: Path,
    k: int,
    provider: str,
    retriever_name: str,
    agent_provider: str,
    expand_parents: bool = False,
) -> int:
    """Full agentic loop: router → RAG / structured / both → typed answer.

    Loads BOTH engines up front because the router's decision is only
    known after an LLM call. ``provider`` controls the RAG / compose
    LLM; ``agent_provider`` controls the router + tool-use agent (these
    are decoupled so you can pair a cheap fast model for routing with a
    higher-quality model for compose).
    """
    store = VectorStore(store_path)
    if store.n_chunks == 0:
        print(
            f"Vector store at {store_path} is empty. Run `jai ingest <csv>` first.",
            file=sys.stderr,
        )
        return 2
    if not structured_csv.exists():
        print(
            f"Structured CSV not found at {structured_csv}.",
            file=sys.stderr,
        )
        return 2

    print(
        f"Building '{retriever_name}' retriever + structured table  "
        f"(agent={agent_provider}, rag={provider})…",
        file=sys.stderr,
    )
    retriever = _build_retriever(retriever_name, store)
    chunks_by_id = {c.chunk_id: c for c in store.iter_chunks()}
    con = load_applications_table(structured_csv)

    router_client = make_llm_client(agent_provider)

    def _router(question: str) -> RouterDecision:
        return classify(question, llm_client=router_client)

    agent = AgenticAgent(
        classifier=_router,
        tool_agent=LiveToolUseAgent(con, provider=agent_provider),
        retriever=retriever,
        chunks_by_id=chunks_by_id,
        rag_client=make_llm_client(provider),
        retrieval_k=k,
        expand_parents=expand_parents,
    )
    result = agent.answer(query)

    print(f"─── router decision: {result.decision.mode} ───")
    if result.decision.mode == "both":
        print(f"  structured: {result.decision.structured_question}")
        print(f"  rag:        {result.decision.rag_question}")
    print()
    print(result.text)
    if result.tool_calls:
        print()
        print(f"─── tool calls  ({result.stopped_reason}) ───")
        for tc in result.tool_calls:
            args_str = ", ".join(f"{k_}={v!r}" for k_, v in tc.arguments.items())
            print(f"  {tc.name}({args_str}) → {tc.output}")
    if result.citations:
        print()
        print("─── citations ───")
        for cit in result.citations:
            print(f"  [{cit.chunk_id}]  rank-score={cit.score:.3f}  {cit.snippet[:80]}…")
    return 0


def _ask_tools(query: str, structured_csv: Path, agent_provider: str) -> int:
    """Run a tool-use round-trip over the structured side-table.

    ``agent_provider`` selects the tool-use backend (Gemini / Anthropic /
    OpenAI). The four tools and the loop are provider-agnostic; only
    the SDK encoding differs.
    """
    if not structured_csv.exists():
        print(
            f"Structured CSV not found at {structured_csv}. "
            f"Run `python scripts/filter_acks.py` first to build it.",
            file=sys.stderr,
        )
        return 2

    print(
        f"Loading structured table from {structured_csv}  (agent={agent_provider})…",
        file=sys.stderr,
    )
    con = load_applications_table(structured_csv)
    agent = LiveToolUseAgent(con, provider=agent_provider)
    result = agent.answer(query)

    print(result.text)
    print()
    print(f"─── tool calls  ({result.stopped_reason}) ───")
    for tc in result.tool_calls:
        args_str = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())
        print(f"  {tc.name}({args_str}) → {tc.output}")
    return 0


def _build_retriever(name: str, store: VectorStore) -> RetrievalFn:
    """Construct the named retriever, paying only the cost it needs.

    Dense pays the BGE-small load (~5s + ~250MB RAM). BM25 pays the
    chunk-readback + tokenization (~1s for ~7k chunks). The factory
    isolates each cost so other commands don't pay it unnecessarily.
    """
    if name == "dense":
        embedder = Embedder()
        return make_dense_retriever(store, embedder)
    if name == "bm25":
        chunks = store.iter_chunks()
        index = BM25Index(chunks)
        return make_bm25_retriever(index)
    if name == "hybrid":
        embedder = Embedder()
        dense = make_dense_retriever(store, embedder)
        index = BM25Index(store.iter_chunks())
        bm25 = make_bm25_retriever(index)
        return make_hybrid_retriever([dense, bm25])
    if name == "rerank":
        # Hybrid (dense + BM25 + RRF) as the candidate funnel, then a
        # cross-encoder reranks the top fetch_k. The chunks are read
        # from the store once and reused for both BM25 indexing and the
        # text-lookup the reranker needs.
        embedder = Embedder()
        dense = make_dense_retriever(store, embedder)
        chunks = store.iter_chunks()
        index = BM25Index(chunks)
        bm25 = make_bm25_retriever(index)
        hybrid = make_hybrid_retriever([dense, bm25])
        chunk_text_by_id = {c.chunk_id: c.text for c in chunks}
        reranker = CrossEncoderReranker()
        return make_reranked_retriever(hybrid, reranker, chunk_text_by_id.__getitem__)
    raise ValueError(f"unknown retriever {name!r}; expected one of {RETRIEVER_NAMES}")


def _eval(store_path: Path, golden_path: Path, k: int, retriever_name: str) -> int:
    """Score the named retriever against the golden set."""
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

    print(
        f"Store: {store.n_chunks:,} chunks  |  golden: {golden_path}  |  "
        f"retriever: {retriever_name}  |  k={k}"
    )
    retriever = _build_retriever(retriever_name, store)
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
    p_ask.add_argument(
        "--retriever",
        choices=RETRIEVER_NAMES,
        default="dense",
        help=(
            "Retriever to use. 'dense' (default; Week-1 baseline), 'bm25' "
            "(lexical), 'hybrid' (RRF fusion of both), or 'rerank' "
            "(hybrid + BGE-reranker-base cross-encoder)."
        ),
    )
    p_ask.add_argument(
        "--mode",
        choices=ASK_MODE_NAMES,
        default="rag",
        help=(
            "Answer mode. 'rag' (default) uses retrieve+generate over the "
            "vector store. 'tools' uses the Gemini tool-use agent over the "
            "structured DuckDB table. 'auto' runs the question router and "
            "dispatches to RAG / structured / both per its decision — "
            "loads both engines up front."
        ),
    )
    p_ask.add_argument(
        "--structured-csv",
        type=Path,
        default=DEFAULT_STRUCTURED_CSV,
        help=(f"Path to the ack-only CSV for tools mode (default {DEFAULT_STRUCTURED_CSV})."),
    )
    p_ask.add_argument(
        "--expand-parents",
        action="store_true",
        help=(
            "After retrieval, expand each chunk to its full parent email "
            "before generation. Improves answer quality on long emails "
            "but multiplies LLM input tokens (~3-5x per query). Off by "
            "default. Has no effect in 'tools' mode (no retrieval)."
        ),
    )
    p_ask.add_argument(
        "--agent-provider",
        choices=AGENT_PROVIDER_NAMES,
        default=DEFAULT_AGENT_PROVIDER,
        help=(
            f"Provider for the tool-use agent and the router (default "
            f"{DEFAULT_AGENT_PROVIDER!r}). Decoupled from --provider so "
            f"you can pair a cheap fast model for routing/tools with a "
            f"higher-quality model for RAG/compose. Only used by "
            f"'--mode tools' and '--mode auto'."
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
    p_eval.add_argument(
        "--retriever",
        choices=RETRIEVER_NAMES,
        default="dense",
        help=(
            "Retriever to evaluate. 'dense' (BGE-small + Chroma cosine, "
            "Week 1 baseline), 'bm25' (lexical, in-memory), 'hybrid' "
            "(both, fused with RRF k=60), or 'rerank' (hybrid + "
            "BGE-reranker-base cross-encoder over the top-50 candidates)."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911 — central CLI dispatch
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.command == "ingest":
        return _ingest(list(args.csvs), args.store)
    if args.command == "ask":
        if args.mode == "tools":
            return _ask_tools(args.query, args.structured_csv, args.agent_provider)
        if args.mode == "auto":
            return _ask_auto(
                args.query,
                args.store,
                args.structured_csv,
                args.k,
                args.provider,
                args.retriever,
                args.agent_provider,
                expand_parents=args.expand_parents,
            )
        return _ask(
            args.query,
            args.store,
            args.k,
            args.provider,
            args.retriever,
            expand_parents=args.expand_parents,
        )
    if args.command == "eval":
        return _eval(args.store, args.golden, args.k, args.retriever)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
