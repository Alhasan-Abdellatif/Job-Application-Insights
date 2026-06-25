# Job Application Insights

> A production-grade hybrid retrieval system that answers natural-language
> questions over a heterogeneous corpus of job-application emails. Combines
> dense + sparse retrieval, cross-encoder reranking, and an LLM agent that
> routes between semantic search and structured SQL queries.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-400%20passing-brightgreen.svg)]()
[![Type-checked: mypy strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy.readthedocs.io/)
[![Ruff](https://img.shields.io/badge/lint-ruff-yellow.svg)](https://github.com/astral-sh/ruff)

---

## What it does

Ask questions in plain English over an email corpus. The system decides
whether the question needs **retrieval** (find the email about X), a
**structured query** (count rows where Y), or **both composed** (count
applications AND quote the role):

| Question | Engine used | Why |
|---|---|---|
| *"Did I apply to Aurora Robotics?"* | Vector retrieval (RAG) | Factoid — one chunk has the answer |
| *"How many applications in 2025?"* | DuckDB tool call | Aggregation — no chunk contains the count |
| *"How many Granite Robotics apps and what role?"* | Both | Aggregation + entity extraction |

Every answer carries citations — the exact chunks the LLM saw — so you can
trace any claim back to the source email.

---

## Architecture

```
                    ┌─────────────────────┐
   user question ──>│  Router (LLM)       │
                    └──────┬──────────────┘
                           │  decides: rag / tools / both
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌─────────────────────┐   ┌─────────────────────┐
   │ Retrieval pipeline  │   │ Tool-use loop       │
   │ ─────────────────── │   │ ─────────────────── │
   │  dense (Qdrant)     │   │  count_applications │
   │  + BM25 (in-mem)    │   │  top_companies      │
   │  + RRF fusion       │   │  monthly_breakdown  │
   │  + cross-encoder    │   │  find_company_role  │
   │  + parent-doc       │   │       │             │
   │    expansion        │   │       ▼             │
   │       │             │   │  DuckDB (in-proc)   │
   │       ▼             │   └──────────┬──────────┘
   │  retrieved chunks   │              │
   └──────────┬──────────┘              │
              │                         │
              └────────────┬────────────┘
                           ▼
                  ┌─────────────────┐
                  │ Compose (LLM)   │
                  │ cited answer    │
                  └─────────────────┘
```

Multi-orchestrator: the agentic loop above runs through either a hand-written
**direct dispatch** (if/elif on the router decision) or a **LangGraph**
`StateGraph` with named nodes and conditional edges. Two implementations, one
parametrized equivalence test suite — toggle between them with
`--orchestrator {direct,langgraph}`.

---

## Tech stack

| Layer | Technology | Why this one |
|---|---|---|
| **Embeddings** | [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (sentence-transformers) | Strong English retrieval at 384-dim — small enough to run on CPU |
| **Vector store** | [Qdrant](https://qdrant.tech/) (HTTP + embedded modes) | Production-grade ANN with payload filtering; embedded mode for single-container deploys |
| **Sparse retrieval** | [rank-bm25](https://github.com/dorianbrown/rank_bm25) | Catches exact-match queries where dense embeddings hallucinate semantic similarity |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Compares ranks not raw scores — robust to scale mismatch between dense + BM25 |
| **Reranker** | [BGE-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | Cross-encoder over the top-K candidates; lifts answer relevance on long emails |
| **Structured store** | [DuckDB](https://duckdb.org/) (in-process OLAP) | SQL over a CSV with zero infra; perfect for the counts/aggregations dimension |
| **Agent framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) | `StateGraph` with explicit nodes + conditional edges; the loop is legible as a diagram |
| **LLM providers** | Anthropic Claude, OpenAI, Google Gemini, plus an `echo` test double | Each provider is one adapter behind a `LLMClient` Protocol — swap providers per-call |
| **Tool calling** | Provider-native (Anthropic, OpenAI, Gemini SDKs) | Four typed Python tools surfaced to the model with JSON-schema args |
| **API layer** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn | Pydantic boundary types, async lifespan for heavy state, OpenAPI docs at `/docs` |
| **UI** | [Streamlit](https://streamlit.io/) (HTTP client only) | Two-port architecture: UI never imports the agent, only POSTs to the API |
| **Orchestration (local)** | [Docker Compose](https://docs.docker.com/compose/) | Three-service stack (qdrant + api + ui) with named volume + healthchecks |
| **Orchestration (cloud)** | [Modal](https://modal.com/) | Serverless multi-service with persistent `modal.Volume` + scale-to-zero |
| **Eval** | Custom golden-set harness (Recall@K, multi-answer-group support) | Measures retrieval quality on a curated 11-question set |
| **Lang** | Python 3.11, full type hints, Pydantic v2 | |
| **Quality** | ruff + ruff-format + mypy strict + pre-commit + 400 pytest tests | |

---

## Live demo

🌐 *Public demo URL coming soon (deploying on Modal)* — runs on synthetic data,
defaults to the `echo` LLM provider (no API keys required to try the
citation/retrieval flow).

---

## Quick start

### Run locally with Docker Compose (recommended)

Three services come up in one command — no Python install needed beyond Docker.

```bash
git clone <repo-url>
cd job-application-insights
docker compose up
```

Then:
- Streamlit UI: <http://localhost:8501>
- FastAPI OpenAPI explorer: <http://localhost:8000/docs>
- Qdrant dashboard: <http://localhost:6333/dashboard>

On first run, populate Qdrant with synthetic data:

```bash
docker compose exec api uv run jai ingest /app/data/synthetic/demo.csv \
    --store-backend qdrant --qdrant-url http://qdrant:6333
```

### Run locally without Docker (uv-based)

```bash
uv sync --extra dev
uv run jai ingest data/synthetic/demo.csv
uv run jai ask "How many applications did I send in 2025?" --mode auto
```

### Deploy to Modal (cloud)

```bash
uv sync --extra deploy
uv run modal token new                                  # one-time auth
uv run modal secret create jai-api-keys \              # optional LLM keys
    ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GOOGLE_API_KEY=...
uv run modal run modal_app.py::ingest                  # populate Qdrant volume
uv run modal deploy modal_app.py                       # ship api + ui
```

---

## CLI

```bash
# Ingest CSVs into the vector store (idempotent — re-runs upsert)
jai ingest <csv> [<csv> ...] [--store-backend chroma|qdrant]

# Three answer modes
jai ask "Did I apply to X?"                  --mode rag    # vector retrieval
jai ask "How many applications in 2024?"     --mode tools  # DuckDB tool calls
jai ask "How many X and what role?"          --mode auto   # LLM routes + composes

# Pick your retriever
jai ask "..." --retriever {dense|bm25|hybrid|rerank}

# Pick your orchestrator
jai ask "..." --mode auto --orchestrator {direct|langgraph}

# Pick your LLM provider
jai ask "..." --provider {anthropic|openai|gemini|echo}

# Evaluate retrievers against the golden set
jai eval --retriever rerank -k 8
```

---

## Project structure

```
src/job_application_insights/
├── ingest/         CSV parsing, MIME decoding, chunking, embedding
├── retrieval/      Chroma + Qdrant stores, BM25, hybrid RRF fusion, reranker
├── structured/     DuckDB table + four typed tools (count, top-N, etc.)
├── agents/
│   ├── backends.py        Multi-provider tool-use loop (Anthropic/OpenAI/Gemini)
│   ├── tool_use.py        Agent that calls tools until satisfied
│   ├── router.py          LLM-as-classifier — picks rag/tools/both
│   ├── orchestrator.py    Direct (if/elif) dispatch
│   └── langgraph_orchestrator.py    StateGraph alternative
├── evals/          Golden-set runner + Recall@K with answer-group support
├── api/main.py     FastAPI service (lifespan-loaded state, three handlers)
├── cli.py          jai command — ingest / ask / eval
└── generate.py     LLM client adapters + RAG compose

tests/              400+ pytest tests (smoke → integration → equivalence)
modal_app.py        Modal serverless deployment (api + ui + ingest)
docker-compose.yml  Local three-service orchestration
Dockerfile          Single image, CMD overridden per service in compose
streamlit_app.py    UI — HTTP client only, no agent imports
scripts/
├── build_synthetic_demo.py     Deterministic 100-row demo CSV
├── filter_acks.py               (private) labels real Gmail exports
└── curate_golden_set.py         Builds the eval golden set

evals/golden_set.jsonl           Curated 11-question retrieval eval
```

---

## Design highlights

### Shape-compatible vector store interfaces

`VectorStore` (Chroma) and `QdrantVectorStore` share `upsert`, `query`,
`iter_chunks`, `clear`, and `n_chunks` with identical signatures. A factory
picks one at runtime:

```python
store = make_vector_store(
    backend="qdrant",
    qdrant_url="http://localhost:6333",
    # or:
    # qdrant_path="./qdrant_data",   # embedded file mode
    vector_size=embedder.dimension,
)
```

Every caller (retrievers, CLI, API, tests) works against either backend.
Migration was a single new file, zero changes to consumers.

### Deterministic point IDs across backends

Qdrant requires UUID/int point IDs; our chunk_ids are strings like
`msg_000123__c000`. A stable UUID5 namespace gives every chunk_id the same
UUID forever — no lookup table, no migration step, upserts stay idempotent:

```python
def _chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_CHUNK_ID_NAMESPACE, chunk_id))
```

### Two orchestrators, one equivalence test suite

Three `pytest.mark.parametrize`'d tests assert
`AgenticAgent.answer(q) == LangGraphAgent.answer(q)` on identical stubs.
Adding LangGraph wasn't a behaviour change — it was a re-spelling of the
same control flow as a diagram you can read.

### Typed FastAPI boundary

Every public API endpoint takes a Pydantic `BaseModel` and returns one.
Validation happens before any handler code runs; OpenAPI docs are generated
automatically; tests use `TestClient` with `create_app(state=...)` to skip
the 5-second embedder load entirely.

### Modal deployment without rewriting the app

The Modal config wraps **the same FastAPI app and the same Streamlit script**
— no fork, no rewrite. `@app.cls + @modal.enter` hoists the heavy state
load out of the FastAPI lifespan into Modal's container pool; `@modal.web_server`
launches Streamlit as a subprocess. A persistent `modal.Volume` solves the
cold-restart re-ingest problem.

---

## Testing & quality

```bash
uv run pytest                  # 400+ tests, ~30s
uv run ruff check .
uv run ruff format --check .
uv run mypy src/               # strict mode
uv run pre-commit run --all    # gate that runs all of the above
```

Coverage hovers around 80–85% line, 70% branch. Integration tests run
Qdrant in `:memory:` mode (no Docker required for CI).

---

## License

MIT — see [LICENSE](LICENSE).
