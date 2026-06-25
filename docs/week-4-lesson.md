# Week 4 lesson: from "it works" to "it's deployed"

Week 3 closed with an agentic system that answers count questions, RAG questions, and compound questions across three LLM providers — but the whole thing runs on your laptop, in your terminal. *That's not a system anyone else can use.* Week 4 turns it into a thing with a URL.

The shift in one sentence:

> **Weeks 1–3 built the engine. Week 4 builds the car around it — and parks it on a real road.**

This week is less about new RAG ideas and more about the *production stack* that surrounds RAG in real jobs. The vocabulary you earn this week is what hiring managers look for: **Qdrant, LangGraph, Ragas, FastAPI, Docker, deployed demo URL.** Every component already exists in your project conceptually; this week swaps in the industry-standard form of each so you can talk about both *the principle* and *the tool*.

This lesson has the same six-part shape as the others.

---

## Part 1 — The production gap

When you say *"I built a RAG system"* and a hiring manager hears *"I built a RAG system,"* you're usually picturing different things.

| What you might mean | What a senior engineer hears |
|---|---|
| A script that calls an LLM with retrieved context | A *web service* that other people can hit |
| Chroma running inside Python | A vector DB running as a separate process with backups |
| Hard-coded if/else routing | A traceable agent graph with state and checkpoints |
| "Recall@8 = 0.82 on 11 questions" | An *answer-correctness* metric like Ragas faithfulness |
| `uv run jai ask "..."` | `curl https://your-demo.fly.dev/ask -d ...` |
| Tests | Tests *and* observability *and* CI |

Week 4 closes that gap. We don't rebuild — we *re-skin* each piece in the form a production engineer would expect. You get the vocabulary AND the underlying understanding, because the architectural seams you built in earlier weeks make every swap a small change.

### The five-step pyramid we'll climb

```
                  ┌──────────────────────┐
                  │  Live demo URL       │  Day 5
                  └──────────┬───────────┘
                             │
                  ┌──────────────────────┐
                  │  FastAPI + Streamlit │  Day 4
                  │  + Docker compose    │
                  └──────────┬───────────┘
                             │
                  ┌──────────────────────┐
                  │  Ragas + LLM-as-judge│  Day 3
                  └──────────┬───────────┘
                             │
                  ┌──────────────────────┐
                  │  LangGraph orchestr. │  Day 2
                  └──────────┬───────────┘
                             │
                  ┌──────────────────────┐
                  │  Qdrant vector DB    │  Day 1
                  └──────────────────────┘
```

Each layer turns one of your DIY pieces into the named industry tool. Below the pyramid sits everything you've already built — your retriever, your tools, your router, your orchestrator. None of that goes away.

**Next**: the first swap — moving from "vector store inside Python" to "vector DB as a separate service."

---

## Part 2 — Vector databases in production (Qdrant)

### What we have today

Chroma. It runs *inside* your Python process. Your script starts, Chroma loads from disk, you query, the script exits. Simple, fast, perfect for a single-user prototype.

### Why that doesn't work in production

Three things break the moment more than one person uses the system:

1. **Concurrent users.** Two users querying at the same time means two Python processes each loading their own copy of Chroma into RAM. With ~11k chunks that's fine; with 11 million it's not.
2. **Updates while running.** A new email arrives → you want to add it without restarting the service. Chroma in-process can do this; Chroma also corrupts under concurrent writes if you're unlucky.
3. **Backup, monitoring, scaling.** Chroma is "files on disk." Real ops teams want a service with health checks, metrics, snapshots, replicas.

Same shape of problem as: **SQLite vs Postgres.** SQLite is a file your program reads directly. Postgres is a separate server your program talks to over a socket. SQLite is perfect for a single-user app; Postgres is what production runs on.

### What Qdrant brings

Qdrant is a vector database written in Rust that runs as a service. Your Python code talks to it over HTTP. It handles concurrent reads, atomic writes, snapshots, replication, monitoring — all the things a "real" database does.

Plus it has features Chroma lacks or does poorly:

- **Native metadata filtering.** *"Find the top-K similar emails sent in 2025."* In Qdrant, that's a filter expression sent with the query; Qdrant uses it to skip non-matching vectors *during* the ANN search (not after). Chroma post-filters after-the-fact, which can hurt recall.
- **Hybrid search built in.** Qdrant can store both dense (BGE) and sparse (BM25-style) vectors in the same collection, doing RRF fusion server-side. Today we run BM25 in Python in your process.
- **Production-grade ops.** Health endpoint, metrics, snapshot restore, GRPC + REST.

### What the migration looks like

The Week 1 `VectorStore` class was deliberately a thin wrapper. The Day 1 of Week 4 is mostly:

```python
# Old: chromadb in-process
client = chromadb.PersistentClient(path="./data/chroma")

# New: Qdrant over HTTP
client = QdrantClient(host="localhost", port=6333)
```

Plus a one-liner `docker compose up qdrant` to bring up the server. Everything downstream — the retriever, the agent — keeps working because they only see the `VectorStore` interface.

### What it earns you on a CV

> *"Migrated the production vector store from Chroma (in-process) to Qdrant (Docker-deployed, Rust-backed), enabling concurrent reads, server-side metadata filters, and native sparse+dense hybrid search."*

**Next**: the vector DB stores chunks. The *orchestration* of the LLM around those chunks is the next layer to industrialize.

---

## Part 3 — Agent frameworks (LangGraph)

### What we have today

Your `AgenticAgent` (Day 4 of Week 3) is an *if/elif/else* dispatcher:

```python
if decision.mode == "rag":      return self._answer_rag(...)
if decision.mode == "structured": return self._answer_structured(...)
return self._answer_both(...)
```

That's correct and works. But:

- The *shape* of the agent — what nodes exist, what edges connect them — is invisible. You'd have to read the code to know.
- Adding a new behavior (e.g., *"after RAG, if the answer is 'I don't know,' fall back to tools"*) means more `if`s.
- There's no built-in checkpoint, no streaming, no easy retry.

### What an "agent framework" actually is

Imagine the agent as a *directed graph*. Each **node** is a function (`route`, `run_rag`, `run_tools`, `compose`). Each **edge** is a conditional transition (`if mode == 'rag' go to run_rag`). The "state" is a dict that flows through the graph — each node reads from it, modifies it, passes it to the next.

That's it. The framework gives you:

- A way to *declare* this graph explicitly.
- *Engine code* that walks the graph, calls each node, manages state.
- Free additions: streaming (yield results as nodes complete), checkpointing (save state at each node so you can resume after a crash), tracing (every node call logged for debugging).

### Why LangGraph specifically

LangGraph (Anthropic 2024, now part of LangChain) has converged on being **the** way people express agentic flows in 2025-2026. It's small (~few hundred LOC of API surface), composable with the rest of LangChain, and — most importantly — *every job listing in this space mentions it.*

The mental model:

```
StateGraph(AgentState)
    │
    ├── add_node("route",      router_fn)
    ├── add_node("rag",        rag_fn)
    ├── add_node("tools",      tools_fn)
    ├── add_node("both",       both_fn)
    ├── add_node("compose",    compose_fn)
    │
    ├── set_entry_point("route")
    ├── add_conditional_edges("route", pick_branch, {
    │       "rag":        "rag",
    │       "tools":      "tools",
    │       "both":       "both",
    │   })
    ├── add_edge("rag",     END)
    ├── add_edge("tools",   END)
    └── add_edge("both",    "compose")
```

That's the entire Week 3 agentic loop, in LangGraph form. The semantics don't change; the *expressed shape* does.

### The honest trade-off

LangGraph adds maybe 30-50 lines of glue for our specific simple flow. It would feel like over-engineering — *if* this were the only thing we were building. But:

- **It's a teach-yourself-the-framework exercise.** Every senior LLM engineer is expected to fluently write LangGraph state machines at interview.
- **The graph view IS the right mental model.** Once you've expressed the agent as a graph, you start *thinking* in graphs, which scales as agents get more complex.
- **The seams pay off again.** The classifier function, the tool agent, the retriever — all already exist as injected dependencies. They become LangGraph nodes with one line of glue each.

### What it earns you on a CV

> *"Implemented the multi-engine agent as a LangGraph `StateGraph` — router node, conditional branching to RAG / structured / hybrid engines, with a composition step for the hybrid path."*

**Next**: the engine is in place. We've never measured *whether the final answers are correct.*

---

## Part 4 — Evaluation that measures answers, not chunks (Ragas)

### What we have today

The Week 2 harness measures **retrieval**: Recall@K, Precision@K, MRR, nDCG. *"Of the chunks I marked relevant, what fraction did the retriever surface?"*

What it never asked: **Was the LLM's final answer correct?**

Those are different questions. You could have R@8 = 1.0 (the retriever found every relevant chunk) and still have the LLM produce a wrong sentence — because it ignored the context, hallucinated, or misunderstood.

### What Ragas measures

Ragas is the dominant RAG evaluation library in production. It provides four metrics that hiring managers ask about by name:

| Metric | Question it answers |
|---|---|
| **Faithfulness** | Is every claim in the answer actually supported by the retrieved context? (Catches hallucination.) |
| **Answer relevancy** | Does the answer actually address the question? (Catches off-topic responses.) |
| **Context precision** | Of the chunks retrieved, what fraction were *necessary* to answer? (Tightens context.) |
| **Context recall** | Of the *necessary* information, what fraction is in the retrieved context? (Resembles our Week 2 metric.) |

The first two need *another* LLM to score them — that's the "LLM-as-judge" pattern. The judge is a separate, ideally larger model (e.g. Claude Sonnet judging Claude Haiku's answers). It reads the question, the context, and the answer, and returns a score.

### The mental model of LLM-as-judge

Sounds dodgy — *"using an LLM to judge an LLM"* — but it's standard practice now because:

- Humans don't scale (you can't review 1000 answers per release).
- Reference-based metrics (BLEU, ROUGE) are bad at semantic correctness.
- LLMs are surprisingly consistent judges at temperature 0 with a careful rubric.

The trick is in the rubric. *"Score the answer's faithfulness from 1-5"* is bad. *"For each factual claim in the answer, check whether the retrieved context contains evidence; the score is the fraction supported"* is the kind of structured prompt that gives reliable judgments.

Ragas builds the rubric for you. You give it `(question, answer, context, reference_answer)` and it returns the four numbers.

### How we'll wire it

The Week 2 `GoldenEntry` gets one new field:

```python
class GoldenEntry(BaseModel):
    question: str
    answer_groups: list[list[str]] | None
    reference_answer: str | None     # ← new for Ragas
```

A new `jai eval-answers` subcommand runs each golden question through the agent, captures the answer, hands `(question, answer, context, reference_answer)` to Ragas, and reports the four metrics. Same shape as the Week 2 retrieval eval — just measuring a different layer.

### What it earns you on a CV

> *"Built an end-to-end RAG evaluation pipeline using Ragas — faithfulness, answer relevancy, context precision/recall — with LLM-as-judge on a hand-curated golden set, scored against the agentic answer."*

**Next**: the agent works and we can measure it. Time to make it accessible.

---

## Part 5 — Wrapping it as a service (FastAPI + Streamlit + Docker)

### The three layers of "shipping it"

```
       USER
        │
        ▼
   ┌─────────┐
   │Streamlit│   browser-facing demo UI
   └────┬────┘
        │  HTTP
        ▼
   ┌─────────┐
   │ FastAPI │   API surface, typed in/out
   └────┬────┘
        │
        ▼
   ┌─────────┐
   │ Agent + │   the system you built in weeks 1-3
   │  Qdrant │
   └─────────┘
```

Each layer is small. Each layer is industry standard.

### FastAPI

A Python web framework. Three reasons it's the default for Python ML services:

1. **Typed in, typed out.** Request bodies are Pydantic models. Response bodies are Pydantic models. *Same boundary pattern as Week 1's `Document`* — outside is hostile JSON, inside is typed Python.
2. **Async-friendly.** LLM calls block for seconds; FastAPI uses async so one slow call doesn't block other users.
3. **Free OpenAPI docs.** Visit `/docs` in a browser and you see a clickable API explorer auto-generated from your Pydantic models. Hand this to anyone and they can immediately use the service.

The whole `/ask` endpoint is maybe 30 lines:

```python
@app.post("/ask")
def ask(req: AskRequest) -> AskResponse:
    answer = agent.answer(req.question, mode=req.mode, ...)
    return AskResponse(text=answer.text, citations=answer.citations, ...)
```

### Streamlit

Streamlit is the *easiest* way to put a UI on a Python program. A few lines of code → a clickable web app with text inputs, dropdowns, and result panels. It's NOT a frontend framework — it's a "let me demo my Python thing" tool.

Concretely: three widgets (question text input, mode dropdown, agent-provider dropdown) and a results panel. About 50 lines. Hits your FastAPI endpoint to get the answer.

This is what you screenshot for your CV / README.

### Docker

Docker bundles your application + its dependencies + system libraries into one *image* that runs the same on your laptop, your colleague's laptop, and the cloud. *"Works on my machine"* stops being a thing.

`docker compose` is the orchestrator that runs multiple containers together. Our `docker-compose.yml` will spin up:

- `qdrant` — the vector DB
- `api` — the FastAPI service
- `ui` — the Streamlit demo

One command — `docker compose up` — brings the whole stack online. The same file is used in dev, staging, and prod.

### The mental model

Each layer adds zero RAG capability. They add **operability** — the ability for other humans to actually use what you built. That's the whole game in production: a 0.82 R@8 model that doesn't ship is less valuable than a 0.70 model that runs on a public URL.

### What it earns you on a CV

> *"Packaged the agent + Qdrant + Streamlit demo as a Docker-Compose stack with a FastAPI service exposing `/ask`, with OpenAPI documentation, typed Pydantic request/response models, and async LLM calls."*

**Next**: the stack runs locally. The last step is putting it online.

---

## Part 6 — Shipping (deployment + observability)

### Deployment options, ranked by friction

| Platform | What it is | Free tier | When to pick |
|---|---|---|---|
| **Modal** | Python-native serverless. Decorate your function, push, get a URL. | $30 credit | The fastest path for ML services. Strong recommendation. |
| **Fly.io** | Docker-native, regional, edge-friendly. | Generous free tier | If you already have a `docker-compose.yml`. Maps cleanly to "deploy this stack." |
| **Railway** | Heroku-style git-push deployment. | Limited free hours | Simplest if you just want a URL fast. |
| **HuggingFace Spaces** | Streamlit/Gradio-specific. | Free, public | Demos only — not suitable if you want a private prod version too. |
| AWS / GCP | The "real" cloud. | Pay-as-you-go | If the role you're targeting requires it specifically; otherwise overkill for a portfolio piece. |

For your CV the message is the same regardless: *"deployed to a public URL"*. Pick the path of least resistance. **Fly.io** is the recommendation: it understands `docker-compose.yml` directly, has a free tier, and lets you keep your private real-data version locally.

### The "synthetic data" rule

When you deploy publicly, do NOT use your real email data. Two reasons:

- Privacy — the demo URL is public, anyone can see retrieved chunks.
- LLM provider TOS — sending personal data through a cloud LLM may violate Gmail's terms.

Build a small *synthetic ack-only.csv* (maybe 200 made-up application acks across plausibly-named companies) and ship the public demo against that. Keep the real-data version private — Docker compose, your laptop, no internet.

### Observability (optional Day 6)

Once you have users — even one — you want to see *what they're asking* and *how the system is doing*. Two ways to get this:

- **LangSmith** (paid SaaS from LangChain) — the industry standard. Generous free tier for solo devs.
- **Langfuse** (open source, self-hosted via Docker) — same idea, free, you own the data.

Both give you a dashboard showing: every LLM call, its latency, token count, cost, and trace through the agent graph. Three CV bullets' worth of vocabulary for an afternoon of plumbing.

### What it earns you on a CV

> *"Deployed the agent stack to Fly.io behind a public URL, with synthetic ack-only data for the demo. Observability via Langfuse traces (latency p95, token counts, error rates) per LLM call."*

---

## Putting it all together — the Week-4 deliverable

By the end of Friday:

- A live URL: `https://your-demo.fly.dev/`. Click "How many GSK applications in 2025?" → see the agent run.
- A `docker-compose.yml` that lets anyone clone the repo and `docker compose up` to get the whole stack.
- A `docs/week-4-results.md` post-mortem.
- A CV that names: **Qdrant, LangGraph, Ragas, FastAPI, Streamlit, Docker, Fly.io, Langfuse**.
- An interview answer to *"walk me through your production RAG stack"* that takes 60 seconds and doesn't sound rehearsed because you actually built every layer.

---

## What an interviewer will probe

- **"Why Qdrant and not Pinecone?"** Pinecone is managed (no Docker), Qdrant is self-hosted Rust-backed. Pinecone is easier to start, Qdrant is cheaper at scale and more flexible. For a portfolio project where you want to show you understand the deployment story, Qdrant wins.
- **"Did you actually need LangGraph for a three-mode agent?"** No — the if/else version works. But the graph form is the canonical expression, it's what scales when the agent gets a fifth and sixth mode, and it's the framework hiring managers expect you to know.
- **"How do you trust LLM-as-judge?"** You measure inter-rater reliability against a small human-labeled subset (~30 examples). If GPT-4-as-judge agrees with a human 90%+ of the time on the easy cases, you use it for the rest. Ragas's published benchmarks include this validation.
- **"How would you scale this to 10M emails?"** Three answers: shard Qdrant by user, push embedding generation to a background queue (Celery / Modal), and route the hottest queries through a cache layer (Redis). The agent shape doesn't change; the storage and the dispatch do.

---

## New vocabulary you should now own

- **Vector database vs in-process index** (Qdrant vs Chroma)
- **HNSW + payload filtering** (Qdrant's native filter-during-search vs Chroma's filter-after)
- **State graph / `StateGraph`** (LangGraph's central concept)
- **Conditional edges** (router-style transitions in LangGraph)
- **Node / edge / state** (the three primitives of any agent graph)
- **Faithfulness, answer relevancy, context precision/recall** (the four Ragas metrics by name)
- **LLM-as-judge** (separate-model evaluation pattern)
- **Reference answer** (the new golden-set field for answer-correctness eval)
- **FastAPI + Pydantic boundary** (typed request/response models with automatic OpenAPI)
- **Async LLM call** (why FastAPI uses async — long-running LLM calls don't block other users)
- **Streamlit demo / `st.text_input` / `st.write`** (the few primitives you need)
- **`docker compose up`** (the production "run it all" command)
- **OpenAPI / Swagger** (the auto-generated API spec from FastAPI)
- **Snapshot / replica / health endpoint** (Qdrant ops vocabulary)
- **Inter-rater reliability** (the trust-but-verify check on LLM-as-judge)
- **Synthetic data version vs private real-data version** (the public/private split)
- **Trace / span / latency p95** (observability vocabulary you'll hear)

---

## Conceptual sanity check

Try answering these before we start coding:

1. Why is "Chroma works fine on my laptop" not enough for production?
2. What's the difference between *filter-during-ANN-search* and *filter-after*? Why does it matter?
3. What does a LangGraph `StateGraph` express that an if/else chain doesn't?
4. Why is **faithfulness** different from **context recall**? Give a one-question example where one is high and the other is low.
5. Why does FastAPI use async endpoints for LLM calls? What breaks if it doesn't?
6. Why do we deploy with synthetic data, not the real ack corpus?
7. Inter-rater reliability between two LLM judges is 70%. Would you trust the metric they produce? What if it's 95%?
8. You add a new tool to the agent. Which layers change? Which don't?

---

## What we're explicitly NOT using (and why)

| Tool | Why skip |
|---|---|
| **LangChain core** | LangGraph already gives you the LangChain ecosystem vocabulary. LangChain proper is mostly chains and prompt templates you don't need. |
| **LlamaIndex** | Replaces hand-built work with a black box. The `RouterQueryEngine` etc. are *exactly* what we built in Week 3 — adopting it would obscure your understanding. Worth 2-hour exploration later, not Week 4. |
| **vLLM / Ollama** | Local LLM hosting. Adds infra without much production value at this scale; the cloud APIs are fine. |
| **Knowledge graphs / structured indices** | Too much scope. Stick to the linear stack. |
| **Real Kubernetes** | Overkill for a portfolio piece. Fly.io / Modal handle the deployment without it. |
| **LangSmith (paid tier)** | Excellent product. If you have budget, use it; otherwise self-host Langfuse for the same capability. |

These will come up in interviews. *"Have you used LlamaIndex?"* — the answer is *"I evaluated it; we built the equivalent patterns by hand to maintain understanding. The trade-off was X."* That's a better answer than *"yes"* with no opinion.

---

## Where this leaves you at end of Week 4

You can answer this interview question in one breath:

> *"Walk me through the production RAG stack you've built end-to-end."*
>
> "Ingestion is a typed Pydantic pipeline. Storage is Qdrant running in Docker — HNSW with payload filters and sparse+dense hybrid search. Retrieval is hybrid — BGE-small dense + BM25 + RRF fusion — with optional cross-encoder reranking. The agent is a LangGraph state machine that routes between RAG, DuckDB-backed structured queries, and a compose step for hybrid questions, with multi-provider tool use across Anthropic, OpenAI, and Gemini. Evaluation uses Ragas with LLM-as-judge — faithfulness, answer relevancy, context precision/recall — on a hand-curated golden set. The service is FastAPI behind a Streamlit demo, all in Docker Compose, deployed to Fly.io with a public URL on synthetic data."

Every clause in that paragraph names a real tool or concept. None of them is buzzword-only — you wrote the underlying logic in Weeks 1-3 and re-skinned it in Week 4. *Three or four CV bullet points, one interview elevator pitch, and a real demo URL to share.*

That's the deliverable. Numbers are temporary. The stack vocabulary is forever.
