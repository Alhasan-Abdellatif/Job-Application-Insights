# Week 4 lesson: from "it works" to "it's deployed" — what we actually built

Week 3 closed with three answer modes (`rag`, `tools`, `auto`) and a CLI that fluently routed counting questions to DuckDB and prose questions to the vector store. It worked great on one machine. **It didn't work on anyone else's.**

Week 4 was about the gap between a working notebook and a public URL — five steps the lesson primer laid out as the "production pyramid":

1. Replace in-process Chroma with a real vector DB (Qdrant).
2. Make the agentic loop legible to others (LangGraph).
3. (Skipped) Add LLM-as-judge evals (Ragas).
4. Put it behind an HTTP API + a UI (FastAPI + Streamlit + Docker).
5. Ship a public URL (Modal).

This is the post-mortem. The companion forward-looking primer is in [week-4-lesson.md](week-4-lesson.md).

---

## The shape of the week

| Day | Commit | Built |
|---|---|---|
| 1 | `42a0ed1` | Chroma → Qdrant migration with factory selection |
| 2 | `be49319` | LangGraph orchestrator as opt-in alternative to direct dispatch |
| 3 | *(skipped)* | Ragas LLM-as-judge evals |
| 4 | `f152489` | FastAPI service + Streamlit UI + Docker Compose stack |
| 5 | *(this commit)* | Modal deployment + synthetic demo data + this writeup |

Plus two follow-up fixes that landed mid-week:

- `20ed2e3` — drop broken Qdrant healthcheck (the official image is distroless; no `curl` to call).
- `15aaf7e` — include `Date` in the `filter_acks` dedup key (templated ACKs collapsed dozens of distinct applications into one row).

---

## Part 1 — Qdrant migration (Day 1)

The Week 1–3 stack ran Chroma in-process. Every CLI invocation paid the index load; every test had to reset a global. Worse: there was no shared store. Two clients couldn't see each other's writes.

The fix was a single new file ([qdrant_store.py](../src/job_application_insights/retrieval/qdrant_store.py)) plus a factory that picks a backend by name:

```python
store = make_vector_store(
    backend="qdrant",                       # or "chroma"
    qdrant_url="http://localhost:6333",
    vector_size=embedder.dimension,
)
```

`QdrantVectorStore` is **shape-compatible** with `VectorStore` — same `upsert`, `query`, `iter_chunks`, `clear`, `n_chunks` — so every caller (retrievers, CLI, tests) is unchanged.

Two design decisions deserve a paragraph each.

**Deterministic UUID5 point IDs.** Qdrant requires point IDs to be int or UUID; our chunk_ids are strings like `msg_000123__c000`. Naively this means `chunk_id → UUID` is a lookup table we have to persist somewhere. The trick:

```python
_CHUNK_ID_NAMESPACE = uuid.UUID("c1a4e1a8-3c2b-4b1f-9e1a-7c2b4b1f9e1a")

def _chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_CHUNK_ID_NAMESPACE, chunk_id))
```

Same chunk_id always maps to the same UUID. No lookup table. Upserts stay idempotent across restarts because the deterministic UUID collides with the existing point. The original chunk_id is preserved in the payload for citation.

**`:memory:` mode for tests.** Qdrant's Python client supports an in-process `:memory:` mode — same API, no Docker required. CI runs sixteen Qdrant tests in under two seconds; no fixtures, no teardown.

---

## Part 2 — LangGraph orchestrator (Day 2)

Week 3's `AgenticAgent` was an if/elif over the router's decision:

```python
if decision.mode == "rag":     return self._handle_rag(...)
elif decision.mode == "tools": return self._handle_tools(...)
elif decision.mode == "both":  return self._handle_both(...)
```

It worked. It also hid the control flow from anyone who hadn't read the file. LangGraph made the flow into a diagram:

```python
graph = StateGraph(AgentState)
graph.add_node("route",    _route_node)
graph.add_node("rag",      _rag_node)
graph.add_node("tools",    _tools_node)
graph.add_node("retrieve", _retrieve_node)
graph.add_node("compose",  _compose_node)

graph.add_edge(START, "route")
graph.add_conditional_edges("route", _branch_after_route, {"rag": "rag", "tools": "tools"})
graph.add_conditional_edges("tools", _branch_after_tools, {"end": END, "retrieve": "retrieve"})
graph.add_edge("rag",      END)
graph.add_edge("retrieve", "compose")
graph.add_edge("compose",  END)
```

Same semantics; you can read the wiring in eight lines.

Three parametrized equivalence tests assert that `AgenticAgent.answer(q) == LangGraphAgent.answer(q)` on identical stubs — the new orchestrator is a *drop-in replacement*, not a behaviour change. Switching backends is a CLI flag (`--orchestrator langgraph`); default stays `direct` so nothing else moves.

The honest trade-off: LangGraph adds 6 MB of deps and one new concept (state graphs). What you get back is wiring you can show a colleague.

---

## Part 3 — FastAPI + Streamlit + Docker Compose (Day 4)

Three new pieces, one consistent topology.

**FastAPI service** ([api/main.py](../src/job_application_insights/api/main.py)) — the same `answer(question)` interface, now over HTTP. Pydantic models at the boundary (`AskRequest`, `AskResponse`, `CitationOut`, `ToolCallOut`); request validation runs before any code in `_serve_rag` / `_serve_tools` / `_serve_auto` sees an unsanitised string. The heavy state — embedder, chunks_by_id, DuckDB connection, retriever cache — loads exactly once in the lifespan and pool across requests.

The `create_app(state=...)` factory makes that testable:

```python
def test_ask_rag_mode_echo_provider(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "...", "provider": "echo"})
    assert resp.status_code == 200
    assert resp.json()["citations"][0]["chunk_id"] == "c1"
```

`client` is built from `create_app(state=stub_state)` — no embedder load, no Qdrant container, no real LLM. Thirteen tests run in 1.2 seconds.

**Streamlit UI** ([streamlit_app.py](../streamlit_app.py)) — pure HTTP client. No imports from `job_application_insights`. The UI process holds zero ML state; it just renders sidebar knobs into a JSON POST and renders the response.

**Docker Compose** ([docker-compose.yml](../docker-compose.yml)) — three services, two ports exposed to the host:

```
qdrant (6333) ──┐
                ├── api (8000) ───── ui (8501)
                │
        named volume: qdrant_storage
```

The two-port separation matters even when both services run on one laptop: the UI and the API can be killed and restarted independently; the embedder load only happens on api restarts. It also matches what you'd do in real prod — one stateless API replica set, one stateless UI, one stateful vector store.

**Two real bugs we fixed in compose:**

1. The Qdrant image is distroless — no shell, no `curl`. The default healthcheck `curl /healthz` always failed; `api` waited forever for `service_healthy`. Replaced with `condition: service_started`.
2. `filter_acks.py` deduped on `(From, Subject, Body)`. Amazon's ACKs ship with empty bodies — 162 distinct emails collapsed to 1 row. Added `Date` to the dedup key. Amazon went from 1 → 40 (matching the notebook).

Both fixes are the kind of integration error you only see once the pieces are running together. The point of Day 4 wasn't writing the FastAPI app — that was easy. It was *getting the three services to actually talk to each other*.

---

## Part 4 — Modal deployment (Day 5)

The plan was Fly.io. Fly killed their free tier. We picked Modal: $30/month free credit, Python-native, multi-service.

Three Modal primitives wrap our docker-compose stack:

```python
@app.cls(image=image, volumes={"/data": qdrant_vol}, secrets=[api_keys_secret])
class APIService:
    @modal.enter()
    def setup(self):
        from job_application_insights.api.main import create_app
        self.fastapi_app = create_app()

    @modal.asgi_app()
    def web(self):
        return self.fastapi_app


@app.function(image=image, secrets=[api_keys_secret])
@modal.web_server(port=8501, startup_timeout=120)
def ui():
    import os, subprocess
    os.environ["JAI_API_URL"] = APIService.web.web_url
    subprocess.Popen(["streamlit", "run", "/app/streamlit_app.py", ...])


@app.function(image=image, volumes={"/data": qdrant_vol}, timeout=600)
def ingest():
    from job_application_insights.cli import ingest_command
    ingest_command([Path("/app/data/synthetic/demo.csv")],
                   store_backend="qdrant", qdrant_path="/data/qdrant")
    qdrant_vol.commit()
```

Three things to call out.

**The Modal Volume solves the cold-restart problem.** `qdrant_vol` persists across redeploys. The first deploy runs `modal run ::ingest` once; every subsequent deploy reuses the populated index — no re-embedding on every code change.

**Embedded Qdrant in the API container.** No separate Qdrant service in the cloud. `QdrantClient(path="/data/qdrant")` is the same client library you'd use against a server — just file-backed. Saves us supervising a second process per container. The HTTP server is still the right tool *locally* (multi-client access, web dashboard); the embedded mode is right *for a single-container demo*.

**Scale to zero.** Both `APIService` and `ui` default to `min_containers=0`. Idle cost = $0. First-request latency = ~15-30s (embedder load + chunks_by_id pull). Subsequent requests = ~200ms. For a portfolio demo with ten hits a day, this is the right shape; for an interview season you flip the flag to `min_containers=1` and pay ~$20/month for always-warm.

**Synthetic data, never real.** The deployed corpus is `data/synthetic/demo.csv` — 100 templated ACKs across 15 fictional companies, built by [`scripts/build_synthetic_demo.py`](../scripts/build_synthetic_demo.py). The real `ack_only.csv` stays gitignored and never leaves the dev machine. The Streamlit UI shows a banner when `JAI_DEMO_MODE=1` (set by the Modal image) so a visitor knows what they're seeing.

---

## What's missing / future work

- **Ragas evals (Day 3).** Skipped intentionally — the retrieval golden set from Week 2 already provides ground-truth recall numbers, and LLM-as-judge metrics weren't load-bearing for the deployment. Worth revisiting if I want to compare answer quality across LLM providers.
- **Persistent UI sessions.** Streamlit holds `st.session_state` in process memory. When a `min_containers=0` UI scales to zero, sessions die. Fine for a demo; would need Redis or signed cookies for anything stateful.
- **Re-ranking and parent-doc expansion in production.** Both are CLI flags; the Modal deploy doesn't toggle them. Cross-encoder reranking adds ~200ms but materially improves answer quality on long emails.
- **Observability.** No traces, no metrics. The Modal dashboard shows invocation counts; nothing about retrieval quality or LLM token spend. A LangSmith integration would be the natural addition.

---

## What this week earned

A public URL on a portfolio. Three commits that show a credible understanding of: vector DB migration with shape-compatible interfaces, agent framework adoption without breaking semantics, multi-service HTTP architecture, container orchestration, and serverless deployment with persistence.

The interview pitch is now:

> Started as a Jupyter notebook over my Gmail. Four weeks later it's a hybrid RAG + structured-query agent (Qdrant + DuckDB), a LangGraph state machine, a FastAPI service, a Streamlit demo, and a public Modal deployment. The code's MIT-licensed at &lt;link&gt;; the deployed demo's at &lt;link&gt;.

Same code does the work in every step. The week-4 commits are the production wrapper — same engine, new packaging.
