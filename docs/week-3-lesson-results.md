# Week 3 lesson: when *not* to retrieve — what we actually built

Week 2 closed with a hybrid retriever at *Recall@8 = 0.82* and a question that still embarrassed it:

> *"How many GSK applications did I send in 2024?"*

No tweak to the retriever could ever answer that. The answer is a count over many rows; it doesn't live in any single chunk. Week 3 was about *teaching the system when not to retrieve* — by building a second engine (structured queries) and an LLM that picks between them.

This document is the post-mortem of Days 1–5. The companion forward-looking primer is in [week-3-lesson.md](week-3-lesson.md); read that first if you want the *why* before the *what we did*.

---

## The shape of the week

Five days, four commits, one consistent thread: *the structured engine is the dual of RAG, and the router is the cheap LLM call that decides which one runs*.

| Day | Commit | Built |
|---|---|---|
| 1 | `3c6d811` | DuckDB side-table over `ack_only.csv` |
| 2 | `84f55d2` | Four typed Python tools (count, top-N, monthly, role) |
| 3 | `36b6752` | Gemini tool-use agent (the LLM finally meets the tools) |
| 4 | `951bace` | Router + orchestrator (the agentic loop closes) |
| 5 | *(this commit)* | Optional parent-doc retrieval + this writeup |

By the end the CLI has three answer modes:

```bash
jai ask "Did I apply to ALESAYI?"                  --mode rag        # Week 2 default
jai ask "How many GSK applications in 2025?"       --mode tools      # Day 3
jai ask "How many GSK applications and what role?" --mode auto       # Day 4
```

And one optional flag that improves answer quality at the cost of tokens:

```bash
jai ask "What was the GSK role?" --mode rag --expand-parents
```

Default behaviour is *unchanged* from Week 2. The agentic path is opt-in.

---

## Part 1 — The structured side-table (Day 1)

The structured engine isn't a parallel ETL pipeline. It's a thin SQL view over the same `ack_only.csv` the RAG pipeline already ingests. One module, one function, one in-memory DuckDB table:

```python
from job_application_insights.structured import load_applications_table
con = load_applications_table("data/synthetic/ack_only.csv")

con.execute("SELECT COUNT(*) FROM applications WHERE company = 'Gsk'").fetchone()
# (10,)
```

Schema is six columns: `doc_id`, `from_addr`, `subject`, `date`, `company`, `role`. The load-bearing invariant — *the structured row N has the same `doc_id` as `msg_{N:06d}` in the chunked corpus* — is enforced by mirroring the parser exactly (same dedup keys, same row-index → ID formula). One test asserts the invariant directly; break it and you'd silently mis-join the two engines.

**The bug that justified the smoke test.** Synthetic unit-test data all parsed cleanly. The real `ack_only.csv` had RFC-2822 dates with trailing `"(UTC)"` markers that pandas' `to_datetime` silently NULLed — **2,122 / 2,173 dates lost**. Fix was one line (reuse the parser's `parse_email_date`). Lesson: **synthetic tests prove behaviour against synthetic data**; the smoke test is what proves it against the real thing.

---

## Part 2 — Four typed tools (Day 2)

Before the LLM joined, the tool layer was just four ordinary Python functions over the DuckDB connection:

| Function | Question shape |
|---|---|
| `count_applications(con, *, company=, since=, until=)` | "How many GSK acks in 2024?" |
| `top_companies(con, *, n=10)` | "Top 5 companies I applied to" |
| `applications_by_month(con)` | "Apps per month" |
| `find_company_role(con, company)` | "What role at MediaTek?" |

Each returns either a scalar or a list of frozen Pydantic models (`CompanyCount`, `MonthCount`, `RoleRecord`). Tested as pure functions with deterministic SQL; no LLM yet.

Three design choices worth repeating:

1. **`con` is always the first arg.** No globals, no module-level connection. Day 3 closes over `con` in a factory; tests pass a fixture. *Hidden state inside a closure is fine; hidden state inside a function body is not.*
2. **Case-insensitive *exact* match, not substring.** The notebook canonicalises company labels (`"Gsk"` is the canonical form), so substring matching would silently collapse distinct entities (`"Workday Gsk"` ≠ `"Gsk"`).
3. **Date filters compare at DATE granularity.** First version compared a Python `date` directly to a TIMESTAMP column; DuckDB extends `date(2024, 12, 31)` to `00:00:00`, excluding any later wall-clock time. Fix was `WHERE date::DATE <= ?`. Type-mismatch bugs that the compiler can't see; the test caught it.

The real-corpus smoke surfaced two pieces of data context that will matter later in this lesson:

- **GSK has 10 acks, all in 2025–2026.** Asking *"How many GSK in 2024?"* honestly returns 0. The agentic system needs to learn to say *"none"*, not hedge.
- **MediaTek role labels are all empty.** The notebook never extracted a role for those rows. So *"What role at MediaTek?"* via the structured engine returns 10 rows of `(doc_id, "")`. That's not a bug; it's a real case where the structured engine *can't fully answer* and the right move is to fall back to RAG. Day 4's router needed to be aware of this.

---

## Part 3 — Wiring the tools to Gemini (Day 3)

Day 3 introduced the small tool-use loop:

1. Send the user's question + four `FunctionDeclaration`s to Gemini.
2. If Gemini returns a `function_call`, dispatch to the matching Python function, append the result as a `function_response`.
3. If Gemini returns text, that's the final answer.
4. Bail at `max_steps=4` so a misbehaving model can't loop forever.

The whole thing fits in one ~250-line module. Layered for testability:

- **`dispatch(con, name, args) -> result`** — pure: tool-name + args → Python result. Tested as a normal function.
- **`serialise(result) -> dict`** — pure: result → uniform JSON shape (`{"value": ...}` for scalars, `{"rows": [...]}` for lists).
- **`build_function_declarations()`** — hand-written tool specs (not auto-derived; descriptions are prompt engineering).
- **`run_tool_use_loop(question, *, con, generate_fn, max_steps)`** — the loop with a stubbable `generate_fn`.
- **`GeminiToolUseAgent`** — live wrapper, ~20 lines.

22 tests use `SimpleNamespace` stubs of Gemini responses — no `google.genai` import in test files. Same trick as Week 2's `StubReranker`: structural typing means the stub doesn't even need to import the real types.

**One LLM-quirk we caught:** Gemini sometimes emits `{"company": ""}` for an *omitted* optional argument instead of leaving the key out. Passing that to SQL would `WHERE lower(company) = ''` and match nothing. One-line fix in `dispatch`: treat empty-string optionals as `None`. Same shape of bug as Week 2's pandas-NaN-to-string leak: the LLM-to-SQL boundary needs its own normalisation, same as the CSV-to-parser boundary did in Week 1.

The CLI gained a new mode:

```bash
jai ask "How many GSK applications in 2025?" --mode tools
# (router skipped — the user told us this is structured)
```

---

## Part 4 — Router + orchestrator (Day 4)

The structured engine and the RAG engine each handle their own question shapes well. Day 4 added the small piece of intelligence that *decides which*.

### The router

One Gemini call, structured JSON output, three buckets:

```python
class RouterDecision(BaseModel):
    mode: Literal["rag", "structured", "both"]
    structured_question: str | None = None  # populated only for "both"
    rag_question: str | None = None         # populated only for "both"
```

The system prompt is heavy on examples (~50 lines of "use *rag* for X, use *structured* for Y, use *both* for Z"). The `response_mime_type="application/json"` config flag biases Gemini toward parseable JSON output. Even so, the conservative fallback is the safety net: any failure to parse a valid `RouterDecision` becomes `mode="rag"` — RAG always produces *some* answer (even *"I don't know"*); a wrong tool route can silently return zero.

A `model_post_init` validator enforces consistency at the boundary:
- `mode="both"` requires both sub-questions.
- `mode="rag"` or `"structured"` must *not* carry sub-questions.

If the LLM ever emits inconsistent JSON we want a `ValidationError` at the seam, not silent corruption five method calls deeper.

### The orchestrator

`AgenticAgent` holds the four collaborators — classifier, tool agent, retriever, RAG LLM client — and dispatches:

```
                      question
                         │
                         ▼
                   classifier (1 LLM call)
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
      "rag"        "structured"          "both"
        │                │                │
   retrieve+gen      tool-use         both engines
   (Week 2 path)     (Day 3 path)      + compose
        │                │                │
        └────────────────┴────────────────┘
                         ▼
                  AgenticAnswer
   (text + decision + tool_calls + citations + stopped_reason)
```

The orchestrator itself has zero LLM knowledge. Production passes Gemini-backed clients; the 16 orchestrator tests pass stubs that record calls and return canned responses. **30 router + orchestrator tests run in ~0.5 seconds with no network**.

A "both" question is two engine runs plus one final *compose* LLM call that writes the answer using both pieces of evidence. Worst case: 4 LLM calls (router + tool-use + retrieve+nothing + compose). Best case (pure RAG): 2.

---

## Part 5 — Optional parent-doc retrieval (Day 5)

Week 2's retriever surfaces *chunks* — 512-token slices of an email. For long emails the LLM sees one slice and may answer with partial context.

Parent-document retrieval is the simple fix: for every retrieved chunk, **expand to its full parent email** at generation time. The retriever still finds the same chunks (recall metrics are unaffected); the LLM just sees more of each.

```python
from job_application_insights.retrieval.parent_docs import expand_to_parent_documents
parents = expand_to_parent_documents(retrieved_results, chunks_by_id)
```

Each parent result uses `doc_id` as its citation handle (so citations point at the email, not the slice), and multiple retrieved chunks from the same email collapse into one parent.

**Opt-in, off by default.** Each expansion typically multiplies the LLM input tokens by 3–5x. Enabled via the new flag, in both retrieve-using modes:

```bash
jai ask "What did the GSK ack actually say?"       --mode rag  --expand-parents
jai ask "How many GSK and what role?"              --mode auto --expand-parents
```

The `tools` mode is unaffected (no retrieval happens there).

---

## What surprised us

1. **The router prompt is mostly examples.** The Gemini-side decision quality went up substantially when each category got three concrete examples ("for questions like..."). Abstract rules ("use structured for aggregations") produced more wrong routes than typed examples did. The router prompt now reads more like documentation than instruction.

2. **Empty data is information.** The MediaTek role labels are all blank. The naïve move would be to filter empty roles out of `find_company_role`'s output. The right move is to keep them — because an empty role is *evidence the structured engine can't fully answer*, which is exactly what Day 4's router needs to know when it considers falling back to RAG.

3. **The function signature is the LLM's contract.** Tool-use in Gemini lets us declare functions; the LLM reads the *descriptions* to decide when to call them. The Python signature is enough to *call* the function; it's not enough to know *when to call it*. Each `description` field in `build_function_declarations` is small prompt engineering — examples in plain English of the questions that tool answers. Empty descriptions would compile but route badly.

4. **Conservative fallback is structurally important, not just defensive.** "If the router emits bad JSON, route to RAG" sounds like an afterthought until you realise it's the only thing preventing the system from silently returning `0` for *"how many?"* when the parser drops a parenthesis. **The cheapest LLM call (the router) is also the most consequential — pick its failure mode deliberately.**

5. **Inject collaborators, don't construct them.** Every test of the orchestrator runs without ever importing `google.genai`. The orchestrator's constructor takes its four dependencies as arguments; the CLI factory is the only place that builds the real Gemini-backed clients. Three-line tests instead of three-page mock setups.

---

## What this looks like in numbers

Week 3 deliberately *didn't* add a routing-accuracy or structured-correctness eval. The Week 2 set-cover-recall harness still works for the `rag` path; the new modes would need a new golden-set extension and new metrics, which we chose to defer.

So the headline numbers are unchanged from Week 2:

```
                k=8                          k=20
            P     R     MRR   nDCG       P     R     MRR   nDCG
Dense    0.284  0.371  0.377  0.371     0.200  0.469  0.385  0.399
BM25     0.284  0.636  0.301  0.384     0.195  0.636  0.301  0.384
Hybrid   0.318  0.818  0.558  0.621     0.268  0.825  0.565  0.625
Rerank   0.500  0.727  0.655  0.672     0.327  0.727  0.655  0.672
```

What changed this week is *the class of questions the system can answer*, not the precision/recall on any one of them. *"How many GSK applications in 2025?"* simply has no recall number in the Week 2 framework — there is no chunk to retrieve — and now there's a working answer.

Adding measurement for the new modes is Week 4's first job:

- **Routing accuracy** — for each golden-set question, did the router pick the right mode?
- **Structured correctness** — for `mode="structured"` questions, did the right tool run with the right arguments and return the right value?
- **Answer correctness (LLM-as-judge)** — the enterprise standard, layered on top of the per-mode metrics.

---

## The CV bullet

> *Extended a hybrid-retrieval RAG system with a typed tool-use agent (DuckDB structured table + four Pydantic-typed Python tools surfaced via Gemini function calling) and an LLM-based question router that classifies each query into retrieval / structured / hybrid and decomposes compound questions. Closes the "count and aggregation" failure class that pure RAG cannot answer. Optional parent-document retrieval at generation time for long-email questions.*

**What an interviewer will probe:**

- *Why function calling and not text-to-SQL?* — Smaller surface, no injection risk, the function signature is the contract. The right level for this scale; text-to-SQL is the answer when the query language is the product.
- *How does the router fail?* — Three modes (misclassify / bad decomposition / malformed JSON), with a conservative fallback to RAG. RAG is the safe default because it produces *some* answer; structured can silently produce zero. The cost of a wrong classification is bounded; not silent corruption.
- *How do you evaluate this?* — Honestly, we haven't yet. The Week 2 set-cover recall still applies to the RAG path. Three new metrics (routing accuracy, structured correctness, answer correctness via LLM-as-judge) are the natural next step and Week 4's first task.
- *Why is parent-document retrieval an opt-in flag?* — Token cost. Expanding each retrieved chunk to its full parent multiplies LLM input by 3–5x per query. Justified for long emails where the chunked context is genuinely lossy; wasteful for short factoid questions.
- *Is the router call worth its cost?* — At ~$0.0005 per question on Gemini Flash, ~200ms latency, the routing decision pays for itself the first time it sends a count question to the structured engine instead of having RAG hallucinate a number.

---

## What we deferred

- **LLM-as-judge for answer correctness** — explicitly out-of-scope for Week 3 as requested. The enterprise default metric for RAG quality.
- **Golden-set extension with count / hybrid questions** — needs new entries plus the routing/structured-correctness metrics above.
- **HyDE / query rewriting** — the remaining "naive RAG technique we haven't tried." Marginal on this corpus given hybrid retrieval already does well; useful when query and corpus vocabulary differ sharply.
- **Configuration sweep** — chunk size × embedder × retriever × reranker, automated. The seams exist (every component is a factory); a 50-line script would automate it.
- **Production wrapper** — FastAPI service + Streamlit demo + Docker + a public-facing URL.

---

## The seams that paid off

Three lines that — across all of Week 3 — never had to change:

```python
RetrievalFn   = Callable[[str, int], list[str]]    # any retriever (Week 2)
ToolUseAgent  = Protocol                           # any tool-use loop  (Day 3)
LLMClient     = Protocol                           # any provider       (Week 1)
```

The `AgenticAgent` in Day 4 takes *one of each* and composes them. The router in Day 4 takes a `Callable[[contents, config], response]`. The parent-doc expansion in Day 5 takes a `list[RetrievalResult]` and a `dict[str, Chunk]`. Every test is one line of `_make_classifier(...)`, `_StubToolAgent(...)`, `_make_retriever(...)`, `_StubLLMClient(...)`.

This is the same lesson Week 2 ended on — *type-driven seams are how a codebase stays clean across multiple rounds of "swap in a better version."* Week 3 swapped in three new components and never touched the old ones.

That's the deliverable. The retriever still finds chunks. The tools still query SQL. The router decides between them. And the eval harness — the actual load-bearing thing — is what Week 4 builds on top.
