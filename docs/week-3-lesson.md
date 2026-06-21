# Week 3 lesson: when *not* to retrieve — structured queries, tool use, and the router

Week 1 built a naive RAG that worked. Week 2 measured it, improved it, and exposed the limits of what retrieval can do. The headline result was *Recall@8 = 0.82* — and yet, on a question like:

> *"How many applications did I send to GSK in 2024?"*

the system was almost guaranteed to be wrong. Not because the retriever was bad. Because **no single chunk contains the answer.** The answer is a count over many rows. The retriever's job is to find chunks that *contain* the answer text; if no chunk does, the system is doomed before generation begins.

The lesson of Week 3 in one sentence:

> **A RAG system that always retrieves is wrong half the time. The discipline is teaching it when *not* to retrieve.**

This week introduces a second engine — a structured query layer — alongside RAG, and a small *router* that decides which engine each question belongs to. By the end you'll have a system that answers *"Did I apply to ALESAYI?"* through retrieval, *"How many GSK applications in 2024?"* through SQL-shaped tool calls, and *"How many GSK applications and what role did I apply for?"* through both engines composed.

The lesson has the same six-part shape as Weeks 1 and 2.

---

## Part 1 — The three classes of questions RAG can't answer

Reading Week 2's golden set carefully, every question has the same shape: *"Find the email where X is true."* Did I apply to GSK? Find the email mentioning GSK. Which universities did I apply to? Find emails matching `is_university`. What role at MediaTek? Find the MediaTek email, extract the role.

Those are **lookup questions**. Their answer lives inside one chunk (factoid) or a small set of chunks (aggregation-by-find). RAG is good at them.

Three families of question break this assumption.

### Class 1 — Counts and aggregations

*"How many applications did I send in 2024?"* The answer is `387`. That number is not written in any email. It's a function of *the set of emails*, computed across all of them. There is no chunk to retrieve.

If you ask RAG, it will helpfully return 8 random application acks (top-k=8 with no obvious lexical or semantic anchor), and the LLM will either guess wildly or say *"I don't know."* Both are bad — the second is the honest failure mode you saw in Week 2; the first is the dangerous one.

### Class 2 — Comparisons and rankings

*"Which company did I apply to the most in 2024?"* This is a `GROUP BY company ORDER BY count DESC LIMIT 1`. Again — no chunk contains *"the company you applied to the most was GSK."* The answer is a derived fact.

A clever RAG might surface several GSK emails, several Amazon emails, and the LLM might *guess* — but it can't *count* properly because it only sees the chunks you give it, not the whole corpus.

### Class 3 — Temporal and structural questions

*"What month did I apply to MediaTek before?"* That one is on the boundary — the date is in the email, and Week 2's golden set actually has a similar question. But *"In which months did I have application volume above the median?"* is firmly on the structural side.

Anything that requires `WHERE`, `GROUP BY`, `ORDER BY`, `HAVING`, or `JOIN` belongs in the structured engine, not the retriever.

### The cheap improvement

You could try to fix this in RAG by:

- Increasing top-k to 200 (now the LLM gets a fuller view, but at huge token cost, with lost-in-the-middle problems)
- Adding metadata filters (good idea, but you still need someone to translate the question into filters)
- Fine-tuning the LLM on the structured data (expensive and brittle)

The cheaper improvement is just: **recognise that the question wasn't a retrieval question in the first place.**

> The most overlooked retrieval improvement is *not retrieving*.

That's the framing for the rest of the week.

**Next**: if some questions don't belong in retrieval, they need a second engine. Time to introduce it.

---

## Part 2 — The structured engine as the dual to RAG

Imagine drawing your project as a `Y`. Both arms are engines that answer questions; both are fed by the same upstream data. They differ in what they store and what they're good at.

```
                    user question
                          │
                          ▼
                      [ router ]
                       /        \
                      ▼          ▼
                [ RAG ]      [ structured ]
                 chunks         rows
                 BGE/BM25       DuckDB
              unstructured      typed columns
                  text            facts
```

### The unit of each engine

- **RAG's unit is the chunk.** A chunk is a piece of free-form text; the relationship between chunks is captured implicitly by the embedding space. Retrieval finds *similar* chunks; generation reads them.
- **The structured engine's unit is the row.** A row is a tuple of typed values; the relationships are *explicit* (this row has `company="GSK"`, `date=2024-05-12`). Queries filter, group, and aggregate over rows.

Both engines exist over the *same source corpus* — the 2,173 ack emails — but they project it through different lenses:

| | RAG | Structured |
|---|---|---|
| Source | full email text, chunked | per-email extracted fields |
| Unit | text chunk | row |
| Index | vector + lexical | SQL columns + indices |
| Query | natural language | call-shaped (count, group, filter) |
| Good at | *"Did I apply to ALESAYI?"* | *"How many in 2024?"* |
| Bad at | *"How many?"* | *"What did the GSK email actually say?"* |

The two engines are not competitors. They're **duals**. Most real questions belong cleanly to one; the interesting questions belong to both.

### Why this is easier than it sounds in our project

You already have the structured data. Your `Applications.ipynb` notebook produced `applications_unique.csv` — 2,246 rows, each with `Company`, `Role`, `Date`, `From`, `Subject`. That CSV is *already* a structured projection of the inbox. The Week 2 ack-only corpus (`ack_only.csv`) inherits those columns.

The structured engine isn't a new ETL pipeline — it's a small Python module that loads that same CSV into a typed query engine and exposes a handful of operations over it.

### Picking the engine — DuckDB vs SQLite vs pandas

Three plausible choices for the storage / query layer:

| Engine | Why pick it | Why not |
|---|---|---|
| **pandas** | already loaded, no extra dep | the API is wrong for our purposes — every query is a Python expression, hard to expose as a tool surface, no real query optimiser |
| **SQLite** | stdlib, zero install, file-based | analytical queries (`GROUP BY`, window functions) are slow on big tables, weak `DATE` handling, you'd hand-roll a schema |
| **DuckDB** | analytical-first, columnar, fast group-bys, reads CSV directly | one extra dep |

We'll use **DuckDB**. Three reasons:

1. **Analytical workloads.** Every question we'll route here is an aggregation. DuckDB is built for that; SQLite is built for OLTP (insert/update/delete).
2. **CSV-native.** `duckdb.read_csv("ack_only.csv")` works directly — no schema migration, no inserts, no ORM.
3. **It's what real RAG stacks use.** DuckDB has become the default analytical companion for LLM apps in 2025. Worth your exposure.

The downside is one extra dependency. Acceptable.

### The schema, sketched

```sql
CREATE TABLE applications AS
SELECT
    chunk_id,             -- foreign key into the RAG side
    "From"     AS from_addr,
    "Subject"  AS subject,
    "Date"     AS date,
    Company    AS company,
    Role       AS role
FROM read_csv('data/synthetic/ack_only.csv');
```

That's enough to answer counts, rankings, and date-range questions. **And critically: every row has a `chunk_id` pointer back into the vector store**, so the router can do *"count + retrieve the matching emails"* without duplicating data.

**Next**: a structured engine is no good if the LLM can't talk to it. Time to figure out what *talking to it* means.

---

## Part 3 — Function calling: tool use as the safe alternative to text-to-SQL

You have a structured engine. The LLM needs to query it. There are three patterns for letting an LLM query structured data; they differ in how much freedom the LLM gets and how much risk that creates.

### Pattern A — Text-to-SQL

```
LLM emits a SQL string.
You execute it.
You return rows.
```

Maximum flexibility, maximum risk:

- The LLM can hallucinate column names (`SELECT applied_at FROM applications` — that column doesn't exist)
- The LLM can be prompt-injected into deleting things (`'; DROP TABLE ...`)
- The LLM can write a query that scans the whole table (no `LIMIT`)
- The LLM can emit syntactically-correct, semantically-wrong SQL (`WHERE company = 'gsk'` against case-sensitive data)

Mitigations are real and standard: read-only database users, query timeouts, `EXPLAIN`-then-validate, allowlisting tables. Frameworks like LangChain's SQL agent or DSPy's `dspy.SQL` apply them. But they're a layer of defense atop a fundamentally dangerous primitive.

Worth knowing exists. Not the right level for this project.

### Pattern B — Function calling (tool use)

```
You declare a fixed set of typed Python functions.
The LLM picks which one to call, with which arguments.
You execute the function.
You return the result.
```

The LLM never writes SQL. It picks from your menu:

```python
def count_applications(company: str | None = None,
                       since: date | None = None,
                       until: date | None = None) -> int: ...

def top_companies(n: int = 10) -> list[CompanyCount]: ...

def applications_by_month() -> list[MonthCount]: ...

def find_company_role(company: str) -> list[RoleRecord]: ...
```

The LLM's job is no longer *"write code"* — it's *"recognise the intent and fill in the arguments."* Far easier; far safer. Hallucinated columns become impossible because there are no columns at the LLM's surface, only function signatures.

This is **the dominant pattern in 2025 production LLM apps**. Anthropic's tool use, OpenAI's function calling, Gemini's function calling — all the same shape. You declare a JSON schema; the model fills it in.

### Pattern C — Structured intent → query template

```
LLM emits a small JSON intent.
You translate it into a pre-written query template.
```

Safest, least flexible. Each intent maps to one template you wrote and tested. Good for very high-stakes systems (medical, legal) where the surface must be auditable.

### Why function calling for this project

We pick **Pattern B**. Why:

- **Right level of safety.** No SQL injection surface. No hallucinated columns. The LLM can only call functions you wrote.
- **Right level of flexibility.** Four functions cover ~90% of the count-shaped questions we care about. Adding a fifth is one Python function.
- **The signature is the prompt.** The LLM sees the function signature and docstring; that's the entire tool spec. Test the function; the tool surface is tested.
- **What you'd actually build at work.** Real LLM agents are 80% tool use and 20% retrieval. Learning this is more useful than learning text-to-SQL.

### What the LLM sees

When you declare a function for tool use, the SDK serialises it into a JSON schema like:

```json
{
  "name": "count_applications",
  "description": "Count application emails, optionally filtered by company and date range.",
  "input_schema": {
    "type": "object",
    "properties": {
      "company": {"type": "string", "description": "Company name, case-insensitive"},
      "since":   {"type": "string", "format": "date"},
      "until":   {"type": "string", "format": "date"}
    }
  }
}
```

The LLM is fine-tuned to recognise this format and respond with a `tool_use` block:

```json
{
  "type": "tool_use",
  "name": "count_applications",
  "input": {"company": "GSK", "since": "2024-01-01", "until": "2024-12-31"}
}
```

Your code executes the function, gets `7`, and replies with a `tool_result` block:

```json
{"type": "tool_result", "content": "7"}
```

The LLM then takes the result and produces the natural-language answer: *"You sent 7 applications to GSK in 2024."*

It's a multi-turn conversation, but every turn is mechanical. The LLM thinks once (which tool, what arguments); you execute; the LLM speaks once (turn the result into a sentence). Three steps, ~2 seconds end-to-end.

### The contract

The function you expose to the LLM is the same function you test. No duplication, no drift:

```python
def count_applications(company: str | None = None,
                       since: date | None = None,
                       until: date | None = None) -> int:
    """Count application acks, optionally filtered."""
    sql = "SELECT COUNT(*) FROM applications WHERE 1=1"
    params: dict[str, Any] = {}
    if company:
        sql += " AND lower(company) = lower($company)"
        params["company"] = company
    if since:
        sql += " AND date >= $since"
        params["since"] = since
    if until:
        sql += " AND date <= $until"
        params["until"] = until
    return con.execute(sql, params).fetchone()[0]
```

You test `count_applications("GSK", date(2024,1,1), date(2024,12,31)) == 7` as a normal unit test. The LLM tool surface is correct iff this is correct.

This is the **same boundary discipline** from Weeks 1 and 2, re-applied: the function is the contract; the LLM is just another caller of the contract.

**Next**: we have RAG and we have tools. We need something to decide which to use.

---

## Part 4 — The router: deciding which engine answers

The router is the small piece of intelligence that sits in front of both engines. Its job: classify the question into one of three buckets.

```
                question
                    │
                    ▼
              ┌──────────┐
              │  router  │  one LLM call
              └────┬─────┘
                   │
       ┌───────────┼────────────┐
       ▼           ▼            ▼
    "rag"     "structured"    "both"
       │           │            │
       ▼           ▼            ▼
   retrieve     tool call    both, then merge
```

### What the buckets mean

- **`rag`** — the answer lives in some email body. Use the Week 2 pipeline (retrieve → generate with citations). Examples: *"Did I apply to ALESAYI?"*, *"What was the role at MediaTek?"*.
- **`structured`** — the answer is a count, top-N, or filter result. Use the tool layer. Examples: *"How many applications in 2024?"*, *"Top 5 companies by application count."*.
- **`both`** — the question has two parts, one of each shape. Decompose into two sub-questions, run each, concatenate the evidence, let the final LLM call compose the answer. Example: *"How many GSK applications and what role did I apply for?"*.

### Why an LLM is the right tool for this

A naive router could be a regex:

```python
if re.search(r"how many|count|number of", question.lower()):
    return "structured"
return "rag"
```

It would work on 70% of questions. It would fail on *"What's the volume of my March 2024 applications?"* (no `count` keyword, but structural). It would also fail on *"Show me how many words are in my Edinburgh ack"* (false positive — that's retrieval + a string-length thing).

Routing is fundamentally a *natural-language understanding* task. The LLM is good at it; rules are not. The cost is small — one LLM call, ~100 tokens in, ~10 tokens out, sub-second.

### The router prompt

The prompt is short:

```
You are a question router for a job-application RAG system.

You will be given a user question. Classify it into exactly one of three categories:

- "rag":         The answer requires reading the text of specific emails.
                 Examples: "Did I apply to X?", "What role at Y?"
- "structured":  The answer is a count, aggregation, top-N, or
                 date-range filter. The answer is not in any single
                 email — it's a function over many.
                 Examples: "How many applications in 2024?",
                           "Top 5 companies by count"
- "both":        The question has two parts, one of each kind.
                 Decompose into a "structured_question" and a
                 "rag_question".

Return JSON only:
  {"mode": "rag"} or
  {"mode": "structured"} or
  {"mode": "both", "structured_question": "...", "rag_question": "..."}
```

The structured-output discipline matters. The router emits JSON, not prose. The parser is strict — if the JSON is malformed, we fall back to `"rag"` (the conservative default).

### Decomposition

For `"both"` questions, the router doesn't just classify — it *splits* the question:

> *"How many GSK applications did I send and what was the role I applied for?"*

becomes:

```json
{
  "mode": "both",
  "structured_question": "How many GSK applications did I send?",
  "rag_question": "What was the role I applied for at GSK?"
}
```

Each sub-question runs through its respective engine. The two results are then passed *together* into a final LLM call that composes the natural-language answer. You'll see this pattern in real agent systems — it's called **query decomposition**, and it's the simplest form of *planning*.

### Failure modes of the router

Three failure modes to design for, none unique to this project:

1. **The router misclassifies** — *"How many words in my Edinburgh ack?"* gets routed to structured because of "how many". The tool layer has no `count_words_in_chunk` function and the call fails. Mitigation: a generous fallback. If the tool layer can't answer, retry as RAG.
2. **The router decomposes badly** — splits a question that should be a single tool call into a tool call + a RAG call. Mitigation: tolerate it; the final composition LLM call cleans it up.
3. **The router emits invalid JSON** — rare with modern LLMs but possible. Mitigation: fall back to RAG. RAG is the conservative default because it always produces *some* answer (even if "I don't know").

The router is fast and cheap to call. If it's wrong, the system gets a slightly worse answer; if it's right, the system gets a much better one. Net positive even at moderate accuracy.

### A note: this is the minimal agent

What we're building is the *minimum viable agent*. There's no loop, no state, no memory of prior turns. The router runs once per question; one of two engines runs once; (optionally) the final composer runs once. Three LLM calls total in the worst case.

This is intentional. Real agentic systems — LangGraph workflows, CrewAI multi-agents, autonomous loops — extend the same shape with more decision nodes, more tools, more state. Once you understand the router pattern, you understand the rest by composition.

**Next**: we have the architectural skeleton. One small generation-side fix to mention before we wrap.

---

## Part 5 — Parent-document retrieval

This is a Week-2 deferred fix that pairs naturally with this week. It's not architectural — it's a small lever that improves the *answer quality* of the RAG side without changing any retrieval metric.

### The problem

Week 2's retriever finds chunks. The generation step gets those chunks. But a chunk is a 512-token piece of an email; if the email is 1,200 tokens, the LLM is seeing one-third of the relevant context.

Example: the retriever surfaces `msg_000826__c003`, which is *one chunk* of a 5-chunk InterDigital reply thread. The LLM reads that chunk, sees a partial sentence trailing off, and either says "I don't know" or guesses what the missing context was.

### The fix

When the retriever returns a chunk, expand it to the **full email** at generation time:

```python
def expand_to_parent_documents(retrieved_chunks: list[Chunk]) -> list[Document]:
    """For each retrieved chunk, return the full email it came from.

    De-duplicate: if two chunks from the same email are retrieved, the
    parent doc appears once in the result.
    """
    seen_doc_ids: set[str] = set()
    docs: list[Document] = []
    for chunk in retrieved_chunks:
        if chunk.doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(chunk.doc_id)
        docs.append(document_store[chunk.doc_id])
    return docs
```

That's the whole fix. ~20 lines including the `document_store` lookup. The retrieval metric (R@8) doesn't change — we're still finding the same chunks — but the generation gets a much fuller picture.

### Why this isn't a retrieval improvement

A common confusion: *"if I send more context to the LLM, doesn't recall go up?"*

No. **Recall is a property of the retriever, not the generator.** The retriever still surfaces a chunk per email, and the chunk has the same `doc_id` it always did. The set-cover recall metric (Week 2 Part 5) measures *did the retriever find an answer-bearing chunk*, not *how much of the parent document the LLM eventually saw*.

What changes is the LLM's faithfulness and answer quality — fewer "I don't know"s, fewer hallucinations from partial context. Those would only show up in an **answer-correctness** metric (which is Week 4's territory).

### The trade-off

The cost of parent-doc retrieval is **token budget**. A typical email is ~600 tokens. Eight retrieved chunks expand to ~8 parent docs = ~5,000 tokens. Still well inside Claude's 200K window, still cheap (~$0.015 per query at Claude Sonnet rates). Acceptable.

If chunks become bigger or top-k becomes larger, this trade-off needs revisiting. For our setup it's a free win.

### Where it lives in the pipeline

```
query
  │
  ▼
retriever (R@8)  ──► chunk_ids
                      │
                      ▼
                 expand_to_parent_documents
                      │
                      ▼
                 full email bodies
                      │
                      ▼
                 generate(query, parent_docs)
```

One new module: `src/job_application_insights/retrieval/parent_docs.py`. One small change to `_ask()` in the CLI. That's the whole fix.

**Next**: putting it all together.

---

## Part 6 — The agentic loop

Here is the full Week-3 pipeline, end to end:

```
                       question
                          │
                          ▼
                  ┌───────────────┐
                  │    router     │  one LLM call
                  │    (mode)     │  returns JSON
                  └───────┬───────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
          "rag"      "structured"     "both"
            │             │             │
            ▼             ▼             ▼
       retrieve →    tool call →    decompose:
       expand to     duckdb query   structured_q ──► tool
       parent docs   returns        rag_q        ──► retrieve+expand
            │        scalar/list         │
            ▼             │              ▼
       generate           ▼          generate
       (rag prompt)  generate        (composition prompt
            │        (numeric-       with both evidences)
            │         answer prompt)     │
            ▼             │              │
        answer ◄──────────┴──────────────┘
```

Every box is one of:

- A pure Python function (`expand_to_parent_documents`, the tool functions, the SQL execution)
- An LLM call with a specific prompt (router, RAG generator, numeric-answer generator, composition generator)

There are at most three LLM calls per question. The retriever is unchanged from Week 2. The tools are a thin shell over DuckDB.

### The CLI surface that emerges

```bash
# Week 2 still works exactly as before
uv run jai ask "Did I apply to GSK?" --retriever rerank
# (now automatically expands to parent docs by default)

# Week 3 adds the agentic mode
uv run jai ask "How many GSK applications in 2024?" --mode auto
# router → structured → count_applications → answer

uv run jai ask "How many GSK applications and what role?" --mode auto
# router → both → decompose → tool + RAG → compose
```

`--mode auto` becomes the default. To force a single engine you can still pass `--mode rag` or `--mode structured`. The CLI is a thin façade over the routing logic.

### What changes in the evaluation harness

This is where Week 3 gets interesting from a measurement perspective. The Week-2 metrics (P, R, MRR, nDCG over chunk IDs) **only apply to the RAG branch**. They say nothing about a tool call.

The eval extension is small but conceptual:

- Add a `mode` field to `GoldenEntry`: `"rag" | "structured" | "both"`. Tells the harness which engine should answer.
- For structured entries, add a `expected_answer` field (the scalar or list the tool should return).
- For `both` entries, expect both a structured answer *and* a chunk-set.

Then we report:

- *Routing accuracy*: did the router pick the right engine?
- *Structured accuracy*: when routed to tools, did the tool return the right answer?
- *Retrieval recall (R@8)*: when routed to RAG, did the retriever still surface the right chunks?

These three metrics are **independent**. The router can be 100% right and the tool layer 0% right (wrong function called). The tool layer can be 100% right and the router 50% right (sends RAG questions to tools and gets nothing). Each metric isolates one failure mode.

Week 4 is where we add **answer correctness** — the LLM-as-judge layer that measures *whether the final natural-language answer is right*, not just whether the pipeline did the mechanics right. That's the metric every enterprise RAG system lives or dies by, and it's the natural next step after this week's plumbing.

### Per-day shape of Week 3

| Day | What we build | Why it matters |
|---|---|---|
| 1 | DuckDB side-table from `ack_only.csv`. Pure data loader. Schema + tests. | Foundation. No LLM yet; pure SQL plumbing. |
| 2 | Four typed Python functions over the table (`count_applications`, `top_companies`, `applications_by_month`, `find_company_role`). Pydantic return models. Pure unit tests. | The tool layer's surface. Tested without the LLM in the loop. |
| 3 | Wire tools to Anthropic's tool-use API. `jai ask --mode tools` invocation. Verify on the count-style questions Week 2 couldn't answer. | The LLM finally meets the structured engine. |
| 4 | LLM-based router + decomposition. `jai ask --mode auto`. Extend the golden set with ~5 count-style and ~3 hybrid questions; re-run eval. | The agentic skeleton. |
| 5 | Parent-document retrieval. `docs/week-3-lesson-results.md` writeup (mirrors Week 2's structure). | The tactical fix and the close. |

---

## What you'll be able to put on your CV after this week

The honest, defensible bullet:

> *Extended the RAG system with a tool-use agent layer (DuckDB-backed structured engine + four typed Python tools surfaced via Anthropic's tool-use API) and an LLM-based question router that classifies each query into retrieval / structured / hybrid and decomposes compound questions. Closed the "count and aggregation" failure class that pure RAG cannot answer.*

The CV bullet pitches the architectural move, not a number — because the metric for routing accuracy + structured correctness is new this week and not directly comparable to Week 2's R@8.

What an interviewer will probe:

- *Why function calling and not text-to-SQL?* — Smaller surface, no injection risk, the function signature is the contract. Right level of safety for this scale of system; text-to-SQL is the answer when the query language is the product (e.g. Vanna, Snowflake Cortex).
- *How does the router fail?* — Three modes (misclassify / bad decomposition / malformed JSON), with a conservative fallback to RAG. The bad-classification cost is bounded: you get RAG's answer when you wanted structured, or vice versa, and the user can rephrase. Not silent corruption.
- *How do you evaluate this?* — Three independent metrics — routing accuracy, structured accuracy, retrieval recall — instead of one. Week 4 layers LLM-as-judge on top for answer correctness.
- *Is the LLM call to the router worth it?* — Cheap (~$0.0005 per question on Haiku, ~50ms), much better than regex on the kinds of questions users actually ask. If the router became a bottleneck we'd cache by question template or fine-tune a small classifier.

---

## New vocabulary you should now own

- **Structured query layer** — a typed, queryable side-table that lives alongside the vector store
- **Tool use / function calling** — the LLM picks a function from a menu and fills in arguments; the menu is the safety boundary
- **JSON schema for tools** — what the LLM actually sees; derived from your Python signature
- **`tool_use` and `tool_result` blocks** — the message shape in a multi-turn tool-use conversation
- **Question router** — small LLM call that classifies a question by the engine that should answer it
- **Query decomposition** — splitting a compound question into single-engine sub-questions
- **Parent-document retrieval** — at generation time, expand each retrieved chunk to its full source document
- **Routing accuracy** — fraction of golden-set questions sent to the right engine
- **Structured accuracy** — for tool-routed questions, fraction with correct tool output
- **Compound question** — one that needs both structured and RAG evidence to answer well
- **Text-to-SQL** — the alternative pattern we deliberately did *not* adopt, and why
- **Agentic loop / agent skeleton** — router + tools + composition is the minimum viable form
- **Conservative fallback** — when in doubt, route to RAG (always produces *some* answer)

---

## Conceptual sanity check

Answer these out loud before we start coding:

1. Why is *"How many applications did I send in 2024?"* a question retrieval can't answer, no matter how good the retriever is?
2. What's the difference between *"the unit of relevance"* in RAG vs in the structured engine?
3. Why is function calling safer than text-to-SQL, given both let the LLM "query the database"?
4. What does the router's JSON output look like, and what happens if it's malformed?
5. What's the difference between *routing accuracy* and *structured accuracy*? Can one be high and the other low?
6. Parent-document retrieval doesn't move Recall@8. Why not? Which metric *would* it move?
7. What's the conservative fallback if the router fails, and why is it RAG and not structured?

---

# Appendix — Conceptual deep-dives that will come up during implementation

## Appendix A — The JSON schema as a contract

When you declare a Python function for tool use, you're really declaring a contract in two languages at once:

1. **Python**: the function signature with type hints — what your *code* will call.
2. **JSON schema**: a serialised description of the same signature — what the *LLM* will see.

Anthropic's SDK can derive the JSON schema from the Python signature automatically if you use Pydantic or `BaseModel`-typed arguments, but it's worth understanding what's being generated:

```python
def count_applications(company: str | None = None,
                       since: date | None = None) -> int: ...
```

becomes

```json
{
  "type": "object",
  "properties": {
    "company": {"type": "string"},
    "since":   {"type": "string", "format": "date"}
  },
  "required": []
}
```

The docstring of the Python function becomes the `description` field; the parameter types become `properties`; `Optional[X]` parameters become non-required.

This is structurally the same pattern as Pydantic's `model_json_schema()` — and not coincidentally. Tool use was designed to slot into the JSON-schema ecosystem that already existed for OpenAPI, FastAPI, and Pydantic.

The interview-ready way to say this:

> *"The Python function signature and the LLM tool spec are the same contract, expressed twice in different languages. Pydantic + Anthropic's SDK keep them in sync; the function I test is the function the LLM calls."*

## Appendix B — Why DuckDB and not pandas

A subtle point that came up at Week 2's wrap: *"we already use pandas — why add DuckDB?"*

Three reasons, in increasing order of importance:

1. **Query expressivity.** `SELECT company, COUNT(*) FROM applications WHERE date BETWEEN ? AND ? GROUP BY company ORDER BY 2 DESC LIMIT 5` in DuckDB is one line. In pandas it's `df[df.date.between(s,e)].groupby("company").size().sort_values(ascending=False).head(5)`. Both work, but the first reads like the question, and the LLM emits queries that *read* like the question.
2. **Tool-spec stability.** A function that wraps a SQL string can have a stable signature (`since`, `until`, `company`) even if the underlying schema changes. A function that wraps pandas operations is harder to evolve without changing argument names.
3. **Career exposure.** DuckDB has eaten the analytical-Python niche over 2023-2025. Knowing it is a real skill; you'll see it at companies that pair LLMs with data.

We use DuckDB *as a query engine*, not as a database. The data lives in CSV; DuckDB reads CSV directly; no migrations, no schema management. The lightest possible adoption.

## Appendix C — The router as the simplest form of planning

There's an entire research area called *LLM planning* — multi-step reasoning where the LLM decides a sequence of tools to call, observes the result of each, and updates its plan. ReAct, Chain-of-Thought, Tree-of-Thoughts, LangGraph's `StateGraph` — all variations on the same theme.

Our router is the **single-step degenerate case** of this:

> *Step 1: pick the right tool.*
> *Step 2: there is no step 2.*

That's deliberate. We're not building autonomy; we're building **task routing**. The router has full information up front (the question), makes one decision (the engine), and the system executes it. No loops, no observation, no replanning.

If we ever needed loops — e.g. *"first count GSK acks, then if more than 5, retrieve the most recent ones"* — we'd extend the router into a small `StateGraph`. The shape is:

```python
graph.add_node("route", router_fn)
graph.add_node("structured", tool_executor)
graph.add_node("rag", rag_executor)
graph.add_edge("route", "structured" | "rag", condition=mode)
```

LangGraph and similar libraries formalise this. You don't need them for Week 3; you do need them when the routing decision depends on the result of an earlier tool call.

The lesson: **start with a fixed router**, see what it can't express, *then* reach for the graph framework. Pre-emptive complexity is the failure mode of every "agent framework" tutorial.

## Appendix D — Eval discipline carries over

Week 2 invested heavily in the eval discipline: golden set + pure metrics + type-seam runner. The temptation in Week 3 is to skip it because routing accuracy "feels easy to eyeball."

Don't. The same discipline applies, just over a larger surface:

```python
class GoldenEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    question: str
    mode: Literal["rag", "structured", "both"]
    # rag fields
    answer_groups: list[list[str]] | None = None
    # structured fields
    expected_tool: str | None = None        # e.g. "count_applications"
    expected_args: dict[str, Any] | None = None
    expected_result: Any | None = None
    # for "both" — both kinds populated
```

The boundary pattern from Weeks 1 and 2 is the same: validate at load time, treat downstream as trusted. The metrics become a family:

```python
def routing_accuracy(predicted_mode, expected_mode) -> float: ...
def structured_accuracy(predicted_result, expected_result) -> float: ...
def recall_at_k_groups(retrieved, groups, k) -> float: ...
```

Three numbers, each isolating one component. **Composability over conflation.** When the system regresses, you can tell which component is at fault without re-running everything.

## Appendix E — Fallbacks and the conservative-default principle

Throughout this week we'll hit choice points where the system has uncertainty: router emits malformed JSON, tool call raises an exception, retriever returns zero chunks. The temptation is to ask the LLM to retry, or to surface the error.

The right move is almost always: **fall back to the most general engine, return an honest "I don't know" if needed.**

In our setup, RAG is the conservative default because:

- It always produces *some* output, even if "I don't know"
- It can't return obviously wrong numeric answers
- It defaults to citing the user's own data, which is auditable

The hierarchy in fallback order:

1. Router says `structured`, tool succeeds → use the answer.
2. Router says `structured`, tool fails → fall back to RAG.
3. Router fails → fall back to RAG.
4. RAG fails (no chunks) → say "I don't know."

This is a small but real engineering discipline. Each fallback edge is one line of code, but writing them down before you build the system stops the production version from being a tower of `try/except`.

## Appendix F — Why we're not adopting LangChain / LangGraph for this week

You'll notice the Week 3 plan doesn't mention LangChain, LangGraph, LlamaIndex, or any "agent framework." That's intentional. Two reasons:

1. **The router pattern is ~50 lines of Python.** Adopting a framework costs more lines and more concepts than writing the router directly.
2. **You learn the shape better by building it.** Once you've written a router-tools-compose loop by hand, you'll recognise it in every framework you read later. The reverse — learning framework abstractions first — leaves you fluent in a vocabulary but unable to debug what it hides.

The frameworks become useful when the agent graph gets big (5+ tools, multi-step reasoning, conditional branching, state management). We're not there yet.

If you ever want to *port* this project to LangGraph, the seams we've designed (RetrievalFn from Week 2, the tool functions from this week, the router function) map cleanly to LangGraph nodes. The framework would *describe* the same architecture; it wouldn't change it.

---

## Where to go next

Week 3 closes the *"questions RAG can't answer"* gap. What it explicitly leaves for Week 4:

- **LLM-as-judge for answer correctness** — the metric every production RAG system uses. Reference answers + a separate LLM judge + a Ragas-style rubric.
- **HyDE / query rewriting** — addresses the vocab-mismatch failures that survive even hybrid retrieval.
- **Configuration sweep** — chunk size × embedder × retriever × reranker, automated.
- **Production wrapper** — FastAPI service + Streamlit demo + Docker + a public-facing URL.

The Week 3 architecture is what the deployment layer will sit on top of. The router becomes the API entrypoint; the tools become whatever the user surface needs them to be; the RAG path keeps doing what it did.

Numbers and metrics will keep evolving. The agentic skeleton you build this week — **router + tools + RAG + composition** — is the shape that stays.

That's the deliverable. Build the skeleton; the muscles will follow.
