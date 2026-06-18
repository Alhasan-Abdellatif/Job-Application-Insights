# Week 1 lesson: from raw emails to a question-answering system

> Foundational lesson for Week 1 of the job-application-insights project.
> This is the **why and what** before the **how**. After reading it, you should
> be able to describe the entire Week 1 RAG pipeline on a whiteboard without
> looking anything up.

The lesson is organised in 6 parts. Each builds on the previous. Each one ends with a sentence connecting it to the next.

---

## Part 1 — What is RAG, and why does it exist?

### The core problem

You can ask Claude or ChatGPT: *"How many applications did Alhasan fire in June 2026?"* and it will confidently make something up. The LLM has no idea who you are. Its training data is fixed and frozen at some date in the past, and your 2,200-row CSV was never part of it.

You have three families of solutions:

| Approach | What it does | Cost |
|---|---|---|
| **Fine-tune** the LLM on your data | Re-train the model itself so it "knows" your emails | Expensive ($$$$), retraining every time data changes, hard to remove or update specific facts, risk of leaking private data |
| **In-context only** (paste everything) | Dump your whole dataset into the prompt every time | Free of training, but limited by context window. Even Claude's 200K-token window can't fit 2,200 emails — and even if it could, you'd pay for those tokens *on every single query* |
| **Retrieval-Augmented Generation (RAG)** | At query time, find the *small handful* of relevant documents, paste *only those* into the prompt | Cheap, fast, easy to update, easy to delete |

RAG isn't an algorithm — it's a **pattern**. The whole pattern in one sentence:

> *Given a question, retrieve the small set of documents from your corpus that are most likely to contain the answer, paste those documents into the prompt as context, and ask the LLM to answer using only that context.*

### The three letters

- **R — Retrieval**: how do we find the relevant documents? (This is where most of the engineering goes — and where most "RAG demos" fail.)
- **A — Augmentation**: how do we paste the retrieved documents into the prompt? (Prompt design.)
- **G — Generation**: the LLM produces the final answer. (Off-the-shelf API call.)

Counter-intuitive truth: **the "G" is the easy part**. Anyone can call an LLM API. The hard part is the "R". You will spend most of Week 2 making the "R" better.

### When does RAG NOT make sense?

Knowing this is important — interviewers will ask:

- If your data fits in the context window AND your query latency tolerates the full dump, just paste it in (no RAG needed).
- If your data needs *reasoning* across many documents (e.g., aggregations, comparisons, math), pure RAG struggles — you need an **agent** that can call SQL or pandas (which is what we build in Week 3).
- If your data changes 10× per second (real-time stock prices), the embedding pipeline becomes the bottleneck and you want a streaming architecture.

Your job-application data fits RAG perfectly: ~2,000 documents, mostly text, mostly read-only, queries are semantic ("which apps mentioned scientific ML"), and we'll add structured-query routing in Week 3 for the count-style questions.

**Next**: to retrieve relevant chunks, we first need to break documents into chunks. To do that we need to understand tokens.

---

## Part 2 — Tokens and chunking

### What's a token?

LLMs don't see characters or words — they see **tokens**. A token is a sub-word unit produced by a learned tokeniser (BPE — Byte Pair Encoding). Some heuristics:

- ~4 characters of English = 1 token
- ~0.75 words = 1 token
- The word `"applications"` might be `["app", "lic", "ations"]` (3 tokens)
- The word `"the"` is usually 1 token
- Emoji, code, non-Latin scripts → many more tokens per character

Why this matters: every model has a **context window** measured in tokens, every API call charges per token, and every chunking decision is a token-budget decision.

For your data: a typical LinkedIn ACK body is ~400-600 tokens. A Greenhouse rejection might be ~150-300. Sub-bodies vary wildly. We need to handle this.

### Why we can't paste the whole corpus

Even with Claude Sonnet 4.6's large context window (~200K tokens), you'd hit problems:

1. **Money**: ~$3 per million input tokens. 2,200 docs × 500 tokens × 1 query = 1.1M tokens × $3 = ~$3 per query. Run it 1,000 times during development = $3,000.
2. **Latency**: an LLM processing 200K tokens of input takes 5-15 seconds. Useless for a UI.
3. **Quality**: LLMs get *worse* at long contexts — the "lost in the middle" problem. Information buried in the middle of a long prompt is often ignored.

So we have to be selective. We send only the ~5-10 most relevant chunks per query.

### What's a chunk?

A **chunk** is a contiguous piece of a document, small enough to fit comfortably in the LLM context budget alongside others. Typical sizes are 200-1000 tokens.

We chunk because:

- An entire long body might be 2,000 tokens but only one paragraph is relevant — embedding the whole body dilutes the signal
- Retrieval works at the *chunk* level, so the chunk is the smallest "unit of relevance" you can return

### Chunking strategies (in order of sophistication)

**Strategy 1 — Fixed-size by character count**

```
Take 1000 chars, then next 1000, then next 1000...
```

Brain-dead simple. Splits sentences mid-word. Bad for any structured text. **Almost never used in practice** except in toy demos.

**Strategy 2 — Fixed-size by tokens with overlap**

```
Token 0-500, then 450-950, then 900-1400... (50-token overlap)
```

Better. Overlap means if a sentence is cut at the boundary, the next chunk captures it whole. The standard naive baseline.

**Strategy 3 — Recursive character splitting** ← what we'll use in Week 1

```
Try to split on \n\n first. If too big, split on \n. Then on '. '. Then on ' '.
```

Respects natural document structure. LangChain's `RecursiveCharacterTextSplitter` does this. Easy, effective for most cases. Sensible default.

**Strategy 4 — Semantic chunking**

```
Embed each sentence. Split when adjacent sentences become semantically dissimilar.
```

Smart but slow and expensive. Marginal quality gain. Defer to Week 2+.

**Strategy 5 — Late chunking** (cutting-edge, 2024)

```
Embed the WHOLE document first, then chunk the embeddings (not the text).
```

Preserves long-range context in each chunk's embedding. Research-quality. Worth knowing exists; don't bother implementing in Week 1.

### Why overlap helps

If you chunk `"Thank you for applying to Vatic Labs. We received your application on 2026-06-08. Our team will review."` at a fixed boundary and the cut falls between sentence 1 and sentence 2, a query about "Vatic Labs application date" might retrieve sentence 1 alone — losing the date. With 50-token overlap, chunk 2 starts a bit earlier, so both pieces of context co-occur in at least one chunk.

The cost: you have ~10% more storage and embedding work. Usually worth it.

### Concrete numbers for Week 1

| Setting | Value | Why |
|---|---|---|
| Chunk size | 512 tokens | Big enough to hold a typical email; small enough to retrieve many per query |
| Overlap | 64 tokens | ~12% — standard recommendation |
| Splitter | `RecursiveCharacterTextSplitter` | Respects paragraph/sentence boundaries |
| Metadata per chunk | `{doc_id, subject, from, date, company, source}` | Lets us filter later (e.g. "only UK apps") |

**Next**: we have chunks. Now we need to turn them into something we can search by meaning.

---

## Part 3 — Embeddings: turning text into vectors

### The fundamental idea

Imagine plotting every English sentence on a map. Sentences with similar *meaning* should be close together; sentences about totally different things should be far apart. We'd want:

- *"The cat sat on the mat"* and *"A feline rested on the rug"* → very close
- *"The cat sat on the mat"* and *"Quarterly earnings report"* → far apart
- *"Thank you for applying to Amazon"* and *"We've received your application to Amazon"* → very close

That "map" is high-dimensional (typically 384, 768, 1024, or 1536 dimensions — not 2). And the way each sentence gets a position is via an **embedding model** — a neural network that takes text and outputs a fixed-size vector.

This is exactly the same idea as word2vec from your ML background, but trained on much more data, at sentence/passage level, with a transformer encoder.

### What an embedding model is

It's a transformer **encoder** (no decoder). You give it tokens, it returns a single fixed-size vector. The model was trained on huge datasets of (similar pair, dissimilar pair) examples so that semantic similarity maps to vector closeness.

Crucially:

- It's **deterministic** — same input always gives the same vector
- It's **fast** — milliseconds per query, batch-process thousands per second on GPU
- It's **separate from the LLM** — embedding model and LLM are two different models

### Distance / similarity metrics

Given two vectors, how do we measure similarity?

- **Cosine similarity**: cos(θ) between the vectors. Range [-1, 1]; 1 = identical direction. **The default for most embedding models** — they're trained for this.
- **Dot product**: just the dot product. Equivalent to cosine if vectors are L2-normalised (which most modern embedding models do). Faster to compute.
- **L2 / Euclidean distance**: straight-line distance. Works but rarely used for text.

For Week 1: use cosine. The vector DB will handle it.

### Picking an embedding model

| Model | Dim | Size | Use when |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | ~130MB | Default starter — fast, free, great quality for size. **Pick this for Week 1.** |
| `BAAI/bge-large-en-v1.5` | 1024 | ~1.3GB | Better quality, slower, more memory. Try in Week 2 if quality is the bottleneck. |
| `intfloat/e5-base-v2` | 768 | ~440MB | Strong alternative, comparable to BGE |
| `text-embedding-3-small` (OpenAI) | 1536 (configurable) | API | If you don't want to host a model. Costs ~$0.02 per million tokens. |
| `mxbai-embed-large-v1` | 1024 | ~670MB | The current free-model king as of late 2025 |

We start with `bge-small-en-v1.5`. Three reasons:

1. **Free** — runs locally via sentence-transformers
2. **Small** — fits in laptop RAM, can also embed batches quickly on CPU
3. **Good** — top of the MTEB leaderboard for its size class

### The embed-once-query-many pattern

This is the key efficiency trick:

- **At ingestion**: embed every chunk in your corpus. Store the vectors. *This happens once.*
- **At query time**: embed only the user's query. Compare the query vector to all stored vectors. Return the closest k.

Embedding all 2,200 docs (chunked into ~5,000 chunks) takes about 1 minute on a laptop CPU. Embedding a query takes ~10ms. Searching the index for nearest neighbours takes ~5ms.

So the slow step happens once, offline, and queries are fast.

### What "semantic search" actually does

Given query `"machine learning applications in Switzerland"`:

1. Embed the query → vector q (384 dims)
2. For every chunk vector c in storage, compute cos(q, c)
3. Return top-k chunks by similarity

The query never has to literally appear in the document. *"ML postdoc in Zurich"* would match because semantically they're close.

This is the **superpower** over keyword search. It's also the **danger** — semantic search can return *plausibly-related-but-irrelevant* chunks. That's why we add BM25 + reranking in Week 2.

**Next**: we have vectors. We need somewhere to store them and search them efficiently.

---

## Part 4 — Vector databases

### Why not just numpy?

For 5,000 vectors? You absolutely could. `numpy` with cosine similarity over a single matrix takes <100ms. We could implement everything in numpy in 50 lines.

But you'd be missing:

1. **Persistence** — load/save from disk without re-embedding
2. **Metadata filtering** — "search only UK apps from 2025"
3. **Approximate Nearest Neighbour (ANN)** — for when you have 50M vectors and need <10ms queries
4. **Updates** — add new documents without rebuilding the index

A vector database packages these.

### Approximate Nearest Neighbour (ANN)

For small data, exact search (look at every vector) is fine. For big data, we need approximate methods.

**HNSW** (Hierarchical Navigable Small World) — the dominant algorithm — builds a multi-layer graph:

- Top layer: sparse graph with long-distance connections
- Bottom layer: dense graph with short-distance connections
- A query traverses top-to-bottom, narrowing down rapidly

Trade-off: 95-99% recall (you find 99% of the true top-k, not 100%) for 100-1000× speed-up. Acceptable for almost all use cases.

You don't need to implement HNSW. The vector DB does it. But understand that **dense vector search is approximate**, and that's a parameter you can tune.

### The choices

| DB | Tier | When to use |
|---|---|---|
| **Chroma** | Local, embedded | **Week 1** — zero setup, runs in-process, perfect for prototyping |
| **Qdrant** | Production, Rust-backend | **Week 2+** — runs in Docker, has the metadata filtering features we'll need |
| **FAISS** | Library only | When you want to embed in another system (no server) |
| Pinecone, Weaviate, Milvus | Production, cloud | Real-scale deployment. Overkill for this project. |
| pgvector (Postgres extension) | If you already have Postgres | Honourable mention — your data is in Postgres anyway |

Week 1: Chroma. Move to Qdrant in Week 2 for the metadata filtering and reranking integrations.

### Metadata: the unsung hero of RAG

Every chunk we store should have rich metadata:

```python
{
    "doc_id": "msg_12345",
    "subject": "Thank you for applying to Vatic Labs",
    "from": "...",
    "date": "2026-06-08",
    "company": "Vatic Labs",
    "country": "United States",
    "source": "ats_relay",
    "is_research_track": False,
}
```

This lets us answer queries like *"Show only UK applications from 2025"* with a filter, not a hope-it's-in-the-text retrieval. Hybrid filter+vector retrieval is far better than vector alone.

**Next**: we have stored vectors. Now the retrieval logic.

---

## Part 5 — Retrieval: from query to relevant chunks

### The minimum viable retrieval

```
1. user_query  →  embed with SAME model used for documents
2. query_vector  →  vector_db.search(query_vector, k=8)
3. return top-8 chunks
```

That's it.

### Why "same model"

If chunks were embedded with `bge-small` but the query is embedded with `e5-base`, the vector spaces don't align — similarities become meaningless. Use the same model for both.

### How big should k be?

Trade-off:

- Too small: miss relevant chunks (low recall)
- Too big: include irrelevant chunks, dilute the prompt, confuse the LLM

For Week 1: **k=8**. Justifiable defaults: 5-10. We'll measure recall in Week 2 and tune.

### What retrieval CAN'T do

Naive vector retrieval will fail spectacularly on:

- **Exact-match queries**: *"applications with reference ID 2026-139165"* — BM25 wins, vector loses
- **Numeric / structural questions**: *"count applications by month"* — neither vector nor BM25 helps. We need SQL.
- **Negation**: *"applications that did NOT mention AWS"* — vector models are bad at "not"
- **Multi-step questions**: *"compare UK to Switzerland by application volume per month"* — needs aggregation

This is why Week 2 adds BM25 + reranking, Week 3 adds an agent that routes to SQL or vector depending on the query.

For Week 1, we accept these limitations and just build the simple semantic pipeline.

### Citations

Every retrieved chunk has a `doc_id`. We must preserve this all the way to the final answer so the user (and any evaluator) can verify the LLM didn't hallucinate.

In Week 1 the citation pattern is: format each chunk in the prompt as `[source: msg_12345]` and instruct the LLM to cite its sources by ID.

**Next**: we have relevant chunks. Now we ask the LLM.

---

## Part 6 — Augmented generation: prompting the LLM

### The anatomy of a RAG prompt

```
SYSTEM: You are an assistant that answers questions about Alhasan's
job application emails. Use ONLY the provided context to answer.
If the context doesn't contain the answer, say "I don't know"
rather than guessing. Always cite the source IDs you used.

USER: <Question>
Where did I apply that mentioned scientific ML?
</Question>

<Context>
[source: msg_001] Thank you for applying to Tomorrow Climate. Your
research on neural operators for scientific computing aligns with
our team's work on physical simulation...

[source: msg_007] Application received - Senior Research Scientist
position at Schrödinger. We're impressed by your scientific ML
background...
</Context>

Answer the question above using only the context. Cite sources.
```

### Why each piece exists

- **System message** sets the role and the rules. The "say I don't know" instruction is the single most important behavior we want — without it, the LLM hallucinates confidently.
- **`<Question>` and `<Context>` tags** (XML-style) help Claude parse the structure. Anthropic's docs strongly recommend this pattern.
- **Cite sources** — turns the LLM's output into something verifiable.

### Two LLM-specific things to learn

**1. Temperature**

A number 0-1 controlling randomness. For RAG, you want **temperature=0** (or 0.1). You want the *most likely* answer, not a creative one.

**2. The LLM is an SDK call**

```python
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    temperature=0,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_message}],
)
print(response.content[0].text)
```

This is the *entirety* of the "Generation" step. The hard work is in retrieval, not generation.

### Common naive-RAG failure modes (we'll see all of these in Week 1)

1. **Wrong chunk retrieved** → LLM answers based on irrelevant context
2. **Chunk retrieved but doesn't actually answer** → LLM hallucinates, ignores instruction
3. **Question is statistical / aggregating** → vector retrieval finds no chunk that "contains the answer" because the answer doesn't exist in any single chunk
4. **Answer requires synthesis across many chunks** → top-k=8 happens to miss one critical chunk
5. **LLM cites wrong source** → it makes up the IDs

We'll observe these on Week 1 evals and design Week 2-3 to fix them.

---

## Putting it all together — Week 1 deliverable

By the end of Week 1, you have this pipeline:

```
inbox_mail.csv
     │
     ▼
 [parse.py]   ── reads CSV, applies the classifier from Applications.ipynb,
     │           produces a list of normalised Document objects
     ▼
 [chunk.py]   ── RecursiveCharacterTextSplitter, 512 tokens, 64 overlap
     │
     ▼
 [embed.py]   ── sentence-transformers / bge-small-en-v1.5
     │
     ▼
 Chroma local persistent collection (./data/chroma)
     │
     ▼
 [retrieve.py] ── embed query → search → top-8 chunks
     │
     ▼
 [generate.py] ── Anthropic / OpenAI / Gemini SDK → answer with citations
     │
     ▼
 jai ask "Which companies have I applied to in Switzerland?"
```

CLI demo:

```bash
$ uv run jai ingest data/raw/inbox_mail.csv
Parsed 2,246 documents
Chunked into 4,891 chunks
Embedded with BAAI/bge-small-en-v1.5
Stored in ./data/chroma

$ uv run jai ask "Which companies have I applied to in Switzerland?"
Based on the application emails, you've applied to:
- Roche (Basel) — Senior Research Scientist [source: msg_0451]
- ETH Zurich postdoc applications [source: msg_0892]
- Disney Research Zurich [source: msg_1102]
...

$ uv run jai ask "How many applications in March 2026?"
I don't have enough context to count applications. The retrieved
documents are individual emails, not aggregate counts.
```

Notice the last query fails — that's the gap Week 3 fills with SQL routing.

---

## The new vocabulary you should now own

After this lesson, these terms should be no-look words:

- **Token** — sub-word unit, what the LLM "sees"
- **Context window** — max tokens per request
- **Chunk** — a piece of a document, the unit of retrieval
- **Overlap** — chunks sharing some tokens at their boundary
- **Embedding** — fixed-size vector representing semantic meaning
- **Cosine similarity** — the default similarity metric for text embeddings
- **Encoder model** — neural network that produces embeddings
- **HNSW** — the dominant approximate-nearest-neighbour algorithm
- **Vector DB** — system for storing and searching embeddings + metadata
- **Recall@k** — fraction of relevant items found in top k (Week 2)
- **Hallucination** — LLM confidently producing wrong information
- **Grounding** — making the LLM answer based on retrieved context, not its training memory
- **Citation** — pointer from the answer back to the source

---

## Conceptual sanity check

Try these mentally before looking up answers:

1. Why do we embed once and query many times instead of the other way around?
2. Why does the chunk size matter? Why not just embed full documents?
3. Why do we use the same embedding model for documents and queries?
4. Why does naive RAG fail on "How many applications in March 2026?"
5. What would happen if temperature was 1.0 in our RAG prompt?
6. Why is metadata stored alongside the vector?
7. What's the single most important instruction to put in the system prompt?

---

# Appendix — Conceptual deep-dives that came up during implementation

These are the concept explanations that surfaced during the Week 1 build, in the order they came up.

## Appendix A — The boundary pattern (the "Pydantic v2 typed ingestion layer" CV bullet)

We wrote `parse.py` so its job is converting *untrusted, unstructured input* into *trusted, structured data*. Everything downstream (chunker, embedder, retriever, LLM) assumes the data is clean. The cost of being strict at the boundary pays back tenfold later.

The pattern in three layers:

### Layer 1 — Parsing (input → raw values)

Read the CSV. Don't trust types. Use `row.get(...)` not `row[...]` so missing columns become empty strings instead of `KeyError`.

### Layer 2 — Normalisation (raw → clean)

Apply the messy real-world fixes you've already discovered:

- MIME decoding (`=?UTF-8?Q?…?=` → readable text — fixes the ALESAYI HOLDING case)
- Whitespace collapsing (`"\r\n Technology Oxford"` → `"Technology Oxford"` — fixes the LinkedIn folded subject)
- HTML stripping (`<div>Hello</div>` → `Hello` — handles GSK / Workable templates)
- NaN-safe via `_is_missing()` (pandas reads empty strings as `NaN`)

### Layer 3 — Validation (clean → typed)

Wrap each row in a `Document(...)` constructor. Pydantic validates. If anything is wrong — wrong type, missing required field, mutating a frozen field — you get a `ValidationError` immediately, not a `TypeError` 500 lines downstream.

**Why frozen?** Once a Document exists, its content is final. This is what mathematicians call a "value object" — equal Documents are interchangeable. Frozen instances are safe to use as dict keys, share across threads without copying, and won't be mutated by some clever-but-wrong code three modules deeper.

### Why this matters as a "real engineer" signal

Every production LLM system has an ingest module that looks like this. Almost every portfolio RAG project skips it — they pass raw `dict`s around and silently corrupt data. Yours doesn't.

The CV-bullet version:

> *"Pydantic v2 typed ingestion layer with immutable value objects, MIME-aware decoding, and per-field validation — boundary cleanly separates trusted internal data from raw I/O."*

Phrase-by-phrase breakdown of that bullet:

- **Pydantic v2** — a library that turns class definitions into runtime data validators. "v2" because there was a major breaking rewrite in 2023 (engine moved to Rust, `class Config` became `model_config = ConfigDict(...)`). Mentioning v2 signals you're current.
- **typed** — every field has explicit Python type hints (`doc_id: str`, `date: datetime | None`). Mypy reads them statically; Pydantic uses them at runtime.
- **ingestion layer** — the layer of the architecture responsible for bringing external data into the system.
- **immutable value objects** — objects identified by their data (not identity); frozen after construction. Set via `model_config = ConfigDict(frozen=True)`.
- **MIME-aware decoding** — understands RFC 2047 (`=?UTF-8?Q?…?=`) so non-ASCII subjects parse correctly.
- **per-field validation** — each field has its own rules (`min_length=1`, etc.). Violations raise at the boundary, not 500 lines deeper.
- **boundary** — architectural term: the edge of your system where it touches the outside world. *Outside* it: nothing is trusted. *Inside*: everything is.
- **cleanly separates** — no leakage. After the boundary, no downstream code needs to know that the data came from CSV.
- **trusted internal data** — three things now guaranteed: known shape, known cleanliness, known invariants.
- **raw I/O** — bytes / strings as they come from disk / network. Possibly malformed, unvalidated, no guarantees.

Interview answer in one breath:

> *"I treated the data ingest as a hard boundary. Raw email CSVs go in — they might have MIME-encoded subjects, embedded HTML, NaN cells, header line-folding. I use a Pydantic v2 model called `Document` that's frozen, so once a row crosses the boundary it's both validated and immutable. The chunker and retriever downstream don't have to defend against malformed inputs; they just consume Documents."*

## Appendix B — The `# public API` comment

When you write a module, some things are meant for the outside world (other modules, tests, the CLI) — and some things are internal scaffolding. The "public API" is the contract you promise to keep stable. The internal stuff is implementation detail you reserve the right to change.

Python expresses this three ways:

1. **The `_` prefix** (universal, lightweight). `_normalise` is private; `load_documents` is public. Tooling treats `_`-prefixed names as implementation details.
2. **`__all__`** (explicit export list). Controls what `from module import *` brings in. The formal "public surface" declaration.
3. **Visual section headers** like `# ────── public API ──────`. Pure comments, but they give reviewers a 1-second mental map of the file and enforce the discipline of "private first, public second."

Same idea as project-level layering (`ingest → retrieval → agents → api`), applied to a single file.

CPython itself: `_thread`, `_collections_abc`, `_ssl` — underscore-prefixed modules signalling internal use. Pydantic v2: `pydantic._internal.*` is all private; `pydantic.*` is the public surface.

## Appendix C — `uv run` vs plain `python`

`python` alone resolves to whatever interpreter is first on `PATH` — often the system Python (3.9 on this Mac) with no project packages installed. Run `python src/cli.py` and you'll hit `ModuleNotFoundError`.

`uv run <command>` does three things:

1. Verifies `.venv/` is current vs `uv.lock`; installs any missing deps in milliseconds
2. Prepends `.venv/bin/` to `PATH` for *just this command*
3. Runs your command with the venv's interpreter

So you never activate. You never forget to activate. Each command is hermetic. CI runs the exact same `uv run pytest` you run locally.

Compared to the older `virtualenv + pip + source activate` workflow, `uv run` rolls up env management, package install, and command execution into one fast Rust binary. It's the modern Python tooling stack collapsed into one tool.

When to use plain `python`: rarely, in a `uv`-managed project. Quick scratch (`python -c "print(1+1)"`) is fine. For project work, always `uv run`.

Interview-ready version:

> *"`uv run` is the modern equivalent of activating a virtual environment. It checks the lockfile against the venv, installs missing deps, and runs the command with the venv's interpreter on PATH — all in one step. Every command runs in a validated, reproducible environment without manual activation."*

## Appendix D — The `@property` decorator

`@property` turns a method into something callers access *like a plain attribute*. No parentheses.

```python
class Embedder:
    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

# usage:
emb.dimension      # → 384       ← no parens
emb.dimension()    # ✗ TypeError
```

Why use it:

- **The value is a *fact* about the object, not an action.** Attribute-style access signals "query about state"; method-style signals "do something." Property = no parens for facts; method = parens for actions.
- **The value is computed, not stored.** We don't `self.dimension = …` in `__init__` — we ask the underlying model each time. The property hides this computation behind an attribute interface.

The deeper principle is the **uniform access principle**: whether a value is stored or computed should be invisible to the caller. You start with a plain attribute; if you later need lazy computation, caching, or validation, you swap it for a property *without breaking any callers*.

Use `@property` when: the value is a fact about the object, it's cheap to compute, no arguments, no side effects, read-only.

Don't use `@property` when: it does real work (disk I/O, network calls), it mutates state, it takes arguments, it might raise often.

Pandas (`df.shape`, `df.empty`), numpy (`arr.shape`, `arr.dtype`), pathlib (`path.name`, `path.parent`) — all use properties for "facts" and methods for "actions" (`path.exists()` is a method because it touches the filesystem).

## Appendix E — Subscription vs API access (and why `--echo` exists)

A common confusion: Claude.ai Pro ($20/mo) is a chat interface; the Anthropic API is pay-per-token programmatic access. They are different products and don't share quotas. The subscription does **not** give you API access — you need a separate API key with a billing wallet.

Cost math for this project is small. Per query: ~3,000 input tokens + ~300 output tokens.

| Model | Cost per query | Cost for 100 dev queries |
|---|--:|--:|
| Claude Haiku 4.5 | $0.0045 | **$0.45** |
| Claude Sonnet 4.6 | $0.0135 | **$1.35** |

For all of Week 1 dev, you'd spend $1-2 on API costs. The $5 free credit on signup covers all 4 weeks easily.

The `EchoClient` (`--echo` flag) is the **diagnostic / offline** path. It runs no LLM, makes no network call, costs nothing — just echoes back the source IDs from the prompt:

```bash
$ jai ask "anything" --provider echo
ECHO: answer derived from 3 sources: msg_..., msg_..., msg_...
```

It exists for three concrete reasons:

1. **Run tests for free.** Test suite uses `EchoClient`, costs $0, completes in <1s.
2. **Verify the pipeline before paying.** Confirm retrieval is finding the right chunks without an API key.
3. **Develop offline.** No internet, no key, no firewall — the pipeline still runs end-to-end.

Separating retrieval quality (which is real even with EchoClient) from generation quality (only visible with a real LLM) is the standard production-RAG diagnostic move. We built it into the CLI from day 1.

## Appendix F — The Protocol pattern (why adding OpenAI + Gemini was easy)

`LLMClient` is defined as a `typing.Protocol`:

```python
class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, ...) -> str: ...
```

Any class with a method of that signature satisfies the Protocol — no inheritance required. This is **structural typing**: ducks that quack get typed as ducks, mypy verifies the quack signature, and adding new providers costs ~25 lines each.

Adding OpenAI + Gemini after the Anthropic + Echo pair was built cost:

- **+2 small classes** in `generate.py` (~25 lines each)
- **+1 factory function** (`make_llm_client`)
- **0 changes** to `parse.py`, `chunk.py`, `embed.py`, `vector_store.py`, `format_prompt`, `generate_answer`, `Answer`, `Citation`
- **2 imports + 1 line** changed in `cli.py`

This is the **Open/Closed Principle**: open for extension (add providers), closed for modification (don't touch existing code). The Protocol was the seam.

CV-bullet version:

> *"LLM-provider-agnostic RAG pipeline behind a `typing.Protocol` interface. Production code depends on the Protocol, not the concrete SDK. Anthropic, OpenAI, Google Gemini, and a deterministic test double all interoperable; provider chosen at runtime via a single CLI flag."*

That sentence answers what hiring managers actually look for in LLM-engineering roles: not "can you call an API" but "did you build the seam so the API can change without breaking everything."

---

## Where to go next

**Week 2** lifts the ceiling on retrieval quality:

- BM25 alongside vector search, fused via Reciprocal Rank Fusion
- Cross-encoder reranker over the top-50 hybrid candidates
- A golden-set evaluation harness (RAGAS faithfulness + recall@k)
- Measurable improvement over the naive baseline (target: recall@10 0.65 → 0.91)

**Week 3** adds agentic routing: the LLM decides whether to use SQL, vector search, or plot generation. Closes the gap where naive RAG fails (count-style queries, comparisons, aggregations).

**Week 4** wraps it in FastAPI + Streamlit + Docker + deploy. Public synthetic-data demo, private real-data version, live URL you can put on your CV.
