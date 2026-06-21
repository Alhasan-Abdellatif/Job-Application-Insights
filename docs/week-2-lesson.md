# Week 2 lesson: hybrid retrieval, reranking, and the science of evaluation

Week 1 built a complete naive RAG: parse → chunk → embed → store → retrieve → generate. End-to-end, working, useful. And one question made it look broken:

> *"Could you list the universities I have applied to?"*

Dense retrieval returned the most semantically-similar "Thank you for applying" emails it could find. None of them were from universities. The LLM said *"I don't know"* because — given the chunks shown — it didn't.

The natural reaction is *"we need a better retriever."* That's true, but it's not enough. The deeper truth this week is:

> **You can't improve a retriever you can't measure, and measuring retrieval well is harder than building it.**

Week 2 is the journey from "intuition about retrievers" to "numbers about retrievers" — through five days, four retrieval architectures, two metric redesigns, and one ground-truth corpus pivot. By the end the headline result is **Recall@8 = 0.82 (from a 0.37 baseline)**. But what you actually walk away with — the transferable skill — is *the eval discipline that produced that number*.

This lesson has the same shape as Week 1's: a six-part conceptual sweep, the "putting it all together" deliverable, vocabulary, and appendices for the deep-dives that came up.

---

## Part 1 — Why measure before improving

When you look at a single bad answer, every fix sounds plausible:

- *Maybe a bigger embedding model?*
- *Maybe smaller chunks?*
- *Maybe BM25 instead of dense?*
- *Maybe a reranker?*
- *Maybe HyDE?*
- *Maybe more LLM context?*

Each of those is a real technique. None of them is the right *next* answer without something you can run *before* and *after* each change. That something is an **evaluation harness**.

### The minimum useful harness

Three pieces, in this order:

1. **A golden set** — a small list of `(question, expected_answer)` pairs you trust. The expected answer is whatever ground truth makes sense for the system: chunk IDs that *should* be retrieved, or reference text the LLM *should* produce.
2. **Pure metrics** — functions that take a system's output, take the golden truth, and return a number. *Pure* means: no I/O, no model, no network. Just math on two collections.
3. **A runner** — code that holds (1) and (2) together: for each entry in (1), call the system, hand its output and the truth to (2), aggregate the results.

That's it. Once those three exist, every "should I change X?" question reduces to *"compute the metric with X off, then with X on, then compare."*

The thing that's *not* in this list is "decide what the system should be." That comes later. **Eval first; improvements after** — because without eval, every improvement is a vibe, and vibes don't compose.

### What we built on Day 1

- [`evals/metrics.py`](../src/job_application_insights/evals/metrics.py) — five pure functions: `precision_at_k`, `recall_at_k`, `reciprocal_rank` (MRR), `dcg_at_k`, `ndcg_at_k`. Each takes a ranked list + a set of relevant IDs + an integer K. Returns a float.
- [`evals/golden_set.py`](../src/job_application_insights/evals/golden_set.py) — `GoldenEntry` Pydantic model + JSONL loader/saver. Translates disk bytes to typed objects at the boundary. Inside the boundary, nothing's optional.
- [`evals/runner.py`](../src/job_application_insights/evals/runner.py) — the orchestration: `evaluate(golden_set, retrieve_fn, k) -> EvalReport`.

The shape that mattered most for the whole rest of the week was the **type seam** between the runner and any retriever:

```python
RetrievalFn = Callable[[str, int], list[str]]
```

A function from `(query, k)` to a list of chunk IDs. **That's the entire interface.** Every retriever in Week 2 — dense, BM25, hybrid, reranked — conforms to this type. The eval harness, the CLI, the report formatter, none of them care which retriever they're holding. That seam is what made it possible to compose them on Day 2-3 without changing the harness.

### Day-1 measurement

```
Dense BGE-small + Chroma cosine, k=8, 13 questions
                              ↓
                         R@8 = 0.202
```

That's the Week-1 baseline. Every subsequent change has to beat it — and we'd never have known the number without writing the harness first.

---

## Part 2 — BM25, the lexical retriever

BM25 ("Best Matching 25", from the 1990s) scores how well a document matches a query by **literal token overlap**, with smart adjustments for:

- **TF** (term frequency): a doc that says "qualcomm" three times beats one that says it once — but the boost saturates so spam can't game it.
- **IDF** (inverse document frequency): rare words matter more. "qualcomm" might appear in 4 of 7,000 docs — high IDF, every match counts. "thank" appears in 3,000 — near-zero IDF, matches don't move the needle.
- **Length normalisation**: a 50,000-char marketing email mentioning "qualcomm" twice doesn't dominate a 200-char ack mentioning it once.

You don't need the formula. You need this:

> **Rare tokens carry the signal.** Dense embeddings don't reliably encode rare tokens. BM25 does. That's the whole reason to add it.

### The dense / lexical contrast

| | Dense (Week 1) | BM25 |
|---|---|---|
| Good at | paraphrases, concepts | exact matches, proper nouns, acronyms |
| Bad at | rare tokens, names | synonyms, vocab mismatch |
| Speed | fast (HNSW index) | trivially fast (CPU) |
| Cost | model load + GPU optional | no model, pure Python |

You don't pick. You run both. That's what hybrid retrieval is.

### The tokenisation invariant

The single most important rule about BM25:

> **Whatever function tokenises the corpus must also tokenise the query.**

If they drift, BM25 silently breaks. We use the same `tokenize()` function for both, exposed as a public API so future code (synonym expansion, query rewriting) can call it too:

```python
def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())
```

That's the whole interface to the lexical layer. Lowercased ASCII alphanumerics. Coarse, but right for the *role* of BM25 in this pipeline: catch exact-token wins (`qualcomm`, `gsk`, `ETH`) that the dense model missed. We don't need stemming, stop words, ICU — we need rare-token recall.

This invariant generalises beyond BM25. Whenever two parts of a system have to agree on a transformation — embedding both sides, hashing both sides, signing both sides — **make the transformation a callable they both import**. Don't reimplement.

### Score scale heterogeneity (and why we widened `RetrievalResult.score`)

Cosine: `[-1, 1]`. BM25: unbounded, typically `[0, 30]`, dips slightly negative for small-corpus IDF edge cases. Cross-encoder logits (Day 3): anywhere from `-10` to `+10`.

Yesterday's `RetrievalResult.score` was `Field(..., ge=-1.0, le=1.0)`. Today we have to relax it:

```python
class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    chunk: Chunk
    score: float  # unbounded — different retrievers use different scales
```

This is structural foreshadowing for Part 3: **scores aren't comparable across retrievers**. Anything that fuses signals from multiple retrievers has to use ranks, not scores.

---

## Part 3 — Reciprocal Rank Fusion (RRF)

You have two ranked lists now — dense's top-50 and BM25's top-50. How do you merge them?

### Why score-weighted fusion fails

The naïve fix:

```python
score = 0.5 * cosine + 0.5 * bm25
```

That equation adds **inches to kilograms**. Cosine 0.78 plus BM25 12.4 is a meaningless 13.18. The constants 0.5 and 0.5 don't fix the units. You'd need per-query calibration, which is fragile and noisy.

### The trick — compare ranks, not scores

For each document `d`, sum over the retrievers that returned it:

$$RRF(d) = \sum_{r \in retrievers} \frac{1}{k + \text{rank}_r(d)}$$

where `k = 60` is the field-standard constant (Cormack, Clarke & Büttcher, 2009 — battle-tested for 15 years, never seriously beaten on comparable budgets).

What you get for free:

1. **Scale invariance.** Cosine and BM25 produce different scale ranks, but rank `1` is rank `1` in both. Ranks are integers; scores are noise.
2. **Rewards consensus.** A doc that both retrievers rank in their top-K gets credit from *both* sums. A doc only one saw gets credit from one. Consensus dominates.
3. **Parameter-free in practice.** One constant. Set it to 60. Move on.

Eight lines of code:

```python
def reciprocal_rank_fusion(rankings, rrf_k=60):
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
```

### `make_hybrid_retriever` — composition over inheritance

The hybrid retriever is *not* a new class. It's the **same `RetrievalFn` type** with two retrievers stuffed inside a closure:

```python
def make_hybrid_retriever(retrievers, *, fetch_k=50, rrf_k=60):
    def retrieve(query, k):
        rankings = [r(query, fetch_k) for r in retrievers]
        fused = reciprocal_rank_fusion(rankings, rrf_k=rrf_k)
        return [chunk_id for chunk_id, _ in fused[:k]]
    return retrieve
```

The `evaluate()` function doesn't know it's holding a hybrid. The CLI doesn't know. The metric functions don't know. They all see a `RetrievalFn`. That's the seam paying off.

Even better: tomorrow's reranker will also be wrapped in a `RetrievalFn`, and it can sit on top of `hybrid` exactly the same way. **Day 2 + Day 3 = three lines of composition**.

### Day-2 measurement (Week-1 corpus)

```
                  P     R     MRR   nDCG
Dense          0.135  0.202  0.269  0.210
BM25           0.077  0.178  0.240  0.189
Hybrid (RRF)   0.106  0.198  0.262  0.187
```

Means barely moved. **That's not the result the textbook predicts** — and Day 2 was the first hint that the issue wasn't the retriever, it was the eval (Part 5 is where we fix it).

Per-question wins were real though:

- **Q3 Imperial: 0.000 → 0.750** — exactly the proper-noun rescue BM25 promised.
- **Q12 Amazon: 1.000 → 0.250** — and exactly the cautionary tale: BM25 noise on "Amazon" (common in corpus, low IDF) dragged a perfect dense answer down.

The mean doesn't lie about the *average* but it hides where each retriever earns its keep.

---

## Part 4 — Cross-encoder reranking

The Day-3 question: *how do we get the precision back that hybrid lost on Amazon?*

### Bi-encoder vs cross-encoder

The Week-1 embedder (BGE-small) is a **bi-encoder**:

```
query  ──► encode ──► q_vec ─┐
                              ├── cosine
chunk  ──► encode ──► c_vec ─┘
```

The encoder sees query and chunk *independently*. The chunk vector can be pre-computed once and stored. Query time is microseconds. But the model never gets to *reason about the pair* — only "is this chunk close, in the abstract, to this query, in the abstract?"

A **cross-encoder** sees both together:

```
[query] [SEP] [chunk]  ──► encode  ──► score (one number)
```

Encoder attention runs over the concatenation. The model can reason: *"the query asks about applications; this chunk is an AWS billing alert; mismatch."* Or: *"the query asks about Amazon; this chunk is a Vimeo ad mentioning Amazon Music; match strength low."*

Trade-off: no pre-computation. Every `(query, chunk)` pair is one forward pass — ~10ms each on CPU. Doing it over 10,000 chunks is hopeless; doing it over 50 candidates is fine.

### Two-stage retrieval — wide cheap funnel, then precise filter

```
query
  ├──► dense ──► top-50 ─┐
  │                       ├──► RRF ──► top-50 fused ──► cross-encoder ──► top-K
  └──► BM25 ──► top-50 ─┘
       ↑           ↑              ↑                          ↑
       Day 1       Day 2          Day 2                      Day 3
```

Same pattern as web search. Same pattern as recommender systems. Same pattern as any IR system that has more candidates than it can afford to score precisely.

### The Reranker Protocol

```python
class Reranker(Protocol):
    def rerank(self, query: str, items: Sequence[tuple[str, str]]) -> list[tuple[str, float]]: ...
```

Structural typing. Anything with this method satisfies the Protocol — no inheritance, no registration. We use it twice:

- `CrossEncoderReranker` — wraps `sentence_transformers.CrossEncoder`, model defaults to `BAAI/bge-reranker-base` (~280 MB, ~10ms/pair on CPU). Same lineage as the Week-1 embedder.
- `StubReranker` — used in tests. Maps chunk_ids to pre-canned scores. Conforms to the Protocol structurally; no model load required.

Tests are deterministic. Swapping to a bigger reranker (`bge-reranker-v2-m3`) is a one-line change in the CLI factory. None of this would have been clean without the Protocol seam.

### Day-3 measurement (Week-1 corpus)

```
                       P     R     MRR   nDCG
Dense              0.135  0.202  0.269  0.210
BM25               0.077  0.178  0.240  0.189
Hybrid (RRF)       0.106  0.198  0.262  0.187
Rerank (CE)        0.125  0.195  0.269  0.177
```

**Means still didn't move.** That's the second time three different architectural changes left the headline number flat. By the end of Day 3 we'd built every retriever the original Week-2 plan said to build, and the eval was telling us none of them mattered.

That couldn't be right.

---

## Part 5 — Set-cover recall, and the pandas `nan` bug

The reason the means weren't moving wasn't the retrievers. It was three problems in the *eval* itself — each one quiet, each one wrong — that the careful per-question analysis had been silently telling us about for two days.

### Problem 1 — chunk-level recall undercounts aggregation answers

Q3 was *"Which companies invited me to an interview?"* — answered by emails from Hubert, InterDigital, Baringa, Generative Engineering, Allps, Flatgigs. Six companies.

We'd marked **one canonical chunk per company** as relevant. Six chunks total in the golden set's `relevant_chunk_ids`.

The retriever returned `msg_000817__c003` (InterDigital reply-thread chunk 3 of 5) and got 0 credit, because the canonical chunk we marked was `msg_000826__c000`. *We knew about InterDigital. The retriever found InterDigital. The chunk IDs didn't match.*

This is the **chunk-vs-answer** confusion. For an aggregation question, the *unit of relevance* is the **entity**, not the chunk. A retriever that surfaces *any* InterDigital chunk has correctly found "InterDigital" as an answer to "which companies invited me to interview?"

### The fix — answer groups, set-cover recall

The data shape changes from:

```python
relevant_chunk_ids: ["msg_000826__c000", "msg_000705__c000", "msg_000672__c000", ...]
```

to:

```python
answer_groups: [
    ["msg_000826__c000", "msg_000817__c000", ..., "msg_001108__c000"],  # InterDigital (any chunk works)
    ["msg_000705__c000", "msg_000853__c000", ...],                       # Hubert
    ["msg_000672__c000"],                                                # Baringa
    ...
]
```

Each inner list is **one answer**. A group is "found" the first time *any* chunk from it appears in the top-K. Recall becomes:

$$R@K = \frac{\#\{\text{groups with} \geq 1 \text{ chunk in top-}K\}}{\#\text{groups}}$$

This is the **set-cover** formulation. It's how Ragas (the industry RAG eval framework) computes its `context_recall` — for each claim in the answer, check whether the chunk that supports it was retrieved.

A factoid question — *"Did I apply to ALESAYI HOLDING?"* — has one answer group with one chunk: `[["msg_000048__c000"]]`. Recall is 1.0 if found, 0.0 if not. Identical to the old chunk-level metric for factoid questions. Backwards compatible.

### Problem 2 — `RetrievalResult.score` was too tight

Already covered in Part 2. The bound `ge=-1.0, le=1.0` from the cosine-only era was now wrong for BM25 and cross-encoder. Widened to unbounded `float`.

### Problem 3 — pandas `nan` leaking into embeddings

While auditing the data we found that empty-body emails like Qualcomm's "Successfully submitted application for Alhasan Abdellatif" had chunk text:

```
'Successfully submitted application for Alhasan Abdellatif\n\nnan'
```

That literal three-character string `"nan"` was there because:

1. CSV cell is empty.
2. pandas reads it as `NaN` (a float).
3. `str(NaN)` yields `"nan"`.
4. Our `_normalise()` function checked for None and pd.isna() — but didn't know about the *string* `"nan"`.

So every empty-body email was being embedded as `subject + "nan"`. Trace amount of noise per chunk. Probably not catastrophic on its own. But:

> When the eval drives you to look at *exactly which chunks the retriever returned and why*, you find things like this. You wouldn't find them by reading the parser code in isolation.

This is the eval doing its second job — surfacing data hygiene bugs that look invisible until you measure.

The fix:

```python
def _is_missing(value: Any) -> bool:
    if value is None: return True
    try:
        if pd.isna(value): return True
    except (TypeError, ValueError): pass
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped: return True
        if stripped.lower() == "nan": return True   # ← new
    return False
```

Three lines. Wipe Chroma. Re-ingest. The next eval run was on clean embeddings.

### Day-4 measurement (Week-1 corpus, post-fix)

```
                       P     R     MRR   nDCG
Dense              0.292  0.429  0.523  0.382
BM25               0.153  0.278  0.349  0.278
Hybrid (RRF)       0.250  0.411  0.569  0.424
Rerank (CE)        0.264  0.354  0.426  0.313
```

R@8 went from 0.202 to 0.429 on dense — **not because the retriever got better, but because the metric is finally measuring what the questions ask.**

Hybrid now shows a clear MRR/nDCG lead (consensus paying off). Reranker recovers most of hybrid's losses on dense-favoured questions. The signal is real.

But Q4 (rejections, 36 entities) is still 0 across the board. Q9 (academic jobs) is still 0. The eval is honest now but the *questions* are still asking the corpus to do things the corpus isn't shaped for.

---

## Part 6 — Pivoting the corpus

The Day-4 numbers were honest but they were still answering the wrong question: *"how well does this retriever do on a noisy, heterogeneous personal inbox?"* That's not a learnable signal. It's eval over noise plus signal.

### The insight

The user's notebook (`Applications.ipynb`) had already done the hard data-cleaning work — separating *application acknowledgments* from interviews, rejections, LinkedIn job suggestions, newsletters, AWS bills, and everything else. The output was `applications_unique.csv`: 2,246 confirmed acks, each labelled with a Company and (sometimes) a Role.

That's the corpus the eval should use. **Homogeneous content, entity-labelled, ground-truth-grounded by construction.**

### The Day-5 pivot

```bash
# Step 1 — filter the raw inbox CSVs against the notebook's ack list
uv run python scripts/filter_acks.py
# → data/synthetic/ack_only.csv  (2,173 rows, every one a labelled ack)

# Step 2 — wipe + re-ingest
rm -rf data/chroma
uv run jai ingest data/synthetic/ack_only.csv
# → 11,313 chunks indexed

# Step 3 — design 11 entity-grounded questions
# (each one keyed on a Company label)
```

The golden set went from "questions I tried to think up across a noisy inbox" to "questions about specific labelled entities the notebook already identified." Every entry has a definitive answer.

Six single-entity factoids (Edinburgh, GSK, MediaTek, Roku, Heriot-Watt, ALESAYI), three aggregations (universities, GSK roles, ML roles), one role-extraction (MediaTek), one temporal (Edinburgh first applied). Eleven entries.

### Day-5 measurement

```
                k=8                          k=20
            P     R     MRR   nDCG       P     R     MRR   nDCG
Dense    0.284  0.371  0.377  0.371     0.200  0.469  0.385  0.399
BM25     0.284  0.636  0.301  0.384     0.195  0.636  0.301  0.384
Hybrid   0.318  0.818  0.558  0.621     0.268  0.825  0.565  0.625
Rerank   0.500  0.727  0.655  0.672     0.327  0.727  0.655  0.672
```

Compared to Day-4 (same retrievers, different corpus + golden set):

| Metric | Day-4 best | Day-5 best | Change |
|---|---|---|---|
| Recall@8 | 0.429 (dense) | **0.818 (hybrid)** | **+0.39, +91%** |
| Recall@20 | 0.506 (rerank) | **0.825 (hybrid)** | **+0.32, +63%** |
| Precision@8 | 0.292 (dense) | **0.500 (rerank)** | **+0.21, +71%** |
| MRR | 0.569 (hybrid) | **0.655 (rerank)** | **+0.09, +15%** |
| nDCG@8 | 0.424 (hybrid) | **0.672 (rerank)** | **+0.25, +59%** |

The retrievers didn't change. The corpus did, and the questions did. **Most of the apparent retrieval improvement across the whole week came from the eval design, not the retriever.** That's the most important sentence in this lesson — read it twice.

---

## Putting it all together — the Week-2 deliverable

Five commits, one journey:

```
1cdaa91  Week 2 day 5  ack-only corpus + 11-question entity-grounded golden set
d6db68f  Week 2 day 4  set-cover recall + nan fix + curated golden set
1e14c61  Week 2 day 3  cross-encoder reranking
48c2e29  Week 2 day 2  BM25 + RRF hybrid retrieval
bff7b81  Week 2 day 1  evaluation harness + Week 1 baseline
```

The CLI surface that emerged:

```bash
# ingest a CSV
uv run jai ingest data/synthetic/ack_only.csv

# evaluate any of 4 retrievers at any K
uv run jai eval --retriever {dense,bm25,hybrid,rerank} -k 8

# ask the RAG using any retriever + any LLM provider
uv run jai ask "Did I apply to GSK?" --retriever rerank --provider gemini -k 8
```

The architectural seams:

```python
RetrievalFn   = Callable[[str, int], list[str]]    # any retriever
Reranker      = Protocol                            # any reranker
LLMClient     = Protocol                            # any provider
GoldenEntry   = Pydantic model                      # any (chunk-level or answer-group) ground truth
```

Each retriever in the CLI is one factory function in `cli._build_retriever`. Each LLM provider is one branch in `generate.make_llm_client`. Adding HyDE next month would be one more factory; the eval harness would not change.

---

## Final results (Day 5, ack-only corpus, 11-question golden set)

```
                k=8                          k=20
            P     R     MRR   nDCG       P     R     MRR   nDCG
Dense    0.284  0.371  0.377  0.371     0.200  0.469  0.385  0.399
BM25     0.284  0.636  0.301  0.384     0.195  0.636  0.301  0.384
Hybrid   0.318  0.818  0.558  0.621     0.268  0.825  0.565  0.625
Rerank   0.500  0.727  0.655  0.672     0.327  0.727  0.655  0.672
```

Headline numbers, defensible at an interview:

- **Hybrid Recall@8 = 0.82 vs Dense 0.37 baseline** → **2.2× improvement**, *+120% relative*
- **Reranker Precision@8 = 0.50 vs Dense 0.28 baseline** → **1.8× improvement**, *+76% relative*
- **Reranker MRR = 0.66 vs Dense 0.38** → *+74% relative*

The CV bullet:

> *Designed a hybrid retrieval pipeline (BM25 + BGE-small dense + RRF fusion, optional cross-encoder reranker) over ~2,200 application acknowledgment emails. Hybrid retrieval reached **Recall@8 = 0.82 (+120% over dense baseline)** on a hand-curated 11-question entity-grounded eval set with set-cover recall semantics.*

---

## What I learned that surprised me

1. **Each retriever has its own failure mode, predictable in advance.** Dense fails on rare proper nouns (Imperial, GSK as acronym). BM25 fails on vocabulary mismatch (academic ≠ research associate). RRF fails on noise-dilution (Amazon — many AWS billing emails compete with application acks). Cross-encoder fails on short, sparse chunks (single-line subject acks). You don't fix any of them by adding more components. You fix each by understanding *which question hits which failure mode* and routing around it. The architecture lets you compose; the eval tells you what to compose against.

2. **The aggregate mean hides the per-question reality.** Mean R@8 was identical between hybrid and dense on Day 2. Per-question, dense won Q12 (1.0 vs 0.25) and hybrid won Q3 (0.75 vs 0). The mean told us "no progress"; the per-question view told us "hybrid trades Amazon for Imperial." Always report both.

3. **The eval is the load-bearing thing.** Week 2 had four retriever changes. Each was 30-60 minutes of clean code. None of them moved the headline number until we fixed the eval. After fixing the eval, the headline moved from 0.20 → 0.82 — and most of that came from the corpus pivot in Day 5, not the retrievers. *Most of what looks like "I improved my retriever" in any real RAG project is actually "I improved my eval."*

4. **Set-cover recall is the right primitive for aggregation.** Chunk-level recall asks "did you find this specific chunk?" Set-cover recall asks "did you find this answer?" The second is what users care about. The first is what's easy to write. Most introductory RAG content uses chunk-level. Most production systems should use set-cover.

5. **Data hygiene bugs surface through eval, not unit tests.** The pandas `nan`-to-string leak passed all the existing unit tests because no unit test asserted what an empty-body chunk *looked like* in the store. The eval surfaced it via "wait, every Qualcomm chunk has `nan` at the end." Unit tests check invariants. Eval checks *empirical behaviour*. Both are necessary.

---

## New vocabulary you should now own

- **BM25**, TF, IDF, length normalisation
- **Sparse vs dense retrieval**
- **Tokenisation invariant** (one tokeniser, two call sites)
- **Hybrid retrieval**, Reciprocal Rank Fusion (RRF), `k=60`
- **Bi-encoder vs cross-encoder**
- **Two-stage retrieval** (wide cheap funnel → precise filter)
- **Reranker**, candidate set, top-N vs top-K
- **Answer groups**, **set-cover recall**, set-cover DCG
- **Score heterogeneity** (different retrievers, different scales — fuse on ranks not scores)
- **Recall@K, MRR, nDCG@K, Precision@K**
- **Faithfulness, answer relevance, context precision/recall** (Ragas vocabulary, for next steps)
- **LLM-as-judge** (the dominant enterprise pattern)
- **Set-cover insight** (the difference between *availability* and *sufficiency* of evidence)
- **Synthetic eval generation + critique filtering** (HF cookbook pattern)

---

## Conceptual sanity check

Try answering these out loud before moving on:

1. Why can't you fuse cosine and BM25 by weighted score addition?
2. What does *"k = 60"* in RRF actually do, intuitively?
3. Why is a bi-encoder used for retrieval and a cross-encoder for reranking, instead of the same model for both?
4. What does set-cover recall measure that chunk-level recall doesn't?
5. Why did the headline number move from 0.20 to 0.82 across Week 2, given the retrievers were only modestly changed?
6. Three retrievers each got 0 recall on Q4 (rejections, 36 entities) at k=8. Two reasons that could happen; only one is the retriever's fault — which?
7. The pandas `nan` bug wasn't caught by unit tests. Why not, and what kind of test *would* catch it?

---

## Appendix — deep-dives from implementation

### A. The boundary pattern, applied to the golden-set file

The same idea as Week 1's `Document` model, applied to the JSONL eval set. Code outside the boundary is hostile (could be malformed, could be stale, could be from yesterday's schema). Code inside the boundary is *trusted* — typed, validated, frozen.

- **Outside**: a JSONL file with one object per line, written by hand or by a Python script.
- **The boundary**: `load_golden_set(path)` — opens the file, parses each line, validates with Pydantic, raises with line numbers on failure.
- **Inside**: `GoldenEntry` — frozen Pydantic model. `.question` is always non-empty. `.answer_groups` is always validated for uniqueness across groups.

Once a row crosses the boundary, the rest of the project can rely on it. No defensive checks downstream. The runner doesn't validate `question` again; it was validated at the door.

Two implementation details worth seeing:

```python
# Line-numbered errors when a curator hand-edits the file:
try:
    obj = json.loads(line)
except json.JSONDecodeError as exc:
    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
```

```python
# Backwards-compat via @model_validator: exactly one of two fields is required.
@model_validator(mode="after")
def _require_one_form(self) -> "GoldenEntry":
    if self.relevant_chunk_ids is None and self.answer_groups is None:
        raise ValueError("entry must provide either relevant_chunk_ids or answer_groups")
    if self.relevant_chunk_ids is not None and self.answer_groups is not None:
        raise ValueError("entry must provide exactly one of ... not both")
    return self
```

These are *small*. They're the kind of thing that has to exist to make hand-curated data robust to typos and refactors. Worth budgeting for.

### B. The `RetrievalFn` seam — why it's the most important type in the project

```python
RetrievalFn = Callable[[str, int], list[str]]
```

That's not a class. It's not a protocol with methods. It's just a *type alias*: "a function that takes a query and a K, returns a list of chunk IDs."

Every piece of the eval harness — `evaluate()`, `format_report()`, the runner tests, the CLI — is written against this type. **None of them mention Chroma, BGE, embeddings, BM25, cross-encoders, or anything else.** They consume `RetrievalFn` and that's it.

| Week | Retriever | Lines changed in `evaluate()` / CLI |
|---|---|---|
| 1 | Dense (BGE + Chroma) | `make_dense_retriever(store, embedder)` |
| 2 day 2 | BM25 only | `make_bm25_retriever(BM25Index(chunks))` |
| 2 day 2 | Hybrid (BM25 + dense + RRF) | `make_hybrid_retriever([dense, bm25])` |
| 2 day 3 | Hybrid + reranker | `make_reranked_retriever(hybrid, reranker, lookup)` |

Each new retriever is a one-function factory that returns a `RetrievalFn`. The eval harness doesn't care. The CLI gets a `--retriever` flag and dispatches to the right factory.

You've seen this shape before — `LLMClient` in `generate.py` is the same idea on the generation side. **Type-driven seams** are how a codebase stays clean across multiple rounds of "swap in a better version."

The opposite — `evaluate(golden, store, embedder, bm25_index, reranker_model, ...)` — is the road to a 12-argument function and a god module. We did not take that road.

### C. The Reranker Protocol — structural typing in action

```python
class Reranker(Protocol):
    def rerank(self, query: str,
               items: Sequence[tuple[str, str]]) -> list[tuple[str, float]]: ...
```

Two implementations satisfy it:

```python
class CrossEncoderReranker:        # production
    def __init__(self, model_name="BAAI/bge-reranker-base", ...): ...
    def rerank(self, query, items): ...    # calls CrossEncoder.predict

class StubReranker:                # tests
    def __init__(self, scores_by_id: dict[str, float]): ...
    def rerank(self, query, items): ...    # looks up scores from the dict
```

`StubReranker` doesn't inherit from `Reranker`. Doesn't register anywhere. **Doesn't import from `rerank.py`.** It just has the right method signature, so Python's static type system (mypy) and runtime duck-typing both accept it.

The win: 14 fast unit tests on the reranker logic, without ever loading a 280 MB model. Without the Protocol, every test would either need the real model loaded (slow, network) or some hand-rolled mock framework (fragile). Structural typing in Python is *cheaper than mocks* for this kind of seam.

### D. The set-cover insight

The most important conceptual move of Week 2. Worth a paragraph in its own right.

**Old framing — chunk-level recall:**

> *Of the chunks I marked relevant, what fraction did the retriever surface?*

**New framing — set-cover recall:**

> *Of the **answers** to my question, what fraction did the retriever find evidence for?*

The shift from *"chunks I marked"* to *"answers to my question"* is the whole game. For a factoid question (*"Did I apply to ALESAYI?"*), they're identical: one answer, one chunk, recall is binary. For an aggregation question (*"Which companies invited me to interview?"*), they're radically different: the answer is a set of 6 companies, and finding *any* chunk for each company should give full credit.

The metric implementation is small:

```python
def recall_at_k_groups(retrieved, groups, k):
    top_k_set = set(retrieved[:k])
    hits = sum(1 for g in groups if g & top_k_set)
    return hits / len(groups)
```

Three lines of math. But the underlying *model of relevance* changed:

- The unit of relevance is now an "answer group", not a "chunk".
- A group is "found" iff at least one of its chunks is retrieved.
- Recall, MRR, nDCG all reformulate around this primitive.

This is the same shape as **Ragas's `context_recall`** — for each claim in the answer, was a supporting chunk retrieved? It's also the same as **set-cover** in classical IR. We arrived at it via the eval failing in a specific way; the literature arrived at it via theory. Both routes converge.

### E. Score heterogeneity (and why we widened `RetrievalResult.score`)

A side effect of building multiple retrievers: their scores aren't comparable.

| Retriever | Score range | Higher is | Calibrated? |
|---|---|---|---|
| Dense cosine | `[-1, 1]` | more similar | yes (cosine) |
| BM25 | unbounded, `[0, ~30]` typical | more lexical overlap | no |
| Cross-encoder | unbounded, `[-10, +10]` typical | more relevant | sometimes |

You **cannot** linearly combine these. You **can** sort within each one and combine the ranks.

That's why:

1. `RetrievalResult.score` had to lose its `[-1, 1]` bound.
2. Hybrid fusion uses RRF (rank-based), not weighted score sum.
3. The CLI `ask` command synthesises a rank-derived score for display so citations look ordered without lying about which scale produced them.

The general principle generalises beyond retrieval: **anytime you compose scoring systems, fuse on ranks, not values.** Scores are local. Ranks are universal.

### F. The pandas `nan`-to-string leak

A small bug, a big lesson.

```python
# Pandas reads an empty CSV cell as NaN (a float).
# str(NaN) → "nan"  (a three-character string)
# _normalise() checks for None and pd.isna() but not the string "nan"
# So every empty-body email gets the literal "nan" embedded.
```

The fix is three lines. The lesson is about *how* it surfaced:

- All unit tests passed.
- All Pydantic validators passed.
- The chunks were the right length and type.
- The cosine math was correct.
- The CLI's `ask` worked end-to-end.

The bug only became visible when the eval *showed us which chunks the retriever was actually returning* for a query like Q1 Qualcomm — and we read the chunk text and saw `"...nan"` at the end of every one.

**Eval is the second test suite.** Unit tests assert invariants of the code; eval surfaces empirical behaviour of the data passing through the code. You need both. Most data pipelines have only the first.

### G. The honest CV bullet

The numbers in the table are real. The percentages are arithmetic. The questions you'll be asked at interview are not about the numbers — they're about whether you understand the limits of them.

The strongest claim:

> *Doubled Recall@8 (0.37 → 0.82) and improved Precision@8 by 76% on an 11-question hand-curated retrieval eval over ~2,200 application acknowledgment emails, by adding BM25 + RRF hybrid retrieval and a BGE-reranker-base cross-encoder over a dense BGE-small baseline.*

What an interviewer will probe:

- *Eleven questions is small — is this statistically significant?* → No. The per-question rankings are consistent (hybrid > rerank > dense > BM25 on recall) which is more informative than the absolute mean. To get statistical rigor we'd want ~50-100 questions.
- *In-distribution or held-out eval?* → In-distribution. The golden set was built against the same corpus the retrievers index. This is a measurement, not a generalisation claim.
- *Did you compare against a strong baseline?* → BGE-small (`BAAI/bge-small-en-v1.5`) is a real production embedding model, not a strawman. The +120% is over that, not over a worse baseline cherry-picked to look bad.
- *Why doesn't reranker beat hybrid on recall?* → Reranker re-orders within the same 50 candidates hybrid retrieved. Top-K recall is capped at what hybrid would have surfaced; reranker improves precision and ordering, not the candidate set.

Answering those without flinching is the bar. If you can, the bullet is yours.

---

## What's not in this lesson (and where to go next)

Week 2 covered retrieval thoroughly. What it did *not* cover:

- **LLM-as-judge for answer correctness** — the enterprise default. Ragas, TruLens, LangSmith all measure this primarily. Would add a *reference answer* field to `GoldenEntry` and a `jai eval-answers` subcommand.
- **Parent-document retrieval** — passing the full email (not just the retrieved chunk) to the LLM. Affects user-facing answer quality, doesn't change our retrieval metrics. ~1 hour to implement.
- **Synthetic question generation** — HF cookbook style. LLM generates questions from sampled chunks, LLM-critic filters bad ones. Brings us from 11 questions to ~50 cheaply.
- **HyDE / query rewriting** — addresses the vocab-mismatch failure (academic ≠ research associate). Marginal otherwise on our corpus.
- **Configuration sweep** — chunk size × embedder × rerank. The seams are there; a 50-line script would automate it.
- **Production observability** — what real RAG systems live on (latency p95, user thumbs-up rate, cache hit rate).

Pick any one of those as Week 3 if you want to keep going on RAG. Or pivot to the agentic-routing / structured-query work the README mentions, which uses the same data but answers a different class of question.

Either way, the eval harness you built this week is portable. The retrievers can change, the data can change, the LLM can change. The shape of the evaluation — golden set, pure metrics, type-seam runner, set-cover recall — stays the same.

That's the deliverable. Numbers are temporary. Eval discipline is forever.
