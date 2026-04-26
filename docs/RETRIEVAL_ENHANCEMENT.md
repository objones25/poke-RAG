# A general-purpose plan for improving retrieval in poke-RAG

You're right to push back. The format-filter idea was a band-aid. Let me reframe around what actually limits a retriever's ceiling, and where the recent literature has converged.

## The conceptual framing first

The question "what makes a retriever robust" decomposes into four orthogonal concerns. Most of the literature I surveyed maps onto exactly these:

1. **What's in the chunk** — chunking and metadata. Determines what's _possible_ to retrieve. Cap on everything downstream.
2. **What gets searched** — query understanding (routing, transformation, decomposition). Determines whether the right index is even hit.
3. **What comes back** — candidate generation and ranking (hybrid search, fusion, reranking). The classic "retrieval" step.
4. **What gets used** — post-retrieval filtering and refinement (CRAG-style, sufficiency gates). Determines what context the generator actually sees.

Your current pipeline has #2 partially (router + HyDE), strong #3 (BGE-M3 hybrid + RRF + reranker), and almost nothing for #1 beyond `entity_name`/`entity_type` or for #4 beyond top-k truncation. That's why you can't easily explain failures — there are gaps in two of the four pillars.

The Garchomp issue is informative because it lives in pillar #1 (the chunk doesn't expose the format/generation it represents). Fixing it via a hardcoded "default to gen9" router rule is pillar #2 patchwork — it papers over a #1 deficiency for one entity class. The architectural fix is to expose richer metadata on every chunk and let multiple pillars use it. Same change, but it pays dividends across many query types, not just competitive moveset queries.

---

## Pillar 1 — Metadata enrichment

**Current state.** Your `RetrievedChunk` has `source`, `entity_name`, `entity_type`, `chunk_index`, `original_doc_id`. That's a solid skeleton but it's load-bearing: there's nowhere to put a generation, a tier, a move type, a stat block, a release year.

**What the literature says.** The Microsoft/Azure RAG enrichment guide and a recent arxiv paper (2512.05411) on metadata-enriched RAG converge on a consistent finding: **metadata-augmented chunks consistently outperform content-only baselines across chunking strategies**, and the gains compound when metadata is used both as a filter and as part of the embedding (TF-IDF weighted or prefix-fused embeddings). The 2026 freshness paper found a recency prior alone hits perfect freshness accuracy when you have a timestamp; without one you can't even start.

**What this means for you.**

- Add a `metadata: dict[str, Any]` field to `RetrievedChunk` (or, more typed, a Pydantic model with optional fields). Keep `entity_name` etc. as first-class because they're load-bearing in your existing code, but stop treating the schema as closed.
- For each source, extract the metadata that's _already in the text or filename_ during chunking. You're not adding work — you're surfacing structure that exists:
  - **smogon_data**: `format` (gen9ou, gen9uu, …), `generation` (int, 9), `tier` (OU, UU, …), `chunk_kind` ("overview" or "set"), `set_name`, plus parsed `tera_type`, `item`, `ability`, `nature` from set sections.
  - **pokeapi**: `entity_subtype` ("species" / "moves" / "encounters") from the `original_doc_id` prefix you already have. This is genuinely free.
  - **bulbapedia**: `entity_subtype` from filename, plus optionally an LLM-extracted `topic` field per chunk if you want to invest more later.
- The metadata enables three downstream things you don't currently have: **payload pre-filtering**, **recency-aware re-ranking**, and **slot-aware deduplication** in the assembler. None of those work without it.

**A note on Qdrant filtering performance.** From the Qdrant docs and the ACORN paper: pre-filtering breaks HNSW connectivity at low selectivity (small filtered subsets), which can hurt recall. Post-filtering risks running out of candidates. The right pattern for you is **soft constraints used in fusion / boosting**, not hard `must` clauses, except for the cases where the user explicitly named the constraint (e.g. "Garchomp in Gen 6 OU" → hard filter). This matters because a global hard filter on `format=gen9` would silently break historical queries.

---

## Pillar 2 — Query understanding

**Current state.** You have keyword routing (`QueryRouter`) and optional HyDE. Both are good baseline ideas; the literature has moved on.

**What the literature says.**

- **Adaptive-RAG (Jeong et al. 2024, NAACL)** trains a T5-Large classifier to route each query into one of three buckets: no-retrieval / single-step / iterative. The classifier is trained on automatically-derived labels from "did the simpler strategy answer this correctly?" — no human annotation. The 2026 RAGRouter-Bench follow-up confirmed that lightweight classifiers hit ~28% token savings vs always-iterative while matching its accuracy. The key insight isn't the model size — it's the **three-way decision shape**.
- **Query decomposition** (Self-Ask, IRCoT, RT-RAG, CompactRAG, NVIDIA's own RAG blueprint): for genuinely multi-hop questions, decomposing into sub-questions and retrieving per-subquery beats single-shot retrieval substantially. RT-RAG (Jan 2026) shows +7% F1 / +6% EM by decomposing into a tree structure, but a much cheaper version (Ammann's "Question Decomposition for RAG", ACL 2025) just generates parallel sub-queries with an LLM, retrieves for each, merges, and reranks. The cheaper version is the right place to start.
- **Single-shot HyDE has known weaknesses** the literature is explicit about: when a question is precise, HyDE hurts because the hallucinated answer drifts. Your two-pass threshold gating (raw-first, HyDE on low confidence) is actually a good design — keep it. The literature's lesson is that decomposition + per-sub HyDE is more robust than HyDE on the original query.

**What this means for you.**

- **Replace the binary "route to sources" model with a three-way strategy decision** (à la Adaptive-RAG): `direct_lookup` (entity + property mentioned), `single_step` (current pipeline), `decomposed` (LLM splits into sub-queries, you retrieve each, merge, rerank). You can start this rule-based using your existing keyword router, and graduate to a learned classifier later when you have enough labelled data from your eval harness.
- **Decomposition first, sources second.** Keep your existing `QueryRouter` — but apply it _to each sub-query_, not the whole query. This is the right composition: if a complex query touches both bulbapedia mechanics and pokeapi stats, decomposition naturally separates those concerns and your existing per-source routing handles each cleanly. This single change is probably the biggest robustness win for the multi-hop questions you've already tagged in your eval harness.
- **Constraint extraction is part of query understanding.** When a user says "Gen 9 OU sets for Garchomp," that's a structural query: `entity=garchomp, format=gen9ou, intent=competitive_set`. An LLM-based constraint extractor (one Gemma call, structured output) populates the metadata filter for pillar #3 and the validation criteria for pillar #4. This is what the FAIR-RAG (Oct 2025) "structured evidence assessment" module does, in essence — and it's also what your `entity_name` parameter on `/query` does today, just in a richer form.

---

## Pillar 3 — Candidate generation and ranking

**Current state.** BGE-M3 hybrid (dense + sparse, optionally ColBERT), Qdrant RRF, BGE reranker. This is genuinely strong — leave it alone for now. The recent literature has not moved past hybrid + reranker for the _base_ of the candidate stage.

**Where the literature does extend it.**

- **Recency / freshness priors.** The 2025 Grofsky paper and Ragie's production guide both implement the same shape: `score_final = α · score_semantic + (1 − α) · score_recency`, with `score_recency` a half-life decay or a step function over time buckets. The FRESCO paper (April 2026) is the contrarian: a uniform recency prior **hurts** on timeless queries, so you need the recency weight to be query-dependent. The clean pattern: extract a `temporal_intent` signal during pillar #2 (latest / current / specific-period / timeless) and use it to set α at query time.
- **Per-query reranker calibration.** This is on the same axis: if a query mentions "Gen 6," α flips and the recency prior inverts (or you swap to a hard filter on the metadata). This generalizes beyond Pokémon — it's the standard pattern for any domain with versioned content.
- **Slot-aware ranking.** SEAL-RAG (Dec 2025) introduces "fixed-budget evidence assembly": rather than expanding context until you find what you want, hold the slot count fixed and _replace_ low-utility chunks with better candidates. The replacement signal is whether the chunk fills a gap in the structured constraint set extracted in pillar #2. For you, this maps onto: if the constraint extractor said `format=gen9ou` and your top-5 contains 3 Gen 6 chunks plus 2 Gen 9 chunks, swap the Gen 6 ones out for Gen 9 candidates from the larger candidate pool. **You already retrieve 75 candidates and only show 5 — most of the cost of this is sunk.**

**What this means for you.** The candidate stage stays. Add two things on top:

1. A **recency/version score** computed from chunk metadata (pillar #1) and `temporal_intent` from query understanding (pillar #2), fused with the reranker score before final top-k selection.
2. A **slot-aware re-selection** step that uses the constraints extracted in pillar #2 to swap out chunks that violate them, drawing from the existing 75-candidate pool.

Neither requires new models. Both are pure post-processing on data you already have.

---

## Pillar 4 — Post-retrieval refinement (this is where CRAG fits)

**Current state.** None. Your `ContextAssembler` does deduplication and token-budget truncation; that's it.

**What the literature says.** CRAG (Yan et al. 2024, ICLR 2025) is _specifically_ a pillar-4 contribution. Rephrasing what it actually does, since the framing in popular write-ups muddles it:

- **Per-chunk scoring with a lightweight evaluator** — CRAG used a fine-tuned T5-Large, but the open-source reproduction (arxiv 2603.16169, March 2026) and most subsequent work just use the existing reranker score, since rerankers are exactly the right tool for "how relevant is this chunk to the query." You already have BGE reranker scores. You don't need a new model.
- **Three-action triage**: above upper threshold → trust and refine; below lower → discard and seek elsewhere; in between → refine + augment. The reproduction paper found that the **action-triage shape** is the durable contribution; the specific T5 evaluator is interchangeable.
- **Decompose-then-recompose** within accepted chunks: split each chunk into "knowledge strips" (~3 sentences each in the reproduction; CRAG's GitHub README also offers fixed-token and excerpt modes), score each strip against the query, drop low-relevance strips, recompose. This is what filters Gen-6-mention-strips out of an otherwise on-topic Garchomp chunk _without_ needing format metadata at all. It's the most general-purpose precision tool in the literature.

**What this means for you.** Add a `KnowledgeRefiner` between retrieval and generation:

1. **Action triage.** Reuse your BGE reranker score (you already compute it). Set `upper` and `lower` thresholds. Above upper: pass through to refinement. Below lower: drop. In between: pass through with a mark that the constraint extractor or generator should be conservative about it.
2. **Strip-level filtering for accepted chunks.** Split each accepted chunk into ~2–3 sentence strips, run BGE reranker on (query, strip) pairs, drop strips below a strip-level threshold, recompose in original order. This is **the** CRAG technique, and it directly addresses the failure mode where a chunk is on-topic overall but contains the wrong specifics.
3. **Sufficiency check** (lighter than FAIR-RAG's SEA, heavier than nothing): if the constraint extractor in pillar #2 said `entity=garchomp, format=gen9ou` and after refinement no surviving strip mentions "gen9" or "gen 9" — that's an explicit gap. Either trigger a second retrieval pass with the format as a hard filter, or surface the gap in the generator's prompt so it can say "context covers older formats; here's what's there."

The CRAG-style "search the web on insufficient" branch you'll want to skip unless/until you wire in a Bulbapedia/Smogon API. The internal-knowledge branch is where the value is for a closed-corpus system like yours.

---

## How this plays against your existing eval harness

Your `questions.yaml` already has the structure to validate this:

- `easy_lexical` and `long_tail` test pillar #1 (chunk metadata + entity extraction).
- `paraphrase`, `confusable`, `aggregation` test pillar #3 (ranking quality).
- `multi_hop` and the `requires_decomposition` flag directly test pillar #2 (decomposition routing). You've already bucketed these out of headline numbers — they're a free measurement target for the decomposition work.
- `variant` (`Mega Charizard X`, `Galarian Darmanitan Zen`) is the existing test for pillar #1 metadata exposure, _exactly_ the same shape as the Gen 9 / Gen 6 problem. If your variant queries fail, the format problem is the same problem.

This is real leverage. You can land each of these changes behind a feature flag and run `run_eval.py` to measure the delta per pillar.

---

## Suggested order of operations

I'd sequence it this way, but the order is justifiable from "biggest leverage per unit of work" — challenge any of it:

1. **Metadata enrichment in chunker + payload schema** (pillar #1). One PR. Touches `chunker.py`, `vector_store.py` (payload write), `types.py` (RetrievedChunk schema), `retriever.py` (parse new fields back). Requires a one-time index rebuild. Unblocks everything below — without it, pillars #3 and #4 have no signal to use beyond text similarity.

2. **CRAG-style strip-level refinement** (pillar #4). One module: `KnowledgeRefiner`. Reuses BGE reranker. No new dependencies. Drops in between `Retriever` and `Generator` in the pipeline. This is the highest-precision gain per line of code in the whole plan, because it works on the chunks you've already retrieved.

3. **Constraint extraction + recency/version-aware reranking** (pillar #2 → #3). Adds a Gemma call before retrieval to extract `{entity, format, temporal_intent, …}` as structured output. The output drives a soft-boost in the reranking stage based on metadata fields from step 1. This is where the Garchomp case finally gets fixed — but as a side effect of a general mechanism.

4. **Query decomposition for multi-hop** (pillar #2). Add a `decomposed` strategy alongside your existing single-step path. Use your `requires_decomposition` eval bucket to measure the gain. Sub-query retrieval results merge into the existing top-k pool before reranking.

5. **Adaptive routing classifier** (pillar #2). Once you have enough labelled queries from the eval harness — say 200+ across the difficulty/category buckets — train a lightweight classifier (or even just a confidence-thresholded LLM call) to predict the strategy: direct / single-step / decomposed. Defer this until #4 is working.

Steps 1–3 give you a system that is architecturally complete in the sense the literature now expects — every pillar has a real component. Steps 4–5 are about handling the long tail of hard queries.

---

## What I'd skip, and why

- **Self-RAG.** Requires retraining the generator with reflection tokens on a curated corpus. Compounds with your existing LoRA work. Doesn't address any failure mode that CRAG-style refinement doesn't already handle for a closed corpus.
- **Knowledge graph approaches** (HopRAG, SubQRAG, GraphRAG). These solve a problem you don't have — your corpus is already structured (Smogon by entity/format/set, PokeAPI by entity/property). The graph is implicit in the metadata. Building an explicit graph is overhead that pays off when entities are densely linked and queries traverse those links; for "what does Garchomp learn at level X" that's not the case.
- **Web search fallback.** CRAG's `Incorrect → web search` branch. Skip it for now; you don't have a Bulbapedia/Smogon API integration, and adding one is a sourcing problem, not a retrieval problem.
- **Sources I read but didn't recommend follow-up on:** SEAL-RAG and FAIR-RAG are state-of-the-art for _complex multi-hop_ benchmarks but their full architecture (entity ledgers, structured evidence assessment) is overkill for a domain where most queries are single-hop. Their core insights — gap analysis, fixed-budget assembly, sufficiency gating — show up in steps 2 and 3 above in lighter form.

---

## Post-implementation findings (Pillar 4 — CRAG)

The `KnowledgeRefiner` from step 2 above is now implemented and wired into both the sync and async pipelines. Two limitations were discovered during live testing that are worth fixing before relying on either feature.

### Limitation 1 — Confidence score measures retrieval relevance, not answer quality

**What's implemented.** `confidence_score` is `sigmoid(chunks[0].score)` — the sigmoid of the BGE reranker's raw logit for the top-ranked chunk.

**The problem.** Sigmoid maps all positive logits to > 0.5, and the reranker almost always assigns positive logits to topically related chunks. In practice this means confidence rarely drops below 0.7 regardless of whether the generated answer is actually correct or grounded. A query about gen6 sweepers that returns a hallucinated answer still reports 0.74 confidence. The score is accurate about retrieval match quality but says nothing about generation quality — these are different things and conflating them is misleading.

**Candidate fixes (not yet implemented):**

- **Score gap signal.** `confidence = sigmoid(chunks[0].score - chunks[1].score)`. A large gap between rank-1 and rank-2 indicates a clear best match; a small gap indicates uncertainty. This is query-adaptive without any tuning.
- **Calibrated threshold.** Run the eval harness with `--include-confidence` and fit a Platt scaling layer on top of the raw logit using `(reranker_score, correct_answer)` pairs. Expensive to build but precise.
- **Separate retrieval confidence from generation confidence.** Return both: retrieval confidence from the reranker, generation confidence from the model's token-level log-probabilities (already available from `generate()` output). Let callers decide which to surface.

### Limitation 2 — Constraint gap detection does not fire for well-indexed formats

**What's implemented.** `_check_sufficiency` extracts gen/tier keywords from the query (e.g. "gen6", "ou") and checks whether those strings appear anywhere in the surviving chunks' `.text` fields. If all keywords are present, `knowledge_gaps` is `None`.

**The problem.** Smogon chunks are generated from format sections named `gen6ou`, `gen9ou`, etc. The format label is embedded in the chunk text, so "gen6" and "ou" will appear in every gen6ou chunk regardless of the query. This means `knowledge_gaps` is structurally `None` for any query whose generation and tier are present in the index — which is almost every competitive query. The detector only fires when retrieval pulls cross-source chunks (bulbapedia, pokeapi) that carry no format labels, which is not the common case for competitive queries.

**Candidate fixes (not yet implemented):**

- **Check metadata fields, not text.** Replace the substring search with a comparison against `chunk.metadata["generation"]` and `chunk.metadata["tier"]`. These are populated during Smogon ingestion. A gap fires when the requested generation/tier is absent from the metadata of _all_ surviving chunks, independent of what the text happens to say.
- **Normalise constraint extraction.** The query "gen 6 OU" and the metadata field `generation=6, tier="OU"` need a shared normalisation step (strip whitespace, lowercase, map "gen 6" → 6). This is the same normaliser already used in `_extract_constraint_keywords`; it just needs to compare against structured fields rather than raw text.
- **Fallback to text search only for sources without metadata.** Bulbapedia and pokeapi chunks don't carry generation/tier metadata; text search is still appropriate for those. The fix is to branch on `chunk.source`: use metadata comparison for smogon, text search for everything else.
