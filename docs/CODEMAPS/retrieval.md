# Retrieval Subsystem Codemap

**Last Updated:** 2026-04-21  
**Entry Points:** `src/retrieval/__init__.py` → `BGEEmbedder`, `Retriever`, `BGEReranker`, `ContextAssembler`

## Overview

The retrieval subsystem implements a hybrid semantic + lexical RAG pipeline. It breaks data into semantically coherent chunks, encodes them with `BAAI/bge-m3` (dense 1024-dim + sparse token-id vectors), stores them in Qdrant across three source-specific collections, retrieves candidates via RRF fusion, reranks with `BAAI/bge-reranker-v2-m3`, and assembles a formatted context string for the LLM.

## Compatibility Layer

`src/retrieval/_compat.py` patches two APIs removed in **transformers 5.x** that **FlagEmbedding 1.3.5** still calls at import time. A third removal (`build_inputs_with_special_tokens`) is handled by inlining its logic inside the `prepare_for_model` shim. Without these shims, both `BGEEmbedder` and `BGEReranker` fail to import.

### Patched APIs

| API removed in transformers 5.x                         | Where FlagEmbedding calls it           | Shim strategy                                                                   |
| ------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------- |
| `transformers.utils.import_utils.is_torch_fx_available` | FlagEmbedding model initialization     | Probe `torch.fx` import; return `True`/`False`                                  |
| `PreTrainedTokenizerBase.prepare_for_model`             | BGE-M3 tokenizer pipeline              | Re-implement: combine IDs with special tokens, optionally truncate              |
| `build_inputs_with_special_tokens`                      | Called inside `prepare_for_model` shim | Inlined into `prepare_for_model` using XLMRobertaTokenizer RoBERTa pair pattern |

### XLMRobertaTokenizer special-token pattern

BGE-M3's reranker uses `XLMRobertaTokenizer` which follows the **RoBERTa pair encoding** convention:

- **Single sequence**: `[BOS(0)] + ids + [EOS(2)]`
- **Pair sequence**: `[BOS(0)] + ids1 + [EOS(2), EOS(2)] + ids2 + [EOS(2)]`

The shim reconstructs this pattern using `bos_token_id` and `eos_token_id` from the tokenizer instance.

### Import order

`_compat.py` must be imported **before** any FlagEmbedding import. `embedder.py` and `reranker.py` each start with `import src.retrieval._compat` to guarantee this. Do not remove these imports.

## Architecture

```text
Input (raw files from processed/)
  ↓
chunker.py (source-specific splitting)
  ├─ chunk_bulbapedia_doc: Title + recursive sentence/paragraph merge
  ├─ chunk_smogon_line: Extract name, split body, target 400 tokens
  └─ chunk_pokeapi_line: One line = one atomic fact, no split
  ↓
  [RetrievedChunk]: frozen dataclass with text, score, source, entity_name, entity_type, chunk_index, original_doc_id
  ↓
embedder.py (BGEEmbedder)
  → BGEM3FlagModel.encode(texts) → dense (n×1024) + sparse (token_id→weight)
  → EmbeddingOutput(dense, sparse)
  ↓
vector_store.py (QdrantVectorStore)
  ├─ create 3 collections: "bulbapedia", "pokeapi", "smogon"
  ├─ upsert: embed RetrievedChunks with their computed embeddings
  └─ search: hybrid dense+sparse RRF fusion with optional entity_name filter
  ↓
retriever.py (Retriever orchestrator)
  1. embed query with BGEEmbedder
  2. for each source: search vector_store (top_k=candidates_per_source, default 20)
  3. merge candidates across sources
  4. rerank all candidates with BGEReranker to top_k (default 5)
  5. return RetrievalResult(documents, query)
  ↓
reranker.py (BGEReranker)
  → FlagReranker.compute_score(query-doc pairs)
  → sort by score, return top_k
  ↓
context_assembler.py (ContextAssembler)
  → dedup by f"{original_doc_id}:{chunk_index}" (keep highest score)
  → format with source + entity metadata
  → fit to max_tokens budget (default 4096)
  → join with separator (default "\n\n---\n\n")
  ↓
Output (context string for generation pipeline)
```

## Key Types

### From `src/types.py` (shared)

```python
Source = Literal["bulbapedia", "pokeapi", "smogon"]
EntityType = Literal["pokemon", "move", "ability", "item", "format"]

@dataclass(frozen=True)
class RetrievedChunk:
    text: str                    # The actual content
    score: float                 # Similarity/relevance score
    source: Source               # Which data source
    entity_name: str | None      # Extracted entity (e.g., "Pikachu", "Thunder Shock")
    entity_type: EntityType | None  # Type of entity
    chunk_index: int             # 0 for atomic facts (pokeapi), >0 if split
    original_doc_id: str         # Traces back to source file + line/doc number

@dataclass(frozen=True)
class RetrievalResult:
    documents: tuple[RetrievedChunk, ...]  # Final ranked chunks (frozen tuple)
    query: str                             # The original query

class VectorIndexError(RetrievalError):
    """Raised when vector store unavailable or returns no results."""
```

### From `src/retrieval/types.py` (internal)

```python
@dataclass(frozen=True)
class EmbeddingOutput:
    dense: list[list[float]]       # Shape (n, 1024), one vector per text
    sparse: list[dict[int, float]]  # Token ID → weight per text
```

## Public API

### Protocols (abstract interfaces)

**`EmbedderProtocol`**

```python
def encode(texts: list[str]) -> EmbeddingOutput:
    """Embed batch of texts. Returns dense (n×1024) and sparse (token_id→weight)."""
```

**`VectorStoreProtocol`**

```python
def ensure_collections() -> None:
    """Create the three source collections if they don't exist."""

def upsert(
    collection: Source,
    documents: list[RetrievedChunk],
    embeddings: EmbeddingOutput,
) -> None:
    """Upsert documents with their precomputed embeddings into a collection."""

def search(...) -> list[RetrievedChunk]:
    """Hybrid search: Prefetch dense & sparse independently (limit=top_k*2 each),
    fuse with RRF (Reciprocal Rank Fusion), apply optional entity_name filter,
    limit final result to top_k. Payload parsing is partially resilient: text, source,
    chunk_index, original_doc_id require direct access (KeyError if missing);
    entity_name and entity_type use .get() with None defaults. Returns list of
    RetrievedChunk with scores.
    """
```

**`RerankerProtocol`**

```python
def rerank(
    query: str,
    documents: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Rerank documents by relevance to query. Returns new chunks with updated scores."""
```

**`RetrieverProtocol`**

```python
def retrieve(
    query: str,
    *,
    top_k: int = 5,
    sources: list[Source] | None = None,
) -> RetrievalResult:
    """Retrieve top_k chunks for query across specified sources.

    Raises RetrievalError on any failure. Never returns empty result silently.
    """
```

### Concrete Implementations

**`BGEEmbedder`**

```python
@classmethod
def from_pretrained(*, model_name: str, device: str) -> BGEEmbedder:
    """Load BGEM3FlagModel. device='cuda', 'cpu', or 'mps'.
    Sets use_fp16=True for cuda/mps, False otherwise.
    Suppresses "fast tokenizer" warnings.
    """

def encode(texts: list[str]) -> EmbeddingOutput:
    """Encode a batch. Returns empty EmbeddingOutput if texts is empty."""
```

**`QdrantVectorStore`**

```python
def __init__(client: Any) -> None:
    """Wrap an existing qdrant_client.QdrantClient."""

def ensure_collections() -> None:
    """Create collections with COSINE distance for dense, on-disk=False for sparse."""

def upsert(
    collection: Source,
    documents: list[RetrievedChunk],
    embeddings: EmbeddingOutput,
) -> None:
    """Upsert in batches of _UPSERT_BATCH_SIZE (100).
    Point IDs are deterministic: uuid5(NAMESPACE_URL, f"{original_doc_id}:{chunk_index}").
    Payload includes all RetrievedChunk fields.
    """

def search(...) -> list[RetrievedChunk]:
    """Hybrid search: Prefetch dense & sparse independently (limit=top_k*2 each),
    fuse with RRF (Reciprocal Rank Fusion), apply optional entity_name filter,
    limit final result to top_k. Payload parsing is partially resilient: text, source,
    chunk_index, original_doc_id require direct access (KeyError if missing);
    entity_name and entity_type use .get() with None defaults. Returns list of
    RetrievedChunk with scores.
    """
```

**`Retriever`**

```python
def __init__(
    *,
    embedder: EmbedderProtocol,
    vector_store: VectorStoreProtocol,
    reranker: RerankerProtocol,
    candidates_per_source: int = _DEFAULT_CANDIDATES_PER_SOURCE,  # 20
) -> None:
    """Inject dependencies (enables mocking in tests)."""

def retrieve(
    query: str,
    *,
    top_k: int = 5,
    sources: list[Source] | None = None,
) -> RetrievalResult:
    """1. Embed query (raises EmbeddingError if empty or fails).
    2. Validate embedding: assert len(embedding.dense) == 1 before using.
    3. Search each active source for candidates_per_source items.
    4. Merge candidates (may exceed top_k at this stage).
    5. Rerank to top_k.
    Raises RetrievalError if no candidates found or reranking fails.
    """
```

**`BGEReranker`**

```python
@classmethod
def from_pretrained(*, model_name: str, device: str) -> BGEReranker:
    """Load FlagReranker. device='cuda', 'cpu', or 'mps'."""

def rerank(
    query: str,
    documents: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Compute relevance scores via FlagReranker.compute_score().
    Sort descending by score, return top_k.
    Uses dataclass.replace() to update scores while keeping documents frozen.
    """
```

**`ContextAssembler`**

```python
def __init__(
    *,
    max_tokens: int = _DEFAULT_MAX_TOKENS,        # 4096
    separator: str = _DEFAULT_SEPARATOR,          # "\n\n---\n\n"
) -> None:
    """Configure token budget and separator."""

def assemble(chunks: list[RetrievedChunk]) -> str:
    """1. Dedup by f"{original_doc_id}:{chunk_index}" (keep chunk with highest score).
    2. Preserve order from input.
    3. Format each as '[Source: {source}' + optional ' | Entity: {entity_name}' + ']\n{text}'.
       (Entity line is omitted if entity_name is None.)
    4. Accumulate until max_tokens exceeded.
    5. Truncate last chunk to fit budget.
    6. Join with separator.
    """
```

### Chunking Functions (script-level, also exported)

```python
def chunk_pokeapi_line(
    line: str,
    *,
    doc_id: str,
    entity_type: EntityType | None = None,
) -> list[RetrievedChunk]:
    """No splitting: one line = one atomic fact (pokeapi format).
    Returns single-element list. Extracts entity_name via _extract_pokeapi_name().
    """

def chunk_smogon_line(
    line: str,
    *,
    doc_id: str,
    entity_type: EntityType | None = None,
) -> list[RetrievedChunk]:
    """Split 'Name (tier): body...' by sentences/paragraphs to target 400 tokens.
    Extracts entity_name from 'Name (tier)' prefix.
    Returns list of RetrievedChunk with chunk_index 0..N.
    """

def chunk_bulbapedia_doc(
    doc: str,
    *,
    doc_id: str,
    entity_type: EntityType | None = None,
) -> list[RetrievedChunk]:
    """Split 'Title: ...\nbody...' to target 512 tokens.
    Extracts entity_name from title, ignoring parentheses.
    Returns list of RetrievedChunk with chunk_index 0..N.
    """

def chunk_file(path: Path, *, source: Source) -> list[RetrievedChunk]:
    """Dispatch to source-specific chunker for an entire file.
    Infers entity_type from path.stem (e.g., 'pokemon.txt' → 'pokemon').
    For bulbapedia, splits on 'Title:' boundaries first, then chunks each doc.
    Logs total chunks created.
    """
```

## Configuration & Constants

| Constant                                 | Value                               | Meaning                                 |
| ---------------------------------------- | ----------------------------------- | --------------------------------------- |
| `_DENSE_DIM`                             | 1024                                | BGE-M3 dense vector dimension           |
| `_DENSE_VECTOR_NAME`                     | "dense"                             | Qdrant vector field name                |
| `_SPARSE_VECTOR_NAME`                    | "sparse"                            | Qdrant sparse vector field name         |
| `_SOURCES`                               | ("bulbapedia", "pokeapi", "smogon") | All source collection names             |
| `_UPSERT_BATCH_SIZE`                     | 100                                 | Points per Qdrant upsert call           |
| `_SMOGON_TARGET_TOKENS`                  | 400                                 | Smogon chunk target (approx)            |
| `_BULBA_TARGET_TOKENS`                   | 512                                 | Bulbapedia chunk target (approx)        |
| `_OVERLAP_RATIO`                         | 0.1                                 | Chunk overlap as fraction of target     |
| `_WORDS_PER_TOKEN`                       | 0.75                                | Heuristic: 1 token ≈ 0.75 words         |
| `_DEFAULT_CANDIDATES_PER_SOURCE`         | 20                                  | Pre-reranking candidates per collection |
| `_DEFAULT_MAX_TOKENS` (ContextAssembler) | 4096                                | Max context token budget                |
| `_DEFAULT_SEPARATOR` (ContextAssembler)  | "\n\n---\n\n"                       | Chunk separator in context string       |

## Data Flow: Detailed Steps

### 1. Indexing (build_index.py)

Input: `processed/bulbapedia/`, `processed/pokeapi/`, `processed/smogon/`

```text
For each source:
  For each file in source directory:
    chunk_file(path, source) → list[RetrievedChunk]
      (source-specific chunking logic, see above)

    Batch chunks by source, embed with BGEEmbedder.encode()

    vector_store.upsert(source, chunks, embeddings)
      (creates deterministic point IDs via uuid5)
```

### 2. Query Retrieval (Retriever.retrieve)

Input: query string

```text
1. embedder.encode([query])
   → EmbeddingOutput(dense=[...], sparse=[...])

2. For each source in active_sources:
     vector_store.search(
       collection=source,
       query_dense=embedding.dense[0],
       query_sparse=embedding.sparse[0],
       top_k=candidates_per_source,  # default 20
     )
   Accumulate candidates across sources (may exceed top_k)

3. reranker.rerank(query, candidates, top_k)
   → Sort by reranker score, return top_k

4. Return RetrievalResult(documents=tuple(reranked), query=query)
```

### 3. Context Assembly (ContextAssembler.assemble)

Input: list[RetrievedChunk]

```text
1. Dedup by f"{original_doc_id}:{chunk_index}" (keep chunk with highest score per key)
2. Preserve order from input list
3. Format each chunk:
   "[Source: {source}" + (optional " | Entity: {entity_name}") + "]\n{text}"
4. Accumulate while total tokens < max_tokens
5. Truncate last chunk if needed to fit budget
6. Join with separator
Output: context_string for LLM prompt
```

## External Dependencies

| Package         | Version               | Purpose                                       |
| --------------- | --------------------- | --------------------------------------------- |
| `FlagEmbedding` | (from pyproject.toml) | BGEM3FlagModel + FlagReranker                 |
| `qdrant-client` | (from pyproject.toml) | Vector DB client, models for distance/vectors |
| `torch`         | (from pyproject.toml) | Implicit dependency of FlagEmbedding          |
| `transformers`  | (from pyproject.toml) | Model loading, tokenization                   |

## Key Design Decisions

1. **Protocol-driven architecture**: All concrete classes implement abstract protocols. Enables unit tests to inject mocks (e.g., `MockEmbedder`, `MockVectorStore`) without loading real models.

2. **Frozen dataclasses**: `RetrievedChunk`, `RetrievalResult`, `EmbeddingOutput` are immutable. Enables safe deduplication and prevents accidental mutation.

3. **Hybrid dense + sparse search**: Qdrant `query_points()` with dual `Prefetch` (dense + sparse, limit=top_k\*2 each) + `Fusion.RRF`. RRF (Reciprocal Rank Fusion) avoids per-collection weight tuning; works across all sources equally.

4. **Source-specific collections**: One Qdrant collection per source (`bulbapedia`, `pokeapi`, `smogon`). Enables source-specific payload filters (e.g., `entity_name` to restrict Pokédex lookups to just Pokémon entries).

5. **Deterministic point IDs**: `uuid5(NAMESPACE_URL, f"{original_doc_id}:{chunk_index}")` ensures idempotent upserts. Re-running indexing does not duplicate points.

6. **Token-based chunking**: Uses rough heuristic `len(text.split()) / 0.75` to estimate tokens. Merges segments with ~10% overlap to preserve context across boundaries.

7. **Two-pass reranking**: Retriever fetches `candidates_per_source × num_sources` candidates (default 20 per source = 60 total), then reranks to final `top_k` (default 5). Balances recall (more initial candidates) with precision (reranker refines).

8. **Fail-fast retrieval**: Any exception in embedding, search, or reranking raises `RetrievalError`. Generator layer must never be called on failure—documented in error class docstring.

## Error Handling

```python
class RetrievalError(Exception):
    """Base: retrieval failed, don't call generator."""

class EmbeddingError(RetrievalError):
    """Embedder returned empty or failed."""

class VectorIndexError(RetrievalError):
    """Qdrant unavailable, malformed payload, or no results."""
```

Retriever.retrieve() catches all exceptions and re-wraps as `RetrievalError` or subclass.

## Testing Strategy

All classes are testable via protocol injection:

```python
# Mock embedder
class MockEmbedder:
    def encode(self, texts):
        return EmbeddingOutput(
            dense=[[0.1] * 1024 for _ in texts],
            sparse=[{1: 0.5} for _ in texts],
        )

# In test
retriever = Retriever(
    embedder=MockEmbedder(),
    vector_store=MockVectorStore(),
    reranker=MockReranker(),
)
result = retriever.retrieve("test query")
assert result.documents
```

Chunking functions are pure (no I/O), so fully unit-testable. See `tests/unit/retrieval/` for examples.

## Related Areas

- **Generation pipeline** (`src/generation/`) consumes `RetrievalResult` and passes `ContextAssembler.assemble()` output to LLM.
- **RAG orchestration** (`src/pipeline/`) chains Retriever + ContextAssembler + Generator.
- **Scripts** (`scripts/build_index.py`) uses chunking functions and embedder to populate vector store.
