# Pipeline, API & Scripts Codemap

**Last Updated:** 2026-04-23

**Entry Points:**

- `src/pipeline/rag_pipeline.py` — RAGPipeline orchestrator
- `src/api/app.py` — FastAPI application
- `scripts/build_index.py` — Index builder CLI
- `src/api/dependencies.py` — build_pipeline() factory

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                       FastAPI App                           │
│                      (src/api/app.py)                       │
│                                                              │
│  Lifespan: build_pipeline() → app.state.pipeline            │
│  Routes: GET /health, POST /query                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ get_pipeline() extracts from request.app.state
                     │
                     ▼
          ┌──────────────────────┐
          │   RAGPipeline        │
          │ (src/pipeline/)      │
          │                      │
          │ query() orchestrates │
          │ retrieval + gen      │
          └────┬────────┬────────┘
               │        │
        ┌──────▼─┐      └──────────────────────┐
        │Retriever│                            │Generator
        │         │                            │
        │ Embedding                           │Inference
        │ Vector Search                       │Prompt Building
        │ Reranking                           │Model Loading
        │         │                            │
        └─────────┴────────────────────────────┘
                 │
    ┌────────────┴──────────────┐
    │                           │
    ▼                           ▼
Qdrant (3 collections)    HuggingFace Models
(bulbapedia, pokeapi,   (embedder, generator,
 smogon)                  reranker)

Build Index Script (scripts/build_index.py):
discover_files() → chunk_all_files() → embed_in_batches()
  → group_by_source() → vector_store.ensure_collections()
  → upsert() per source
```

---

## RAG Pipeline

### RAGPipeline Class

**Location:** `src/pipeline/rag_pipeline.py`

The core orchestrator: accepts a query, retrieves context, and passes it to the generator.

**Constructor:**

```python
RAGPipeline(
    *,
    retriever: RetrieverProtocol,
    generator: GeneratorProtocol,
    query_router: QueryRouterProtocol | None = None,
) -> None
```

**Key Invariant:** If `retriever.retrieve()` raises `RetrievalError`, the generator is never called. The exception propagates immediately.

**Query Router Integration:** If `query_router` is set and `sources` is not explicitly specified in the request, the router classifies the query and selects the target sources. This enables keyword-based source routing (e.g., "stats" → pokeapi only, "tier" → smogon only).

**query() Method:**

```python
def query(
    self,
    query: str,
    *,
    top_k: int = 5,
    sources: list[Source] | None = None,
    entity_name: str | None = None,
) -> PipelineResult
```

- Validates `query` is non-empty/non-whitespace; raises `ValueError` if not
- If `sources` is None and `query_router` is set, calls `query_router.route(query)` to determine sources
- Calls `retriever.retrieve(query, top_k=top_k, sources=sources, entity_name=entity_name)`
- If retrieval succeeds, calls `generator.generate(query, chunks)`
- Computes confidence score: `sigmoid(top_chunk.score)` if chunks exist, else None
- Deduplicates and sorts sources from chunks: `tuple(sorted({c.source for c in chunks}))`
- Returns `PipelineResult` with answer, sources used, chunk count, model name, query, and confidence_score

**Raises:**

- `ValueError`: if query is empty or whitespace-only
- `RetrievalError` (or subclasses): propagated immediately from retriever; generator never called

### PipelineResult Type

**Location:** `src/pipeline/types.py`

Immutable output of a single RAG query:

```python
@dataclass(frozen=True)
class PipelineResult:
    answer: str                      # Generated answer from the model
    sources_used: tuple[Source, ...]  # Unique sources in alphabetical order
    num_chunks_used: int             # Count of retrieved chunks passed to generator
    model_name: str                  # Name of the generation model used
    query: str                        # Original query string
    confidence_score: float | None = None  # Sigmoid of top-ranked chunk's score; None if unavailable
```

---

## Shared Types

**Location:** `src/types.py`

### RetrievedChunk

```python
@dataclass(frozen=True)
class RetrievedChunk:
    text: str                  # Chunk text content
    score: float               # Relevance score from search/reranking
    source: Source             # "bulbapedia" | "pokeapi" | "smogon"
    entity_name: str | None    # Extracted Pokémon/move/item name if available
    entity_type: EntityType | None  # "pokemon" | "move" | "ability" | "item" | "format"
    chunk_index: int           # Position within the original document
    original_doc_id: str       # ID of the source document
```

### RetrievalResult

```python
@dataclass(frozen=True)
class RetrievalResult:
    documents: tuple[RetrievedChunk, ...]  # Retrieved chunks
    query: str                              # Original query
```

### GenerationResult

```python
@dataclass(frozen=True)
class GenerationResult:
    answer: str              # Generated text
    sources_used: tuple[Source, ...]  # Sources present in chunks
    model_name: str          # Model identifier
    num_chunks_used: int     # Number of chunks in context
```

### Type Literals

```python
Source = Literal["bulbapedia", "pokeapi", "smogon"]
EntityType = Literal["pokemon", "move", "ability", "item", "format"]
```

### Exception Hierarchy

```python
class RetrievalError(Exception):
    """Raised when retrieval fails. Generator must never be called."""

class EmbeddingError(RetrievalError):
    """Raised when embedding model fails."""

class VectorIndexError(RetrievalError):
    """Raised when vector index is unavailable or returns no results."""
```

---

## FastAPI Application

### App Initialization

**Location:** `src/api/app.py`

**Lifespan Handler:**

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    setup_logging()
    try:
        pipeline, loader, client = build_pipeline()
        app.state.pipeline = pipeline
        app.state.qdrant_client = client
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize RAG pipeline: {exc}") from exc
    try:
        yield
    finally:
        loader.unload()
        # (Loky executor shutdown)
```

- Calls `setup_logging()` to configure root logger
- Calls `build_pipeline()` (see below) to construct and wire the full pipeline
- Stores **3-tuple**: `(RAGPipeline, ModelLoader, QdrantClient)`
- Stores pipeline in `app.state.pipeline` and client in `app.state.qdrant_client`
- On shutdown: calls `loader.unload()` to clear model from memory
- Wraps build errors as `RuntimeError` with context
- Startup failure prevents the app from starting

**Middlewares:**

1. **LatencyTrackingMiddleware**: Adds `X-Response-Time-Ms` header with elapsed time in milliseconds
2. **RateLimitMiddleware**: 20 requests per minute per IP (sliding window). Disabled via `RATE_LIMIT_ENABLED=false`

**CORS:**

- `ALLOWED_ORIGINS` env var controls allowed origins (default `"*"`)
- If `"*"`, allow all origins; otherwise split by comma and strip whitespace

**Exception Handlers:**

1. **RetrievalError** → HTTP 503 (Service Unavailable)

   ```python
   @app.exception_handler(RetrievalError)
   async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
       return JSONResponse(status_code=503, content={"detail": str(exc)})
   ```

2. **ValueError** → HTTP 422 (Unprocessable Entity)

   ```python
   @app.exception_handler(ValueError)
   async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
       return JSONResponse(status_code=422, content={"detail": str(exc)})
   ```

### HTTP Endpoints

#### GET /health

```python
@app.get("/health")
def health() -> dict[str, str]:
```

**Response:**

```json
{ "status": "ok" }
```

Simple health check; requires no dependencies.

#### GET /stats

```python
@app.get("/stats")
def stats(request: Request) -> dict[str, bool]:
```

**Response:**

```json
{
  "bulbapedia": true,
  "pokeapi": true,
  "smogon": true
}
```

Returns a dict of collection names to `True` (indicating they exist). Extracts `qdrant_client` from `request.app.state` and calls `client.get_collections()`.

#### POST /query

```python
@app.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse
```

**Request Body (QueryRequest):**

```python
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    sources: list[Literal["bulbapedia", "pokeapi", "smogon"]] | None = None
    entity_name: str | None = Field(None, max_length=50)  # Optional Pokémon name filter
```

**Response (QueryResponse):**

```python
class QueryResponse(BaseModel):
    answer: str                      # Generated answer text
    sources_used: list[str]          # List (not tuple) of unique sources
    num_chunks_used: int             # Count of chunks passed to generator
    model_name: str                  # Model identifier
    query: str                        # Parsed query that was processed
    confidence_score: float | None = None  # Sigmoid of top chunk's reranker score; None if unavailable
```

**Behavior:**

- `QueryRequest.query` is validated by Pydantic (min_length=1, max_length=2000)
- `QueryRequest.entity_name` is optional; validated against alphanumeric + hyphens/underscores/apostrophes/spaces
- Calls `parse_query(body.query)` to strip and re-validate
- Calls `pipeline.query(normalized_query, sources=body.sources, entity_name=body.entity_name)`
- Converts `PipelineResult` fields to `QueryResponse` (tuple → list for sources)
- Includes `confidence_score` from `PipelineResult` (may be None)
- On `ValueError`: returns HTTP 422 with error detail
- On `RetrievalError`: returns HTTP 503 with error detail

---

## Dependencies & Pipeline Factory

**Location:** `src/api/dependencies.py`

### get_pipeline()

```python
def get_pipeline(request: Request) -> RAGPipeline:
```

FastAPI dependency that extracts the pipeline from request state. Raises `RuntimeError` if not initialized (should not happen if lifespan succeeded).

### build_pipeline()

```python
def build_pipeline() -> tuple[RAGPipeline, ModelLoader, QdrantClient]:
```

**Returns:** 3-tuple of (RAGPipeline, ModelLoader, QdrantClient).

**Orchestration:**

1. Load settings from environment via `Settings.from_env()`:
   - `QDRANT_URL` (required)
   - `QDRANT_API_KEY` (optional)
   - `EMBED_MODEL`, `RERANK_MODEL`, `GEN_MODEL` (with defaults)
   - All generation parameters: temperature, max_new_tokens, top_p, do_sample, etc.
   - `DEVICE` (auto-detected from torch: cuda → mps → cpu)

2. **Retrieval Pipeline:**

   ```python
   embedder = BGEEmbedder.from_pretrained(
       model_name=settings.embed_model,
       device=settings.device,
   )
   reranker = BGEReranker.from_pretrained(
       model_name=settings.rerank_model,
       device=settings.device,
   )
   client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
   vector_store = QdrantVectorStore(client)
   vector_store.ensure_collections()
   
   # Optional query transformation (HyDE)
   if settings.hyde_enabled:
       query_transformer = HyDETransformer(
           inferencer,  # Created below
           max_new_tokens=settings.hyde_max_tokens
       )
   else:
       query_transformer = None
   
   retriever = Retriever(
       embedder=embedder,
       vector_store=vector_store,
       reranker=reranker,
       query_transformer=query_transformer,
   )
   ```

3. **Generation Pipeline:**

   ```python
   gen_config = GenerationConfig(...)       # from Settings
   tok_config = TokenizerConfig(...)        # from Settings
   loader = ModelLoader(config=gen_config, device=settings.device)
   loader.load()                            # Loads Gemma 4 model
   inferencer = Inferencer(                 # Wraps inference
       model=loader.get_model(),
       processor=loader.get_processor(),
       config=gen_config,
   )
   generator = Generator(
       loader=loader,
       prompt_builder=build_prompt,         # Callable from generation.prompt_builder
       inferencer=inferencer,
       config=gen_config,
   )
   ```

4. **Optional Query Router (keyword-based source routing):**

   ```python
   if settings.routing_enabled:
       query_router = QueryRouter()
   else:
       query_router = None
   ```

5. **Combine into RAGPipeline and return 3-tuple:**

   ```python
   pipeline = RAGPipeline(
       retriever=retriever,
       generator=generator,
       query_router=query_router,
   )
   return pipeline, loader, client
   ```

**Dependencies Wired:**

- Embedder: BGE-M3 for dense+sparse embeddings
- Vector Store: Qdrant with 3 collections (bulbapedia, pokeapi, smogon)
- Reranker: BGE Reranker v2-m3 for ranking retrieved chunks
- Generator: Gemma 4 via HuggingFace Transformers
- Processor: Paired with generator model (Gemma 4 uses AutoProcessor, not a tokenizer)
- Prompt Builder: Callable that formats query + context into model input
- Loader: ModelLoader instance (stored for cleanup during lifespan shutdown)
- Client: QdrantClient instance (stored in app.state for /stats endpoint)

---

## Query Parser

**Location:** `src/api/query_parser.py`

```python
def parse_query(query: str) -> str:
    """Strip surrounding whitespace and validate the query is non-empty."""
```

- Strips leading/trailing whitespace
- Raises `ValueError` if the result is empty
- Returns normalized query string

Note: This performs the same validation as `RAGPipeline.query()`, providing defense-in-depth.

---

## Logging Setup

**Location:** `src/utils/logging.py`

```python
def setup_logging(level: str | None = None) -> None:
    """Configure the root logger. Reads LOG_LEVEL env var when level is not given."""
```

**Configuration:**

- Level: `level` parameter, or `LOG_LEVEL` environment variable, or default "INFO"
- Format: `"%(asctime)s %(levelname)-8s %(name)s — %(message)s"`
- Timestamp: ISO 8601 with time component (`"%Y-%m-%dT%H:%M:%S"`)
- Output: `sys.stdout`
- `force=True`: Reconfigures root logger even if handlers exist (safe for reinitialization)

Called once at app startup by the lifespan handler.

---

## Build Index Script

**Location:** `scripts/build_index.py`

Embeds Pokémon data from `processed/` and upserts into Qdrant vector index. Supports checkpointing for resumable indexing.

### CLI Interface

```bash
uv run python scripts/build_index.py \
  [--source bulbapedia|pokeapi|smogon] [--source ...] \
  [--batch-size 32] \
  [--dry-run] \
  [--checkpoint PATH] \
  [--no-checkpoint]
```

**Arguments:**

- `--source` (repeatable): Index only specified source(s). Default: all three (bulbapedia, pokeapi, smogon)
- `--batch-size` (int): Embedding batch size. Default: 32
- `--dry-run` (flag): Log what would be indexed without writing to Qdrant
- `--checkpoint PATH`: Path to checkpoint JSON file. Default: `.build_index_checkpoint.json`
- `--no-checkpoint` (flag): Disable checkpointing; re-index everything

### Main Workflow (main())

1. Load environment variables
2. Parse command-line arguments
3. Validate `Settings.from_env()` (requires `QDRANT_URL`)
4. Initialize `BGEEmbedder` and `QdrantVectorStore` with settings
5. Call `run()` with:
   - embedder
   - vector_store
   - sources
   - processed_dir = `src/../processed`
   - batch_size
   - dry_run
   - checkpoint_path

### Orchestration: run()

```python
def run(
    *,
    embedder: BGEEmbedder,
    vector_store: QdrantVectorStore,
    sources: tuple[Source, ...],
    processed_dir: Path,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
    checkpoint_path: Path | None = None,
) -> None
```

**Checkpoint Load/Save:**

- Loads existing checkpoint JSON (set of completed file keys) or returns empty set
- Checkpoint format: JSON array of strings, e.g. `["bulbapedia/pokemon.txt", "pokeapi/moves.txt"]`
- File key: `"{source}/{filename}"`
- Checkpoint is **only saved after successful upsert** to prevent saving on failure
- On error reading checkpoint, logs warning and starts fresh

**Discovery & Filtering:**

1. `discover_files(processed_dir, sources)` → list of (source, Path) tuples
   - Scans `processed/{source}/` for `.txt` files
   - Excludes files ending in `_aug.txt` (paraphrased variants, separate indices)
   - Returns sorted file list
2. Filters to remaining files not in checkpoint: `[(src, p) for src, p in files if f"{src}/{p.name}" not in completed]`
3. Logs progress: files discovered, already indexed, to process

**Embedding & Upsert (per file):**

1. `chunk_file(path, source=source)` → list of RetrievedChunk
2. For each batch in chunks (stride = batch_size):
   - Extract texts: `[c.text for c in batch]`
   - Call `embedder.encode(texts)` → EmbeddingOutput(dense, sparse)
   - Validate output: `len(result.dense) == len(batch) and len(result.sparse) == len(batch)`
   - If mismatch: raise RuntimeError with counts
   - **If dry_run:** log `[dry-run] would upsert N chunk(s) into '{source}'`
   - **If not dry_run:** call `vector_store.upsert(source, batch, result)`
3. After all batches, append file_key to completed set
4. If checkpoint_path: persist checkpoint
5. Log completion

**Collections:**

- Before first upsert (if not dry_run): calls `vector_store.ensure_collections()`
- Creates 3 Qdrant collections if they don't exist: "bulbapedia", "pokeapi", "smogon"
- One collection per source; upsert targets collection named by Source

**Logging:**

- `_LOG = logging.getLogger(__name__)`
- Info: discovery summary, per-file processing, batch upserts (debug), completion
- Warning: missing source directories, checkpoint read errors
- Error (via exit 1): missing required env vars

---

## Helper Functions in build_index.py

### discover_files()

```python
def discover_files(
    processed_dir: Path,
    sources: tuple[Source, ...],
) -> list[tuple[Source, Path]]:
```

**Returns:** List of (source, file_path) tuples for all `.txt` files in each source directory, excluding `_aug` variants.

**Behavior:**

- Iterates through sources
- Checks if `processed_dir/{source}/` exists; warns and skips if not
- Globs `*.txt` files, filters to `not p.stem.endswith("_aug")`
- Sorts results
- Returns early if any source dir missing (non-fatal)

### chunk_all_files()

```python
def chunk_all_files(
    files: list[tuple[Source, Path]],
) -> list[RetrievedChunk]:
```

**Returns:** Flat list of all chunks from all files.

**Behavior:**

- Iterates over (source, path) pairs
- Calls `chunk_file(path, source=source)` for each
- Extends global chunk list
- **Not used in current run()** (see note below)

_Note: Current `run()` chunks files one-at-a-time rather than chunking all upfront. This helper is available but the main loop handles chunking inline._

### embed_in_batches()

```python
def embed_in_batches(
    embedder: BGEEmbedder,
    chunks: list[RetrievedChunk],
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> EmbeddingOutput:
```

**Returns:** Single `EmbeddingOutput(dense=[], sparse=[])` combining all batches, or empty output if no chunks.

**Behavior:**

- Returns empty output if chunks list is empty
- Iterates in batches: `range(0, len(chunks), batch_size)` with slicing
- Extracts texts from batch: `[c.text for c in batch]`
- Calls `embedder.encode(texts)` → EmbeddingOutput
- Validates count match: `len(result.dense) == len(batch) and len(result.sparse) == len(batch)`
- Raises RuntimeError if mismatch
- Extends global dense/sparse lists
- Returns combined EmbeddingOutput
- **Not used in current run()** (inline embedding happens during file processing)

### group_by_source()

```python
def group_by_source(
    chunks: list[RetrievedChunk],
    embeddings: EmbeddingOutput,
) -> dict[Source, tuple[list[RetrievedChunk], EmbeddingOutput]]:
```

**Returns:** Dictionary mapping each Source to a tuple of:

- List of RetrievedChunks for that source
- EmbeddingOutput (dense + sparse) for those chunks only

**Behavior:**

- Returns empty dict if chunks list is empty
- Iterates through chunks with index
- Groups chunks by `chunk.source`
- Appends chunk and its corresponding embeddings (dense[idx], sparse[idx]) to source bucket
- Reconstructs source-wise EmbeddingOutput objects
- Returns dict with one entry per source present
- **Not used in current run()** (inline grouping via single-file processing per source)

---

## Control Flow: A Typical Query

1. **Client sends POST /query:**

   ```json
   {
     "query": "What are Pikachu's stats?",
     "sources": ["pokeapi"]
   }
   ```

2. **FastAPI Dependency Injection:**
   - `get_pipeline(request)` extracts `request.app.state.pipeline` (set during lifespan)

3. **Request Handler:**
   - `parse_query(body.query)` strips/validates
   - `pipeline.query(normalized, sources=["pokeapi"], top_k=5)`

4. **RAGPipeline.query():**
   - Validates query non-empty
   - If sources not specified and query_router is set: calls `query_router.route(query)` to determine sources
   - Calls `retriever.retrieve(query, top_k=5, sources=["pokeapi"], entity_name=...)`
     - If query_transformer set (e.g., HyDE): transforms query before embedding
     - BGEEmbedder encodes (transformed or original) query → dense + sparse vectors
     - QdrantVectorStore.search() on "pokeapi" collection using hybrid search (parallel)
     - BGEReranker reranks top results
     - Returns RetrievalResult with top_k chunks
   - Calls `generator.generate(query, chunks)`
     - Builds prompt with context via `build_prompt()`
     - Loads/infers Gemma 4 with temperature=0.7, max_tokens=512
     - Returns GenerationResult
   - Computes confidence_score = sigmoid(top_chunk.score)
   - Deduplicates sources from chunks → `("pokeapi",)`
   - Returns PipelineResult with confidence_score

5. **Response Handler:**
   - Converts PipelineResult to QueryResponse
   - Returns JSON

6. **Error Cases:**
   - ValueError (empty query) → HTTP 422
   - RetrievalError (embedding fails, index down) → HTTP 503
   - Other exceptions → FastAPI default 500

---

## Environment Configuration

**Location:** `src/config.py`

### Settings Class

```python
@dataclass(frozen=True)
class Settings:
    # Vector store
    qdrant_url: str                # QDRANT_URL (required)
    qdrant_api_key: SecretStr | None  # QDRANT_API_KEY (optional, masked in logs)

    # Model names
    embed_model: str               # EMBED_MODEL (default: "BAAI/bge-m3")
    rerank_model: str              # RERANK_MODEL (default: "BAAI/bge-reranker-v2-m3")
    gen_model: str                 # GEN_MODEL (default: "google/gemma-4-E4B-it")

    # Generation hyperparameters
    temperature: float             # TEMPERATURE (default: 0.7)
    max_new_tokens: int            # MAX_NEW_TOKENS (default: 512)
    top_p: float                   # TOP_P (default: 0.9)
    do_sample: bool                # DO_SAMPLE (default: true)

    # Tokenizer config
    tokenizer_max_length: int      # TOKENIZER_MAX_LENGTH (default: 8192)
    return_tensors: str            # RETURN_TENSORS (default: "pt")
    truncation: bool               # TRUNCATION (default: true)

    # Device
    device: str                    # DEVICE (auto-detected if not set)

    # Optional: Fine-tuning
    lora_adapter_path: str | None  # LORA_ADAPTER_PATH (optional, path to adapter)

    # Optional: Query transformation (HyDE)
    hyde_enabled: bool             # HYDE_ENABLED (default: false)
    hyde_max_tokens: int           # HYDE_MAX_TOKENS (default: 150)

    # Optional: Query routing
    routing_enabled: bool          # ROUTING_ENABLED (default: false)
```

**from_env()** classmethod:

- Raises `KeyError` if `QDRANT_URL` missing
- Auto-detects device: cuda → mps → cpu
- Returns frozen Settings instance

---

## Dependencies & Protocols

### RetrieverProtocol

```python
class RetrieverProtocol(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        sources: list[Source] | None = None,
    ) -> RetrievalResult:
        """Retrieve top_k chunks. Raises RetrievalError on failure."""
```

Implemented by: `src/retrieval/retriever.py:Retriever`

### GeneratorProtocol

```python
class GeneratorProtocol(Protocol):
    def generate(
        self,
        query: str,
        chunks: tuple[RetrievedChunk, ...],
    ) -> GenerationResult:
```

Implemented by: `src/generation/generator.py:Generator`

### QueryRouterProtocol

```python
class QueryRouterProtocol(Protocol):
    def route(self, query: str) -> list[Source]:
        """Classify query into one or more sources via keyword patterns.
        Returns sorted list; never empty (fallback to all sources if no match)."""
```

Implemented by: `src/retrieval/query_router.py:QueryRouter`

---

## File Organization Summary

| File                           | Purpose                                                                     |
| ------------------------------ | --------------------------------------------------------------------------- |
| `src/pipeline/rag_pipeline.py` | RAGPipeline orchestrator class                                              |
| `src/pipeline/types.py`        | PipelineResult dataclass                                                    |
| `src/types.py`                 | Shared types: RetrievedChunk, RetrievalResult, GenerationResult, exceptions |
| `src/api/app.py`               | FastAPI app, lifespan, exception handlers, endpoints                        |
| `src/api/dependencies.py`      | get_pipeline(), build_pipeline() factory                                    |
| `src/api/models.py`            | QueryRequest, QueryResponse Pydantic models                                 |
| `src/api/query_parser.py`      | parse_query() validator                                                     |
| `src/utils/logging.py`         | setup_logging() configuration                                               |
| `src/config.py`                | Settings dataclass, from_env()                                              |
| `scripts/build_index.py`       | Index builder: discover, chunk, embed, upsert; checkpointing                |

---

## Related Codemaps

- **Retrieval**: `docs/CODEMAPS/retrieval.md` — Embedding, vector search, reranking
- **Generation**: `docs/CODEMAPS/generation.md` — Model loading, inference, prompt building
