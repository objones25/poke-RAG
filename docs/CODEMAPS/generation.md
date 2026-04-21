# Generation Subsystem Codemap

**Last Updated:** 2026-04-21
**Entry Points:** `src/generation/__init__.py`

## Overview

The generation subsystem loads the Gemma 4 model, builds prompts from retrieved context, and runs inference to produce grounded answers to Pokémon queries. It operates in three stages: (1) load the model and processor once via `ModelLoader`, (2) build a formatted prompt from query + chunks via `build_prompt()`, (3) run inference via `Inferencer`, producing a `GenerationResult` with the answer and metadata.

The subsystem is orchestrated by `Generator`, which wires together the loader, prompt builder, and inferencer. All components follow immutable dataclass patterns and use protocol-based abstraction to enable testing with mock builders and models.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Generator (main orchestrator)                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ┌────────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│ │ ModelLoader        │  │ PromptBuilder    │  │ Inferencer      │  │
│ │                    │  │                  │  │                 │  │
│ │ • load()           │  │ • build_prompt() │  │ • infer()       │  │
│ │ • get_model()      │  │   formats query  │  │   builds msgs   │  │
│ │ • get_processor()  │  │   + chunks into  │  │   runs generate │  │
│ │ • unload()         │  │   LLM prompt     │  │   decodes       │  │
│ │                    │  │                  │  │                 │  │
│ │ AutoModel...       │  │ Sorts by score   │  │ model.generate()│  │
│ │ AutoProcessor      │  │ Formats context  │  │ token extraction│  │
│ │                    │  │ Builds Sources   │  │                 │  │
│ └────────────────────┘  └──────────────────┘  └─────────────────┘  │
│         ↓                       ↓                       ↓            │
│   torch, device handling   context_block          torch ops        │
│   dtype selection          header format          batch extraction  │
│   cache management         system prompt          token stripping   │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
                           GenerationResult
                         (answer, sources_used,
                          model_name, num_chunks_used)
```

## Public API

### Classes

#### `Generator`

Main orchestrator that wires loader + prompt builder + inferencer.

```python
class Generator:
    def __init__(
        self,
        loader: ModelLoader,
        prompt_builder: PromptBuilderProtocol,
        inferencer: Inferencer,
        config: GenerationConfig,
    ) -> None: ...

    def generate(
        self,
        query: str,
        chunks: tuple[RetrievedChunk, ...]
    ) -> GenerationResult:
        """
        Generate an answer to a query using retrieved context chunks.

        Args:
            query: User's question
            chunks: Retrieved context from vector index (non-empty)

        Returns:
            GenerationResult with answer, sources, model name, chunk count

        Raises:
            ValueError: if chunks is empty (retrieval must not fail before generation)
        """
```

**Implementation details:**

- Calls `prompt_builder(query, chunks)` to build the full prompt
- Passes prompt to `inferencer.infer()` to get answer text
- Extracts unique sources from chunks and sorts them
- Logs query length, chunk count, answer length, and sources used
- Returns immutable `GenerationResult`

#### `ModelLoader`

Manages lazy loading and unloading of Gemma 4 model + processor.

```python
class ModelLoader:
    def __init__(
        self,
        config: GenerationConfig,
        device: str
    ) -> None: ...

    def load(self) -> None:
        """
        Load model and tokenizer once. Idempotent — safe to call multiple times.

        Device-aware dtype selection:
        - CUDA (GPU): bfloat16 (efficient mixed precision)
        - MPS (Apple Silicon): float16 (Metal Performance Shaders compatible)
        - CPU: float32 (no fp16 support)

        Logs progress at INFO level.
        """

    def get_model(self) -> PreTrainedModel:
        """Return loaded model or raise RuntimeError if not loaded."""

    def get_processor(self) -> Any:
        """Return loaded processor or raise RuntimeError if not loaded."""

    def unload(self) -> None:
        """
        Clear model and processor references. Clears device cache if available.
        """
```

**Implementation details:**

- Uses `AutoModelForImageTextToText.from_pretrained()` + `AutoProcessor.from_pretrained()` — Gemma 4 is a vision-language model
- Uses `dtype=` kwarg (transformers 5.x); `torch_dtype=` was removed
- MPS path: omit `device_map`, load on CPU, call `.to("mps")` after load — avoids `caching_allocator_warmup` allocating ≥14.79 GiB on MPS
- CUDA/CPU path: passes `device_map="auto"`, `dtype=_dtype_for_device(device)`, `attn_implementation="sdpa"`
- Idempotent: checks `if self._model is not None` and skips if already loaded
- Stores model/processor as private attributes; access via getters
- `unload()` clears references and calls `torch.cuda.empty_cache()` on CUDA or `torch.mps.empty_cache()` on MPS

#### `Inferencer`

Builds message format, runs model.generate(), decodes and strips output.

```python
class Inferencer:
    def __init__(
        self,
        model: PreTrainedModel,
        processor: Any,
        config: GenerationConfig,
    ) -> None: ...

    def infer(self, prompt: str) -> str:
        """
        Run inference on a single prompt.

        Args:
            prompt: Full formatted prompt including system, context, question

        Returns:
            Generated answer text, stripped of leading/trailing whitespace

        Raises:
            ValueError: if prompt is empty
            RuntimeError: if model.generate() returns empty sequences
            TypeError: if processor.decode() returns non-str
        """
```

**Implementation details:**

- Builds messages list: `[{"role": "user", "content": prompt}]`
- Calls `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)`
- Tokenizes with `processor(text=text, return_tensors="pt")` and moves to model device
- Extracts `input_len` from `input_ids.shape[-1]` to slice output (exclude prompt tokens)
- Calls `model.generate(**inputs, max_new_tokens=..., temperature=..., top_p=..., do_sample=...)`
- Does NOT call `parse_response()` — Gemma 4's `parse_response()` returns a dict, not a str
- Decodes with `processor.decode(output_ids[0][input_len:], skip_special_tokens=True)`
- Type-checks that decoded output is str, raises TypeError otherwise
- Returns `response.strip()`

### Functions

#### `build_prompt(query, chunks) -> str`

Builds a formatted prompt from query and retrieved chunks.

```python
def build_prompt(
    query: str,
    chunks: tuple[RetrievedChunk, ...]
) -> str:
    """
    Format query + chunks into a prompt for the LLM.

    Args:
        query: User's question
        chunks: Retrieved context (must be non-empty, sorted by score desc)

    Returns:
        Formatted prompt string

    Raises:
        ValueError: if query or chunks empty
    """
```

**Prompt structure:**

```text
You are a knowledgeable Pokémon expert. Answer the question using only the context provided below. Cite your sources at the end of your answer.

Context:
[Source: bulbapedia | Entity: Pikachu]
Pikachu is an electric-type Pokémon...

[Source: pokeapi | Entity: Pikachu]
Stats: HP 35, ATK 55, DEF 40...

Sources: bulbapedia, pokeapi

Question: What are Pikachu's base stats?

Answer:
```

**Implementation details:**

- Strips newline characters (`\n`, `\r`) from user input before building prompt template
- Sorts chunks by score (descending) before formatting
- For each chunk, constructs a header: `[Source: X | Entity: Y]` if entity_name exists, else `[Source: X]`
- Joins chunks with `\n\n` separator
- Builds Sources line from unique sorted sources: `Sources: bulbapedia, pokeapi, smogon`
- System prompt is invariant (same for all queries)
- Leaves `Answer:` tag at the end for the model to complete

### Protocols

#### `PromptBuilderProtocol`

Callable protocol for prompt building. Enables duck typing and testing.

```python
@runtime_checkable
class PromptBuilderProtocol(Protocol):
    def __call__(
        self,
        query: str,
        chunks: tuple[RetrievedChunk, ...]
    ) -> str: ...
```

Used by `Generator` instead of concrete `build_prompt` function, allowing injection of mock builders in tests.

#### `GeneratorProtocol`

Callable protocol for generation. Enables duck typing and testing.

```python
@runtime_checkable
class GeneratorProtocol(Protocol):
    def generate(
        self,
        query: str,
        chunks: tuple[RetrievedChunk, ...]
    ) -> GenerationResult: ...
```

Used by pipeline orchestrators to accept any generator implementation.

## Config Dataclasses

### `GenerationConfig`

Controls LLM inference hyperparameters.

```python
@dataclass(frozen=True)
class GenerationConfig:
    model_id: str                    # HuggingFace model ID (e.g. "google/gemma-4-E4B-it")
    temperature: float = 0.7         # Sampling temperature (0.0–2.0)
    max_new_tokens: int = 512        # Max tokens to generate per inference
    top_p: float = 0.9               # Nucleus sampling (0.0–1.0)
    do_sample: bool = True           # Use sampling vs. greedy decoding
```

All fields frozen (immutable). Passed to `model.generate()` and stored in `Inferencer`.

## Key Behaviors

### Device-Aware Type Selection

`_dtype_for_device(device: str) -> torch.dtype` selects the optimal dtype:

| Device | dtype      | Rationale                                                                       |
| ------ | ---------- | ------------------------------------------------------------------------------- |
| `cuda` | `bfloat16` | NVIDIA AMPERE (A100, RTX 3090) and newer support bfloat16; reduces memory, fast |
| `mps`  | `float16`  | Apple Silicon (M1/M2/M3) Metal Performance Shaders supports float16 efficiently |
| `cpu`  | `float32`  | CPU does not support lower precision without performance loss                   |

Used by `ModelLoader.load()` to instantiate the model with the correct dtype.

### Idempotent Loading

`ModelLoader.load()` checks `if self._model is not None` and returns early if already loaded. This allows calling `load()` multiple times without reloading:

```python
loader = ModelLoader(config, "cuda")
loader.load()  # Downloads and loads model
loader.load()  # No-op: logs "already loaded — skipping"
```

### Token Extraction

`Inferencer.infer()` extracts only new tokens from `model.generate()` output:

```python
output_ids = model.generate(...)  # Shape: (1, prompt_len + new_len)
prompt_len = input_ids.shape[-1]  # e.g., 500 tokens
generated = output_ids[0][prompt_len:]  # Slice: (1, new_len)
```

This ensures the decoded output contains only the model's generated text, not the input prompt echoed back.

### Error Handling

**Retrieval failures must not reach Generator:**

- If retrieval raises `RetrievalError` or subclasses (`EmbeddingError`, `VectorIndexError`), the pipeline MUST NOT call `Generator.generate()`.
- `Generator.generate()` checks `if not chunks:` and raises `ValueError` to enforce this contract.

**Inference errors:**

- Empty tokenizer output: `ValueError("prompt must not be empty")`
- Model returns no sequences: `RuntimeError("Model generate() returned no sequences ...")`
- Tokenizer returns non-str: `TypeError("Tokenizer returned X, expected str")`

## Dependencies

| Package        | Version (from pyproject.toml) | Role                                                                  |
| -------------- | ----------------------------- | --------------------------------------------------------------------- |
| `transformers` | Core (pinned)                 | `AutoModelForImageTextToText`, `AutoProcessor`, type definitions      |
| `torch`        | Core                          | Tensor operations, device management, dtype selection, cache clearing |
| `accelerate`   | Core                          | Device mapping, mixed precision support (used by transformers)        |

**Note:** `google/gemma-4-E4B-it` must be downloaded and cached locally or via HuggingFace hub. Model loading uses HuggingFace cache by default.

## Gemma 4 Model Loading

**Critical:** Gemma 4 is a vision-language model. Load with `AutoModelForImageTextToText` + `AutoProcessor`.

```python
# CORRECT
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained("google/gemma-4-E4B-it")
processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")

# WRONG — this is for causal LM models (Gemma 2)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it")
# This will fail or produce wrong model class
```

Always verify via Context7 HuggingFace docs before writing any model-loading code — the API changes between versions.

## Module Dependency Graph

```text
src/generation/
├── __init__.py
│   └─ exports: Generator, GeneratorProtocol, GenerationConfig, TokenizerConfig, build_prompt
│
├── models.py
│   ├─ GenerationConfig (frozen dataclass)
│   └─ TokenizerConfig (frozen dataclass)
│
├── protocols.py
│   ├─ PromptBuilderProtocol
│   └─ GeneratorProtocol
│
├── loader.py
│   ├─ imports: GenerationConfig, torch, transformers (AutoModelForImageTextToText, AutoProcessor)
│   ├─ _dtype_for_device() helper
│   └─ ModelLoader class
│
├── inference.py
│   ├─ imports: GenerationConfig, torch, transformers (PreTrainedModel)
│   └─ Inferencer class
│
├── prompt_builder.py
│   ├─ imports: RetrievedChunk (from src.types)
│   ├─ _SYSTEM_PROMPT (module constant)
│   └─ build_prompt() function
│
└── generator.py
    ├─ imports: ModelLoader, Inferencer, PromptBuilderProtocol, GenerationConfig
    ├─ imports: GenerationResult, RetrievedChunk (from src.types)
    └─ Generator class (orchestrator)

External imports:
- src.types: RetrievedChunk, GenerationResult, GenerationError (not raised in generation/)
- transformers: AutoModelForImageTextToText, AutoProcessor, PreTrainedModel
- torch: Tensor, device, dtype, cuda.empty_cache(), mps.empty_cache()
```

## Integration Points

### Input: `RetrievedChunk`

From `src/types.py`, passed to `build_prompt()` and `Generator.generate()`:

```python
@dataclass(frozen=True)
class RetrievedChunk:
    text: str                       # Document text
    score: float                    # Relevance score (0.0–1.0, higher better)
    source: Source                  # "bulbapedia" | "pokeapi" | "smogon"
    entity_name: str | None         # "Pikachu", "Earthquake", etc.
    entity_type: EntityType | None  # "pokemon" | "move" | "ability" | "item" | "format"
    chunk_index: int                # Position in original document
    original_doc_id: str            # Document ID from Qdrant
```

### Output: `GenerationResult`

From `src/types.py`, returned by `Generator.generate()`:

```python
@dataclass(frozen=True)
class GenerationResult:
    answer: str                     # Generated answer text
    sources_used: tuple[Source, ...]  # Unique sources in sorted order
    model_name: str                 # Model ID (e.g., "google/gemma-4-E4B-it")
    num_chunks_used: int            # Count of chunks passed to generator
```

## Testing Patterns

The protocol-based design enables easy mocking:

```python
# Mock prompt builder
def mock_builder(query: str, chunks: tuple[RetrievedChunk, ...]) -> str:
    return f"Query: {query}\nChunks: {len(chunks)}"

# Mock model (PreTrainedModel)
class MockModel:
    def generate(self, input_ids, **kwargs):
        return torch.tensor([[0, 1, 2, 3]])  # Mock output

# Inject into Inferencer or Generator
inferencer = Inferencer(mock_model, mock_tokenizer, config)
generator = Generator(loader, mock_builder, inferencer, config)
```

See `tests/unit/test_inference.py`, `tests/unit/test_generator.py` for example test coverage.

## Common Tasks

### 1. Initialize and Generate

```python
from src.generation import Generator, GenerationConfig
from src.generation.loader import ModelLoader
from src.generation.inference import Inferencer
from src.generation.prompt_builder import build_prompt
from src.types import RetrievedChunk

config = GenerationConfig(
    model_id="google/gemma-4-E4B-it",
    temperature=0.7,
    max_new_tokens=512,
)
loader = ModelLoader(config, device="cuda")
loader.load()  # One-time load

inferencer = Inferencer(
    loader.get_model(),
    loader.get_processor(),
    config,
)
generator = Generator(
    loader,
    build_prompt,  # Pass the function as PromptBuilderProtocol
    inferencer,
    config,
)

# Retrieve chunks from vector index (not shown)
chunks = (...)  # tuple of RetrievedChunk

# Generate answer
result = generator.generate("What are Pikachu's abilities?", chunks)
print(result.answer)
print(result.sources_used)
```

### 2. Change Generation Hyperparameters

Edit `GenerationConfig`:

```python
config = GenerationConfig(
    model_id="google/gemma-4-E4B-it",
    temperature=0.5,  # Lower = more deterministic
    max_new_tokens=256,  # Shorter answers
    top_p=0.95,  # Broader sampling
    do_sample=False,  # Greedy decoding (deterministic)
)
```

### 3. Switch Devices and Precision

The `ModelLoader` automatically selects the optimal dtype for your device:

````python
# CUDA: auto-selects bfloat16 for efficient inference
loader = ModelLoader(config, device="cuda")

# MPS (Apple Silicon): auto-selects float16 for Metal Performance Shaders
loader = ModelLoader(config, device="mps")

# CPU: uses float32 for compatibility
loader = ModelLoader(config, device="cpu")


## Performance Notes

- **Model load time:** ~15–30s for Gemma 4 4B (first call only, then idempotent)
- **Inference time:** ~3–8s per query (varies by max_new_tokens, hardware)
- **Memory:** ~8–10 GB VRAM / unified memory (bfloat16 on CUDA / float16 on MPS)
- **Processor chat template:** ~1–2ms per prompt (negligible)
- **Processor decode:** ~1–2ms

## Related Areas

- **Retrieval:** `src/retrieval/` — produces `RetrievedChunk` tuples consumed by `build_prompt()`
- **Pipeline:** `src/pipeline/` — orchestrates retrieval + generation, handles error flow
- **API:** `src/api/` — HTTP endpoint wrapper around `Generator.generate()`
- **Types:** `src/types.py` — shared dataclasses (`GenerationResult`, `RetrievedChunk`, etc.)

## Exports from `src/generation/__init__.py`

```python
__all__ = [
    "Generator",           # Main orchestrator
    "GeneratorProtocol",   # Protocol for duck typing
    "GenerationConfig",    # Frozen dataclass for LLM config
    "build_prompt",        # Function to format prompt
]
````

Public API only. Internal classes (`ModelLoader`, `Inferencer`) are imported directly from submodules in tests and pipeline code.
