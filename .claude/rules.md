# .claude/rules.md

Applies every session. Short. If a rule conflicts with a user instruction, flag it rather than silently ignoring it.

---

**TDD.** Write a failing test before any `src/` code. Confirm it fails. Then implement. Never skip this.

**Context7 before any library call.** HuggingFace Transformers, `FlagEmbedding`, `qdrant-client`, `trl`, `peft`, `unsloth` — fetch current docs via Context7 before writing any call to these. The `qdrant-client` collection API in particular has changed significantly between versions.

**BGE-M3 embedding class.** Use `FlagEmbedding.BGEM3FlagModel`, not `sentence-transformers`. Call `.encode(..., return_dense=True, return_sparse=True)`. For reranking use `FlagEmbedding.FlagReranker` with `BAAI/bge-reranker-v2-m3` — it is a different class from the embedder.

**Generation model class.** Load via `AutoModelForCausalLM` (Gemma 2 is a causal LM, not a vision-language model).

**No generator without retrieved context.** If retrieval raises or returns nothing, propagate the error. Never call `generator.generate()` without grounded documents.

**`processed/` is read-only.** Never write, modify, or delete any file under `processed/`.

**`scripts/training/` is isolated.** Nothing in `src/` may import from `scripts/training/`. Training code stays in training code.

**Qdrant: three collections, not one.** `bulbapedia`, `pokeapi`, `smogon` are separate collections. Never merge them. Every point must have `source`, `pokemon_name`, `chunk_index`, and `original_doc_id` in its payload to enable filtered retrieval.

**`uv` only.** Never use `pip install`. Use `uv add` or `uv sync`.

**Type everything.** All function signatures need full type annotations. Zero `mypy` errors.

**Specific exceptions.** Raise named exceptions (`RetrievalError`, etc.), not bare `Exception`. Never swallow exceptions silently.

**Immutable data objects.** Domain types (`Document`, `RetrievalResult`, `GenerationOutput`) are frozen dataclasses or Pydantic models. Return new objects, don't mutate.

**Never commit secrets.** No API keys, HuggingFace tokens, or RunPod credentials in any file. Use `.env` (gitignored) or environment variables. See `SECURITY.md`.

**When unsure about architecture, ask.** Wrong structural decisions are expensive to undo.
