# Task 2 & 3 — Gemma 4 Thinking Mode + Stream Filtering

**Date:** 2026-04-28
**Status:** Approved

---

## Goal

Enable Gemma 4's built-in reasoning mode behind a `THINKING_ENABLED` flag, parse the thought block out of non-streaming responses, and filter it from the streaming response so clients never see internal reasoning tokens.

---

## Background

Gemma 4 supports a configurable thinking mode. Per the model card:

- Thinking is enabled by passing `enable_thinking=True` to `apply_chat_template` (which injects `<|think|>` at the start of the system prompt automatically).
- When thinking is on, the model emits: `<|channel>thought\n[reasoning]<channel|>[answer]`
- `<|channel>` and `<channel|>` are **special tokens**. Decoding must use `skip_special_tokens=False` to see them; `processor.parse_response(raw)` then extracts the answer.
- `enable_thinking=False` is a valid no-op to explicitly disable thinking (it is not a Qwen-ism — Gemma 4 supports the kwarg natively).

The current codebase hardcodes `enable_thinking=False` and merges the system prompt into the user message. This design separates them correctly and wires thinking through the stack.

**Empirical checks required before shipping:**
1. Verify `processor.parse_response(raw)` return type (expected: `str` containing only the answer).
2. Verify `<eos>` token handling in `skip_special_tokens=False` streaming (expected: streamer naturally terminates without yielding a visible `<eos>` string).

---

## Architecture

### Files changed

| File | Change |
|---|---|
| `src/config.py` | Add `thinking_enabled: bool = False`, parse `THINKING_ENABLED` env var |
| `src/generation/prompt_builder.py` | Remove `SYSTEM_PROMPT` from returned string; export as public `SYSTEM_PROMPT` constant |
| `src/generation/inference.py` | Add `thinking_enabled` to `__init__`; restructure `_prepare_inputs`; conditional decode + `parse_response` in `infer`; `thinking` override on `infer` and `stream_infer`; `_ThinkingStreamFilter` state machine |
| `src/retrieval/query_transformer.py` | Both HyDE transformers pass `thinking=False` to `inferencer.infer()` |
| `src/api/dependencies.py` | Pass `thinking_enabled=settings.thinking_enabled` to `Inferencer(...)` |
| `.env.example` | Document `THINKING_ENABLED=false` |

`PromptBuilderProtocol`, `Generator`, and `RAGPipeline` are unchanged.

---

## Component Design

### `config.py`

```python
# Settings dataclass
thinking_enabled: bool = False

# from_env()
thinking_enabled=_parse_bool(os.getenv("THINKING_ENABLED"), "THINKING_ENABLED", False),
```

Default is `False` to avoid latency regressions until eval confirms a win.

---

### `prompt_builder.py`

`_SYSTEM_PROMPT` is renamed to `SYSTEM_PROMPT` (public) and no longer prepended to the returned string. `build_prompt` now returns only the user message (context block + question).

```python
# Before
return f"{_SYSTEM_PROMPT}\n\nContext:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"

# After
return f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
```

`SYSTEM_PROMPT` is imported by `Inferencer`. The few-shot examples and role instructions remain in `SYSTEM_PROMPT` unchanged — they belong in the system role semantically.

---

### `inference.py`

#### `Inferencer.__init__`

```python
def __init__(
    self,
    model: PreTrainedModel,
    processor: Any,
    config: GenerationConfig,
    *,
    thinking_enabled: bool = False,
) -> None:
    self._model = model
    self._processor = processor
    self._config = config
    self._thinking_enabled = thinking_enabled
```

#### `_prepare_inputs`

Restructured to pass a proper system + user messages list. The `enable_thinking` kwarg is now dynamic.

```python
def _prepare_inputs(self, user_message: str, *, thinking: bool = False) -> tuple[Any, int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    text = self._processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)
    return inputs, inputs["input_ids"].shape[-1]
```

#### `infer`

Adds a `thinking` keyword-only override. `None` means use the instance default.

```python
def infer(
    self,
    prompt: str,
    *,
    max_new_tokens: int | None = None,
    thinking: bool | None = None,
) -> str:
    resolved_thinking = thinking if thinking is not None else self._thinking_enabled
    resolved_max = max_new_tokens if max_new_tokens is not None else self._config.max_new_tokens

    inputs, input_len = self._prepare_inputs(prompt, thinking=resolved_thinking)
    output_ids = self._model.generate(**inputs, max_new_tokens=resolved_max, ...)
    response_ids = output_ids[0][input_len:]

    if resolved_thinking:
        raw = self._processor.decode(response_ids, skip_special_tokens=False)
        answer = self._processor.parse_response(raw)  # return type: verify empirically
        if "<channel|>" in raw:
            thought = raw.split("<channel|>", 1)[0]
            _LOG.debug("thinking_block: %s", thought[:200])
    else:
        raw = self._processor.decode(response_ids, skip_special_tokens=True)
        answer = raw

    stripped = answer.strip() if isinstance(answer, str) else str(answer).strip()
    if not stripped:
        raise RuntimeError("Model generated only whitespace/empty output ...")
    return stripped
```

#### `stream_infer`

Adds `thinking` override and conditionally applies `_ThinkingStreamFilter`.

```python
def stream_infer(
    self,
    prompt: str,
    *,
    max_new_tokens: int | None = None,
    thinking: bool | None = None,
) -> Iterator[str]:
    resolved_thinking = thinking if thinking is not None else self._thinking_enabled

    inputs, _ = self._prepare_inputs(prompt, thinking=resolved_thinking)
    streamer = TextIteratorStreamer(
        self._processor,
        skip_prompt=True,
        skip_special_tokens=not resolved_thinking,
    )

    # ... thread + generate unchanged ...

    filter_ = _ThinkingStreamFilter() if resolved_thinking else None
    for text_piece in streamer:
        if not text_piece:
            continue
        if filter_ is not None:
            text_piece = filter_.feed(text_piece)
            if text_piece is None:
                continue
        yield text_piece
```

#### `_ThinkingStreamFilter` (Task 3)

Private class. Two states: `BUFFERING` (discarding thought content) and `EMITTING` (yielding answer tokens). A 64-char rolling buffer handles the case where `<channel|>` straddles two streamer chunks.

```python
class _ThinkingStreamFilter:
    _CLOSE = "<channel|>"
    _LOOKBACK = 64  # wider than len(_CLOSE) to handle straddled delivery

    def __init__(self) -> None:
        self._emitting = False
        self._buf = ""

    def feed(self, token: str) -> str | None:
        if self._emitting:
            return token
        self._buf += token
        if self._CLOSE in self._buf:
            self._emitting = True
            suffix = self._buf.split(self._CLOSE, 1)[1]
            self._buf = ""
            return suffix or None
        if len(self._buf) > self._LOOKBACK:
            self._buf = self._buf[-self._LOOKBACK:]
        return None
```

When `thinking=False`, `filter_` is `None` and the stream is an unmodified passthrough — zero overhead.

---

### `query_transformer.py`

Both `HyDETransformer.transform` and `MultiDraftHyDETransformer.transform` / `transform_to_embedding` explicitly pass `thinking=False`:

```python
hypothesis = self._inferencer.infer(prompt, max_new_tokens=self._max_new_tokens, thinking=False)
```

HyDE wants a passage, not a reasoning chain. The `thinking=False` override is explicit so it stays off regardless of the instance default.

---

### `dependencies.py`

```python
inferencer = Inferencer(
    model=loader.get_model(),
    processor=loader.get_tokenizer(),
    config=gen_config,
    thinking_enabled=settings.thinking_enabled,   # new
)
```

---

## Testing

### `test_config.py`
- `THINKING_ENABLED=true/false/unset/invalid` — four parametrized cases following existing `_parse_bool` pattern.

### `test_prompt_builder.py`
- `SYSTEM_PROMPT` is importable as a public name.
- `build_prompt(...)` return value does NOT begin with `SYSTEM_PROMPT` content.
- All existing context-block, source-label, sanitization, and sort-order tests pass unchanged.

### `test_inference.py` — new class `TestInferencerThinking`
- `test_infer_thinking_calls_parse_response`: mock `parse_response` returns `"answer"`, mock `decode` returns raw with channel tags → assert return is `"answer"`, `parse_response` called once.
- `test_infer_thinking_override_false`: `Inferencer(thinking_enabled=True)` + `infer(prompt, thinking=False)` → `skip_special_tokens=True` path, `parse_response` not called.
- `test_infer_thinking_default_from_init`: `Inferencer(thinking_enabled=True)` + `infer(prompt)` with no override → thinking path taken.
- `test_thinking_stream_filter_buffers_until_close`: feed tokens one by one across a straddled `<channel|>`, assert nothing emitted until close tag, suffix emitted correctly.
- `test_thinking_stream_filter_passthrough_after_close`: all `feed()` calls after close return their input.
- `test_stream_infer_thinking_filters_thought_block`: streamer mock yields `["<|channel>thought\n", "reasoning", "<channel|>", "answer"]` → only `"answer"` reaches caller.

### `test_query_transformer.py`
- `HyDETransformer.transform` calls `inferencer.infer` with `thinking=False`.
- `MultiDraftHyDETransformer.transform` and `transform_to_embedding` both call with `thinking=False`.

### `test_dependencies.py`
- `Inferencer` is constructed with `thinking_enabled=settings.thinking_enabled`.

---

## Acceptance

- `claim_recall` improves on `aggregation` and `comparative` eval buckets with `THINKING_ENABLED=true`.
- Latency increase measured; if median query time increases >10s, default stays `false` and the tradeoff is documented.
- HyDE draft quality unchanged: dump 5 drafts before and after; they should look the same.
- `curl --no-buffer` streaming test with `THINKING_ENABLED=true`: first client token is from the answer, never `<|channel`.
- Total bytes streamed equals decoded answer length (no thought bleed-through).

---

## Not in scope

- Multi-turn conversation history management (model card §3: thoughts must not appear in history — left for a future task when multi-turn is added).
- `top_k=64` sampling parameter from the model card — a separate tuning task.
