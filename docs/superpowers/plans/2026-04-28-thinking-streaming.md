# Thinking Mode + Stream Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Gemma 4 thinking mode behind `THINKING_ENABLED`, parse thoughts out of non-streaming responses via `processor.parse_response`, and filter the thought block from the streaming endpoint via a state machine.

**Architecture:** `config.py` grows one flag; `prompt_builder.py` exports `SYSTEM_PROMPT` as a public constant and removes it from the returned string; `Inferencer` receives a proper system+user messages list, uses `enable_thinking` dynamically, and gains a `thinking` override on both `infer` and `stream_infer`; a private `_ThinkingStreamFilter` class handles streaming thought suppression; both HyDE transformers hardcode `thinking=False`.

**Tech Stack:** Python 3.13, pytest, torch, transformers `AutoProcessor.parse_response`, `TextIteratorStreamer`.

---

## File Map

| File | Change |
|---|---|
| `src/config.py` | Add `thinking_enabled: bool = False`, parse `THINKING_ENABLED` env var |
| `src/generation/prompt_builder.py` | Rename `_SYSTEM_PROMPT` → `SYSTEM_PROMPT` (public export); remove from `build_prompt` return |
| `src/generation/inference.py` | Add `_ThinkingStreamFilter`; add `thinking_enabled` to `__init__`; restructure `_prepare_inputs` (system+user messages, dynamic `enable_thinking`); conditional decode+`parse_response` in `infer`; `thinking` override on `infer` and `stream_infer` |
| `src/retrieval/query_transformer.py` | Both HyDE transformers pass `thinking=False` to `inferencer.infer()` |
| `src/api/dependencies.py` | Pass `thinking_enabled=settings.thinking_enabled` to `Inferencer(...)` |
| `.env.example` | Document `THINKING_ENABLED=false` |
| `tests/unit/test_config.py` | New tests for `THINKING_ENABLED` |
| `tests/unit/test_prompt_builder.py` | Assert `SYSTEM_PROMPT` importable; assert `build_prompt` no longer starts with it |
| `tests/unit/test_inference.py` | Update existing messages assertion; new `TestInferencerThinking` + `TestThinkingStreamFilter` |
| `tests/unit/test_query_transformer.py` | Assert both HyDE transformers pass `thinking=False` |
| `tests/unit/test_dependencies.py` | Assert `Inferencer` receives `thinking_enabled` from settings |

---

### Task 1: Config — `THINKING_ENABLED` setting

**Files:**
- Modify: `src/config.py`
- Modify: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_config.py` inside the existing test class (follow the pattern of other `_parse_bool` tests already in the file):

```python
def test_thinking_enabled_defaults_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THINKING_ENABLED", raising=False)
    settings = Settings.from_env()
    assert settings.thinking_enabled is False


def test_thinking_enabled_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THINKING_ENABLED", "true")
    settings = Settings.from_env()
    assert settings.thinking_enabled is True


def test_thinking_enabled_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THINKING_ENABLED", "maybe")
    with pytest.raises(ValueError, match="THINKING_ENABLED"):
        Settings.from_env()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/unit/test_config.py -k "thinking" -v
```

Expected: `AttributeError: 'Settings' object has no attribute 'thinking_enabled'`

- [ ] **Step 3: Add `thinking_enabled` to `Settings` dataclass**

In `src/config.py`, add after `retrieval_hard_floor`:

```python
retrieval_hard_floor: float = -2.0
thinking_enabled: bool = False
```

- [ ] **Step 4: Add parsing to `from_env`**

In `Settings.from_env()`, add to the `return cls(...)` block (after `retrieval_hard_floor=retrieval_hard_floor`):

```python
thinking_enabled=_parse_bool(os.getenv("THINKING_ENABLED"), "THINKING_ENABLED", False),
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
uv run pytest tests/unit/test_config.py -k "thinking" -v
```

Expected: 3 passed.

- [ ] **Step 6: Run full config test suite to check for regressions**

```bash
uv run pytest tests/unit/test_config.py -v
```

Expected: all pass.

- [ ] **Step 7: Lint**

```bash
uv run ruff check src/config.py && uv run ruff format --check src/config.py
```

- [ ] **Step 8: Commit**

```bash
git add src/config.py tests/unit/test_config.py
git commit -m "feat: add THINKING_ENABLED config flag"
```

---

### Task 2: `prompt_builder.py` — export `SYSTEM_PROMPT`, strip from return value

**Files:**
- Modify: `src/generation/prompt_builder.py`
- Modify: `tests/unit/test_prompt_builder.py`

- [ ] **Step 1: Write failing tests**

Add at the top of `tests/unit/test_prompt_builder.py` alongside the existing import:

```python
from src.generation.prompt_builder import SYSTEM_PROMPT, build_prompt
```

Then add a new test class:

```python
@pytest.mark.unit
class TestSystemPromptExport:
    def test_system_prompt_is_public(self) -> None:
        assert isinstance(SYSTEM_PROMPT, str)
        assert "PokéSage" in SYSTEM_PROMPT

    def test_build_prompt_returns_user_message_only(self) -> None:
        chunk = _chunk("Some text.", score=0.9)
        prompt = build_prompt("What type?", (chunk,))
        assert prompt.startswith("Context:")
        assert not prompt.startswith(SYSTEM_PROMPT[:30])
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/unit/test_prompt_builder.py::TestSystemPromptExport -v
```

Expected: `ImportError: cannot import name 'SYSTEM_PROMPT'` (or `AssertionError` if import succeeds but content is wrong).

- [ ] **Step 3: Rename `_SYSTEM_PROMPT` and update `build_prompt`**

In `src/generation/prompt_builder.py`:

1. Change line 9 from `_SYSTEM_PROMPT = (` to `SYSTEM_PROMPT = (`
2. Update `build_prompt` — change the return statement from:

```python
    return (
        f"{_SYSTEM_PROMPT}\n\nContext:\n{context_block}\n\nQuestion: {sanitized_query}\n\nAnswer:"
    )
```

to:

```python
    return (
        f"Context:\n{context_block}\n\nQuestion: {sanitized_query}\n\nAnswer:"
    )
```

- [ ] **Step 4: Run new tests**

```bash
uv run pytest tests/unit/test_prompt_builder.py::TestSystemPromptExport -v
```

Expected: 2 passed.

- [ ] **Step 5: Run full prompt builder test suite**

```bash
uv run pytest tests/unit/test_prompt_builder.py -v
```

Expected: all pass (existing tests check context blocks and source labels, not `SYSTEM_PROMPT` content).

- [ ] **Step 6: Lint**

```bash
uv run ruff check src/generation/prompt_builder.py && uv run ruff format --check src/generation/prompt_builder.py
```

- [ ] **Step 7: Commit**

```bash
git add src/generation/prompt_builder.py tests/unit/test_prompt_builder.py
git commit -m "feat: export SYSTEM_PROMPT from prompt_builder; remove from build_prompt return"
```

---

### Task 3: `_ThinkingStreamFilter` state machine

**Files:**
- Modify: `src/generation/inference.py`
- Modify: `tests/unit/test_inference.py`

- [ ] **Step 1: Write failing tests**

Add a new class at the bottom of `tests/unit/test_inference.py`:

```python
@pytest.mark.unit
class TestThinkingStreamFilter:
    def _make_filter(self) -> Any:
        from src.generation.inference import _ThinkingStreamFilter

        return _ThinkingStreamFilter()

    def test_buffers_all_tokens_before_close_tag(self) -> None:
        f = self._make_filter()
        assert f.feed("<|channel>thought\n") is None
        assert f.feed("some internal reasoning") is None

    def test_emits_suffix_when_close_tag_received(self) -> None:
        f = self._make_filter()
        f.feed("<|channel>thought\n")
        result = f.feed("<channel|>The actual answer")
        assert result == "The actual answer"

    def test_returns_none_when_close_tag_has_no_suffix(self) -> None:
        f = self._make_filter()
        f.feed("<|channel>thought\n")
        result = f.feed("<channel|>")
        assert result is None

    def test_passthrough_after_close_tag_seen(self) -> None:
        f = self._make_filter()
        f.feed("<|channel>thought\n<channel|>")
        assert f.feed("more answer tokens") == "more answer tokens"
        assert f.feed("and more") == "and more"

    def test_handles_straddled_close_tag(self) -> None:
        f = self._make_filter()
        f.feed("<|channel>thought\n")
        f.feed("reasoning ")
        f.feed("<channel")      # first half of tag
        result = f.feed("|>answer here")  # second half
        assert result == "answer here"

    def test_close_tag_inline_with_thought(self) -> None:
        f = self._make_filter()
        result = f.feed("<|channel>thought\nreasoning<channel|>answer")
        assert result == "answer"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/unit/test_inference.py::TestThinkingStreamFilter -v
```

Expected: `ImportError: cannot import name '_ThinkingStreamFilter'`

- [ ] **Step 3: Add `_ThinkingStreamFilter` to `inference.py`**

Insert before the `Inferencer` class in `src/generation/inference.py`:

```python
class _ThinkingStreamFilter:
    """Two-state machine that buffers thought content and emits only post-thought tokens.

    BUFFERING: accumulates tokens until '<channel|>' is detected.
    EMITTING: yields tokens directly to caller.
    A 64-char rolling buffer handles '<channel|>' arriving across two streamer chunks.
    """

    _CLOSE = "<channel|>"
    _LOOKBACK = 64

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

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/unit/test_inference.py::TestThinkingStreamFilter -v
```

Expected: 6 passed.

- [ ] **Step 5: Lint**

```bash
uv run ruff check src/generation/inference.py && uv run ruff format --check src/generation/inference.py
```

- [ ] **Step 6: Commit**

```bash
git add src/generation/inference.py tests/unit/test_inference.py
git commit -m "feat: add _ThinkingStreamFilter state machine for thought suppression"
```

---

### Task 4: `Inferencer.__init__` + `_prepare_inputs` restructure

**Files:**
- Modify: `src/generation/inference.py`
- Modify: `tests/unit/test_inference.py`

This task updates `_prepare_inputs` to use a proper system + user messages list and to pass `enable_thinking` dynamically. One existing test (`test_apply_chat_template_called_with_prompt`) asserts the old single-message format — it must be updated as part of this task.

- [ ] **Step 1: Write failing tests for the new `_prepare_inputs` behaviour**

Add inside `TestInferencerInfer` in `tests/unit/test_inference.py`:

```python
def test_prepare_inputs_sends_system_and_user_messages(self) -> None:
    from src.generation.prompt_builder import SYSTEM_PROMPT

    inferencer, _, fake_processor, _ = _make_inferencer()
    inferencer.infer("What is Pikachu?")

    args, _ = fake_processor.apply_chat_template.call_args
    messages = args[0]
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "What is Pikachu?"}


def test_prepare_inputs_enable_thinking_false_by_default(self) -> None:
    inferencer, _, fake_processor, _ = _make_inferencer()
    inferencer.infer("prompt")

    _, kwargs = fake_processor.apply_chat_template.call_args
    assert kwargs["enable_thinking"] is False


def test_prepare_inputs_enable_thinking_true_when_thinking_on(self) -> None:
    from src.generation.inference import Inferencer
    from src.generation.models import GenerationConfig

    fake_model = MagicMock()
    fake_model.device = "cpu"
    fake_processor = MagicMock()
    fake_inputs = _make_fake_inputs(3)
    fake_processor.apply_chat_template.return_value = "formatted"
    fake_processor.return_value = fake_inputs
    fake_model.generate.return_value = torch.arange(5).unsqueeze(0)
    fake_processor.decode.return_value = "answer"
    fake_processor.parse_response.return_value = "answer"

    inferencer = Inferencer(
        fake_model, fake_processor, GenerationConfig(model_id="test/model"),
        thinking_enabled=True,
    )
    inferencer.infer("prompt")

    _, kwargs = fake_processor.apply_chat_template.call_args
    assert kwargs["enable_thinking"] is True


def test_thinking_enabled_defaults_to_false(self) -> None:
    inferencer, _, _, _ = _make_inferencer()
    assert inferencer._thinking_enabled is False


def test_thinking_enabled_true_stored(self) -> None:
    from src.generation.inference import Inferencer
    from src.generation.models import GenerationConfig

    inferencer = Inferencer(
        MagicMock(), MagicMock(), GenerationConfig(model_id="test/model"),
        thinking_enabled=True,
    )
    assert inferencer._thinking_enabled is True
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
uv run pytest tests/unit/test_inference.py -k "prepare_inputs or thinking_enabled" -v
```

Expected: failures due to messages having only one entry and `_thinking_enabled` not existing.

- [ ] **Step 3: Update the stale existing test**

In `tests/unit/test_inference.py`, replace `test_apply_chat_template_called_with_prompt`:

```python
def test_apply_chat_template_called_with_prompt(self) -> None:
    from src.generation.prompt_builder import SYSTEM_PROMPT

    inferencer, _, fake_processor, _ = _make_inferencer()
    inferencer.infer("What is Pikachu?")

    fake_processor.apply_chat_template.assert_called_once()
    args, _ = fake_processor.apply_chat_template.call_args
    messages = args[0]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "What is Pikachu?"}
```

- [ ] **Step 4: Update `inference.py` — `__init__` and `_prepare_inputs`**

In `src/generation/inference.py`, add the import at the top of the file:

```python
from src.generation.prompt_builder import SYSTEM_PROMPT
```

Update `Inferencer.__init__`:

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

Replace `_prepare_inputs` entirely:

```python
def _prepare_inputs(self, user_message: str, *, thinking: bool = False) -> tuple[Any, int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    text: str = self._processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)
    input_len: int = inputs["input_ids"].shape[-1]
    return inputs, input_len
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
uv run pytest tests/unit/test_inference.py -v
```

Expected: all pass (the stale test is now updated, new tests pass).

- [ ] **Step 6: Lint**

```bash
uv run ruff check src/generation/inference.py tests/unit/test_inference.py
uv run ruff format --check src/generation/inference.py tests/unit/test_inference.py
```

- [ ] **Step 7: Commit**

```bash
git add src/generation/inference.py tests/unit/test_inference.py
git commit -m "feat: add thinking_enabled to Inferencer; restructure _prepare_inputs with system message"
```

---

### Task 5: `Inferencer.infer` — thinking decode path

**Files:**
- Modify: `src/generation/inference.py`
- Modify: `tests/unit/test_inference.py`

- [ ] **Step 1: Write failing tests**

Add a new class to `tests/unit/test_inference.py`:

```python
@pytest.mark.unit
class TestInferencerThinking:
    def _make_thinking_inferencer(
        self,
        *,
        raw_decoded: str = "<|channel>thought\nsome reasoning<channel|>The answer",
        parse_response_return: str = "The answer",
    ) -> tuple[Any, Any, Any]:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "formatted"
        fake_processor.return_value = fake_inputs
        fake_model.generate.return_value = torch.arange(5).unsqueeze(0)
        fake_processor.decode.return_value = raw_decoded
        fake_processor.parse_response.return_value = parse_response_return

        inferencer = Inferencer(
            fake_model, fake_processor, GenerationConfig(model_id="test/model"),
            thinking_enabled=True,
        )
        return inferencer, fake_model, fake_processor

    def test_thinking_path_uses_skip_special_tokens_false(self) -> None:
        inferencer, _, fake_processor = self._make_thinking_inferencer()
        inferencer.infer("question")

        _, kwargs = fake_processor.decode.call_args
        assert kwargs.get("skip_special_tokens") is False

    def test_thinking_path_calls_parse_response(self) -> None:
        inferencer, _, fake_processor = self._make_thinking_inferencer()
        inferencer.infer("question")

        fake_processor.parse_response.assert_called_once()
        raw = fake_processor.decode.return_value
        fake_processor.parse_response.assert_called_with(raw)

    def test_thinking_path_returns_parse_response_output(self) -> None:
        inferencer, _, _ = self._make_thinking_inferencer(parse_response_return="Charizard is Fire.")
        result = inferencer.infer("question")
        assert result == "Charizard is Fire."

    def test_thinking_override_false_skips_parse_response(self) -> None:
        """Inferencer with thinking_enabled=True but infer(thinking=False) uses normal path."""
        inferencer, _, fake_processor = self._make_thinking_inferencer()
        fake_processor.decode.return_value = "normal answer"
        result = inferencer.infer("question", thinking=False)

        _, kwargs = fake_processor.decode.call_args
        assert kwargs.get("skip_special_tokens") is True
        fake_processor.parse_response.assert_not_called()
        assert result == "normal answer"

    def test_thinking_override_true_on_non_thinking_instance(self) -> None:
        """infer(thinking=True) overrides instance default of False."""
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "formatted"
        fake_processor.return_value = fake_inputs
        fake_model.generate.return_value = torch.arange(5).unsqueeze(0)
        fake_processor.decode.return_value = "<|channel>thought\n<channel|>answer"
        fake_processor.parse_response.return_value = "answer"

        inferencer = Inferencer(
            fake_model, fake_processor, GenerationConfig(model_id="test/model"),
            thinking_enabled=False,
        )
        result = inferencer.infer("question", thinking=True)

        _, kwargs = fake_processor.decode.call_args
        assert kwargs.get("skip_special_tokens") is False
        fake_processor.parse_response.assert_called_once()
        assert result == "answer"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/unit/test_inference.py::TestInferencerThinking -v
```

Expected: failures — `infer` doesn't yet have the thinking branch.

- [ ] **Step 3: Update `infer` in `inference.py`**

Replace the `infer` method:

```python
def infer(
    self,
    prompt: str,
    *,
    max_new_tokens: int | None = None,
    thinking: bool | None = None,
) -> str:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    resolved_thinking = thinking if thinking is not None else self._thinking_enabled
    resolved_max_new_tokens = (
        max_new_tokens if max_new_tokens is not None else self._config.max_new_tokens
    )

    inputs, input_len = self._prepare_inputs(prompt, thinking=resolved_thinking)
    _LOG.debug(
        "Inferring: prompt_len=%d tokens, max_new=%d",
        input_len,
        resolved_max_new_tokens,
    )

    output_ids = self._model.generate(  # type: ignore[operator]
        **inputs,
        max_new_tokens=resolved_max_new_tokens,
        temperature=self._config.temperature,
        top_p=self._config.top_p,
        do_sample=self._config.do_sample,
    )

    if output_ids.shape[0] == 0:
        raise RuntimeError(
            f"Model generate() returned no sequences (shape={output_ids.shape!r})"
        )

    response_ids = output_ids[0][input_len:]
    if response_ids.shape[-1] == 0:
        raise RuntimeError(
            f"Model generate() returned no new tokens (input_len={input_len}, "
            f"output_len={output_ids.shape[-1]})"
        )

    if resolved_thinking:
        raw: str = self._processor.decode(response_ids, skip_special_tokens=False)
        if not isinstance(raw, str):
            raise TypeError(f"Processor returned {type(raw).__name__}, expected str")
        if "<channel|>" in raw:
            thought = raw.split("<channel|>", 1)[0]
            _LOG.debug("thinking_block: %s", thought[:200])
        response: str = self._processor.parse_response(raw)
    else:
        response = self._processor.decode(response_ids, skip_special_tokens=True)
        if not isinstance(response, str):
            raise TypeError(f"Processor returned {type(response).__name__}, expected str")

    stripped_response = response.strip() if isinstance(response, str) else str(response).strip()
    if not stripped_response:
        raise RuntimeError(
            f"Model generated only whitespace/empty output (input_len={input_len}, "
            f"output_tokens={response_ids.shape[-1]}, decoded_len={len(response)})"
        )

    _LOG.debug("Generated %d chars", len(stripped_response))
    return stripped_response
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/unit/test_inference.py::TestInferencerThinking -v
```

Expected: 5 passed.

- [ ] **Step 5: Run full inference test suite**

```bash
uv run pytest tests/unit/test_inference.py -v
```

Expected: all pass.

- [ ] **Step 6: Lint**

```bash
uv run ruff check src/generation/inference.py && uv run ruff format --check src/generation/inference.py
```

- [ ] **Step 7: Commit**

```bash
git add src/generation/inference.py tests/unit/test_inference.py
git commit -m "feat: add thinking decode path to Inferencer.infer with parse_response"
```

---

### Task 6: `stream_infer` — thinking filter integration

**Files:**
- Modify: `src/generation/inference.py`
- Modify: `tests/unit/test_inference.py`

- [ ] **Step 1: Write failing tests**

Add a new class to `tests/unit/test_inference.py`:

```python
@pytest.mark.unit
class TestInferencerStreamInferThinking:
    def _make_streaming_inferencer(
        self, *, thinking_enabled: bool = False, tokens: list[str] | None = None
    ) -> tuple[Any, Any]:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        if tokens is None:
            tokens = ["hello", " world"]

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "formatted"
        fake_processor.return_value = fake_inputs

        # TextIteratorStreamer mock: iterate over tokens then stop
        fake_streamer = iter(tokens)
        fake_processor.side_effect = None

        with patch("src.generation.inference.TextIteratorStreamer") as mock_streamer_cls:
            mock_streamer_cls.return_value = fake_streamer
            inferencer = Inferencer(
                fake_model, fake_processor, GenerationConfig(model_id="test/model"),
                thinking_enabled=thinking_enabled,
            )
            # Return both so test can patch streamer tokens
            return inferencer, mock_streamer_cls

    def test_stream_infer_thinking_false_uses_skip_special_tokens_true(self) -> None:
        from unittest.mock import patch

        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "formatted"
        fake_processor.return_value = fake_inputs

        with patch("src.generation.inference.TextIteratorStreamer") as mock_cls:
            mock_cls.return_value = iter(["hello"])
            inferencer = Inferencer(
                fake_model, fake_processor, GenerationConfig(model_id="test/model"),
                thinking_enabled=False,
            )
            list(inferencer.stream_infer("prompt"))
            _, kwargs = mock_cls.call_args
            assert kwargs.get("skip_special_tokens") is True

    def test_stream_infer_thinking_true_uses_skip_special_tokens_false(self) -> None:
        from unittest.mock import patch

        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "formatted"
        fake_processor.return_value = fake_inputs

        with patch("src.generation.inference.TextIteratorStreamer") as mock_cls:
            mock_cls.return_value = iter(["<|channel>thought\n<channel|>answer"])
            inferencer = Inferencer(
                fake_model, fake_processor, GenerationConfig(model_id="test/model"),
                thinking_enabled=True,
            )
            list(inferencer.stream_infer("prompt"))
            _, kwargs = mock_cls.call_args
            assert kwargs.get("skip_special_tokens") is False

    def test_stream_infer_thinking_filters_thought_block(self) -> None:
        from unittest.mock import patch

        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "formatted"
        fake_processor.return_value = fake_inputs

        tokens = ["<|channel>thought\n", "internal reasoning", "<channel|>", "The answer"]

        with patch("src.generation.inference.TextIteratorStreamer") as mock_cls:
            mock_cls.return_value = iter(tokens)
            inferencer = Inferencer(
                fake_model, fake_processor, GenerationConfig(model_id="test/model"),
                thinking_enabled=True,
            )
            result = list(inferencer.stream_infer("prompt"))

        assert result == ["The answer"]

    def test_stream_infer_thinking_false_no_filter_applied(self) -> None:
        from unittest.mock import patch

        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "formatted"
        fake_processor.return_value = fake_inputs

        tokens = ["hello", " world"]

        with patch("src.generation.inference.TextIteratorStreamer") as mock_cls:
            mock_cls.return_value = iter(tokens)
            inferencer = Inferencer(
                fake_model, fake_processor, GenerationConfig(model_id="test/model"),
                thinking_enabled=False,
            )
            result = list(inferencer.stream_infer("prompt"))

        assert result == ["hello", " world"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/unit/test_inference.py::TestInferencerStreamInferThinking -v
```

Expected: failures — `stream_infer` has no `thinking` logic yet.

- [ ] **Step 3: Update `stream_infer` in `inference.py`**

Replace the `stream_infer` method:

```python
def stream_infer(
    self,
    prompt: str,
    *,
    max_new_tokens: int | None = None,
    thinking: bool | None = None,
) -> Iterator[str]:
    """Yield tokens one-at-a-time as the model produces them via TextIteratorStreamer.

    When thinking is enabled, thought content is suppressed via _ThinkingStreamFilter.

    Raises:
        ValueError: If prompt is empty or whitespace-only.
        RuntimeError: If model.generate() raises during streaming.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    resolved_thinking = thinking if thinking is not None else self._thinking_enabled
    resolved_max_new_tokens = (
        max_new_tokens if max_new_tokens is not None else self._config.max_new_tokens
    )

    inputs, _ = self._prepare_inputs(prompt, thinking=resolved_thinking)

    streamer = TextIteratorStreamer(
        self._processor,
        skip_prompt=True,
        skip_special_tokens=not resolved_thinking,
    )

    exc_holder: list[BaseException] = []

    def _generate() -> None:
        try:
            self._model.generate(  # type: ignore[operator]
                **inputs,
                max_new_tokens=resolved_max_new_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                do_sample=self._config.do_sample,
                streamer=streamer,
            )
        except Exception as exc:
            exc_holder.append(exc)
            streamer.end()  # type: ignore[no-untyped-call]

    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()

    filter_ = _ThinkingStreamFilter() if resolved_thinking else None
    try:
        for text_piece in streamer:
            if not text_piece:
                continue
            if filter_ is not None:
                text_piece = filter_.feed(text_piece)
                if text_piece is None:
                    continue
            yield text_piece
    finally:
        thread.join()

    if exc_holder:
        raise RuntimeError(f"Model generate() raised: {exc_holder[0]}") from exc_holder[0]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/unit/test_inference.py::TestInferencerStreamInferThinking -v
```

Expected: 4 passed.

- [ ] **Step 5: Run full inference test suite**

```bash
uv run pytest tests/unit/test_inference.py -v
```

Expected: all pass.

- [ ] **Step 6: Lint**

```bash
uv run ruff check src/generation/inference.py && uv run ruff format --check src/generation/inference.py
```

- [ ] **Step 7: Commit**

```bash
git add src/generation/inference.py tests/unit/test_inference.py
git commit -m "feat: add thinking filter to stream_infer; skip_special_tokens conditional on thinking"
```

---

### Task 7: HyDE transformers — hardcode `thinking=False`

**Files:**
- Modify: `src/retrieval/query_transformer.py`
- Modify: `tests/unit/test_query_transformer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_query_transformer.py` inside the existing `TestHyDETransformer` class:

```python
def test_infer_called_with_thinking_false(self) -> None:
    mock_inf = self._make_inferencer()
    t = HyDETransformer(mock_inf, max_new_tokens=100)
    t.transform("What is Pikachu?")

    _, kwargs = mock_inf.infer.call_args
    assert kwargs.get("thinking") is False
```

And inside the existing `TestMultiDraftHyDETransformer` class (find it by checking the test file for `MultiDraftHyDE`):

```python
def test_transform_called_with_thinking_false(self) -> None:
    mock_inf = MagicMock()
    mock_inf.infer.return_value = "a hypothesis"
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = MagicMock(dense=[[0.1]], sparse=[{1: 0.5}])

    t = MultiDraftHyDETransformer(mock_inf, mock_embedder, num_drafts=1)
    t.transform("What is Pikachu?")

    _, kwargs = mock_inf.infer.call_args
    assert kwargs.get("thinking") is False

def test_transform_to_embedding_called_with_thinking_false(self) -> None:
    mock_inf = MagicMock()
    mock_inf.infer.return_value = "a hypothesis"
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = MagicMock(dense=[[0.1]], sparse=[{1: 0.5}])

    t = MultiDraftHyDETransformer(mock_inf, mock_embedder, num_drafts=1)
    t.transform_to_embedding("What is Pikachu?")

    _, kwargs = mock_inf.infer.call_args
    assert kwargs.get("thinking") is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/unit/test_query_transformer.py -k "thinking_false" -v
```

Expected: 3 failures — `thinking` kwarg is not yet passed.

- [ ] **Step 3: Update both HyDE transformers in `query_transformer.py`**

In `HyDETransformer.transform`, change the `infer` call from:

```python
hypothesis: str = self._inferencer.infer(prompt, max_new_tokens=self._max_new_tokens)
```

to:

```python
hypothesis: str = self._inferencer.infer(
    prompt, max_new_tokens=self._max_new_tokens, thinking=False
)
```

In `MultiDraftHyDETransformer.transform`, change:

```python
result: str = self._inferencer.infer(prompt, max_new_tokens=self._max_new_tokens)
```

to:

```python
result: str = self._inferencer.infer(
    prompt, max_new_tokens=self._max_new_tokens, thinking=False
)
```

In `MultiDraftHyDETransformer.transform_to_embedding`, change:

```python
result: str = self._inferencer.infer(prompt, max_new_tokens=self._max_new_tokens)
```

to:

```python
result: str = self._inferencer.infer(
    prompt, max_new_tokens=self._max_new_tokens, thinking=False
)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/unit/test_query_transformer.py -k "thinking_false" -v
```

Expected: 3 passed.

- [ ] **Step 5: Run full query transformer test suite**

```bash
uv run pytest tests/unit/test_query_transformer.py -v
```

Expected: all pass.

- [ ] **Step 6: Lint**

```bash
uv run ruff check src/retrieval/query_transformer.py && uv run ruff format --check src/retrieval/query_transformer.py
```

- [ ] **Step 7: Commit**

```bash
git add src/retrieval/query_transformer.py tests/unit/test_query_transformer.py
git commit -m "feat: HyDE transformers pass thinking=False to prevent reasoning in hypothetical docs"
```

---

### Task 8: Wire `thinking_enabled` in `dependencies.py` + document in `.env.example`

**Files:**
- Modify: `src/api/dependencies.py`
- Modify: `tests/unit/test_dependencies.py`
- Modify: `.env.example`

- [ ] **Step 1: Write failing test**

In `tests/unit/test_dependencies.py`, add a new test class:

```python
@pytest.mark.unit
class TestBuildPipelineThinking:
    def test_inferencer_receives_thinking_enabled_from_settings(self) -> None:
        from unittest.mock import MagicMock, patch

        from src.api.dependencies import build_pipeline

        with (
            patch("src.api.dependencies.ModelLoader") as mock_loader_cls,
            patch("src.api.dependencies.BGEEmbedder"),
            patch("src.api.dependencies.BGEReranker"),
            patch("src.api.dependencies.QdrantVectorStore"),
            patch("src.api.dependencies.QdrantClient"),
            patch("src.api.dependencies.Inferencer") as mock_inferencer_cls,
            patch("src.api.dependencies.Settings") as mock_settings_cls,
        ):
            mock_settings = MagicMock()
            mock_settings.thinking_enabled = True
            mock_settings.hyde_enabled = False
            mock_settings.routing_enabled = False
            mock_settings.refiner_enabled = False
            mock_settings.async_pipeline_enabled = False
            mock_settings.cache_enabled = False
            mock_settings_cls.from_env.return_value = mock_settings
            mock_loader_cls.return_value.get_model.return_value = MagicMock()
            mock_loader_cls.return_value.get_tokenizer.return_value = MagicMock()

            build_pipeline()

            _, kwargs = mock_inferencer_cls.call_args
            assert kwargs.get("thinking_enabled") is True
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
uv run pytest tests/unit/test_dependencies.py::TestBuildPipelineThinking -v
```

Expected: failure — `thinking_enabled` not yet passed to `Inferencer`.

- [ ] **Step 3: Update `dependencies.py`**

Find the `Inferencer(...)` construction in `build_pipeline` (around line 97) and add `thinking_enabled`:

```python
inferencer = Inferencer(
    model=loader.get_model(),
    processor=loader.get_tokenizer(),
    config=gen_config,
    thinking_enabled=settings.thinking_enabled,
)
```

- [ ] **Step 4: Update `.env.example`**

Add the following line after the generation model settings block (near `MAX_NEW_TOKENS` / `TEMPERATURE`):

```
THINKING_ENABLED=false                                      # enable Gemma 4 reasoning mode (adds latency; eval before enabling)
```

- [ ] **Step 5: Run test to confirm it passes**

```bash
uv run pytest tests/unit/test_dependencies.py::TestBuildPipelineThinking -v
```

Expected: 1 passed.

- [ ] **Step 6: Run full dependencies test suite**

```bash
uv run pytest tests/unit/test_dependencies.py -v
```

Expected: all pass.

- [ ] **Step 7: Lint**

```bash
uv run ruff check src/api/dependencies.py && uv run ruff format --check src/api/dependencies.py
```

- [ ] **Step 8: Commit**

```bash
git add src/api/dependencies.py tests/unit/test_dependencies.py .env.example
git commit -m "feat: wire thinking_enabled from Settings through to Inferencer"
```

---

### Task 9: Full suite verification + PR

**Files:** None modified — verification and PR only.

- [ ] **Step 1: Run the complete unit test suite**

```bash
uv run pytest tests/unit/ -v --tb=short -q
```

Expected: all pass, 0 failures.

- [ ] **Step 2: Lint and type-check all modified files**

```bash
uv run ruff check src/config.py src/generation/prompt_builder.py src/generation/inference.py src/retrieval/query_transformer.py src/api/dependencies.py
uv run ruff format --check .
uv run mypy src/generation/inference.py src/generation/prompt_builder.py src/config.py src/retrieval/query_transformer.py
```

Expected: no errors.

- [ ] **Step 3: Create feature branch and open PR**

```bash
git checkout -b feat/task2-3-thinking-mode
git push -u origin feat/task2-3-thinking-mode
gh pr create \
  --title "feat: Gemma 4 thinking mode + stream thought filtering (Tasks 2 & 3)" \
  --body "Adds THINKING_ENABLED flag, wires enable_thinking through apply_chat_template, parses thoughts via processor.parse_response, and suppresses the thought block in streaming via _ThinkingStreamFilter."
```

---

## Empirical Checks (before enabling in production)

Two behaviors must be verified with a live model run before setting `THINKING_ENABLED=true` in production:

1. **`processor.parse_response` return type**: call it with a raw decoded string containing `<|channel>thought\n...<channel|>answer` and confirm it returns a plain `str` (the answer). If it returns a dict, update the `isinstance(response, str)` guard in `infer` to extract the correct key.

2. **`<eos>` in `skip_special_tokens=False` streaming**: verify the streamer does not yield a visible `<eos>` string after the answer ends. If it does, add a filter in `stream_infer`: `if text_piece in ("<eos>", "</s>"): break`.
