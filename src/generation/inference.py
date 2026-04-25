from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from typing import Any

from transformers import PreTrainedModel, TextIteratorStreamer

from src.generation.models import GenerationConfig

_LOG = logging.getLogger(__name__)


class Inferencer:
    def __init__(self, model: PreTrainedModel, processor: Any, config: GenerationConfig) -> None:
        self._model = model
        self._processor = processor
        self._config = config

    def infer(self, prompt: str, *, max_new_tokens: int | None = None) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        resolved_max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else self._config.max_new_tokens
        )

        messages = [{"role": "user", "content": prompt}]
        text: str = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)
        input_len: int = inputs["input_ids"].shape[-1]
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

        response: str = self._processor.decode(response_ids, skip_special_tokens=True)
        if not isinstance(response, str):
            raise TypeError(f"Processor returned {type(response).__name__}, expected str")

        stripped_response = response.strip()
        if not stripped_response:
            raise RuntimeError(
                f"Model generated only whitespace/empty output (input_len={input_len}, "
                f"output_tokens={response_ids.shape[-1]}, decoded_len={len(response)})"
            )

        _LOG.debug("Generated %d chars", len(stripped_response))
        return stripped_response

    def stream_infer(self, prompt: str, *, max_new_tokens: int | None = None) -> Iterator[str]:
        """Yield tokens one-at-a-time as the model produces them via TextIteratorStreamer.

        Raises:
            ValueError: If prompt is empty or whitespace-only.
            RuntimeError: If model.generate() raises during streaming.
        """
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        resolved_max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else self._config.max_new_tokens
        )

        messages = [{"role": "user", "content": prompt}]
        text: str = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._processor,
            skip_prompt=True,
            skip_special_tokens=True,
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
                streamer.end()

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        try:
            for text_piece in streamer:
                if text_piece:
                    yield text_piece
        finally:
            thread.join()

        if exc_holder:
            raise RuntimeError(f"Model generate() raised: {exc_holder[0]}") from exc_holder[0]
