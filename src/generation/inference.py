from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.generation.models import GenerationConfig, TokenizerConfig


class Inferencer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: GenerationConfig,
        tokenizer_config: TokenizerConfig | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._tokenizer_config = tokenizer_config or TokenizerConfig()

    def infer(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        inputs = self._tokenizer(
            prompt,
            return_tensors=self._tokenizer_config.return_tensors,
            max_length=self._tokenizer_config.max_length,
            truncation=self._tokenizer_config.truncation,
        )
        inputs = inputs.to(self._model.device)

        input_ids: torch.Tensor = inputs["input_ids"]
        attention_mask: torch.Tensor = inputs["attention_mask"]
        prompt_len = input_ids.shape[-1]

        output_ids = self._model.generate(  # type: ignore[operator]
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            do_sample=self._config.do_sample,
        )

        generated: torch.Tensor = output_ids[0][prompt_len:]
        decoded = self._tokenizer.decode(generated, skip_special_tokens=True)
        if not isinstance(decoded, str):
            raise TypeError(f"Tokenizer returned {type(decoded).__name__}, expected str")
        return decoded.strip()
