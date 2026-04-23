"""Query transformation strategies for retrieval pre-processing."""

from __future__ import annotations

import logging
from typing import Any

_LOG = logging.getLogger(__name__)

_HYDE_PROMPT_TEMPLATE = (
    "Write a short Pokémon knowledge base passage that directly answers this question:\n\n"
    "{query}\n\nPassage:"
)


class PassthroughTransformer:
    """Identity transformer — returns query unchanged."""

    def transform(self, query: str) -> str:
        return query


class HyDETransformer:
    """Hypothetical Document Embedding transformer.

    Generates a pseudo-answer with the LLM and uses that text for embedding
    instead of the raw query, shifting retrieval to answer-to-answer similarity.
    Falls back to the original query on any failure.
    """

    def __init__(self, inferencer: Any, *, max_new_tokens: int = 150) -> None:
        self._inferencer = inferencer
        self._max_new_tokens = max_new_tokens

    def transform(self, query: str) -> str:
        prompt = _HYDE_PROMPT_TEMPLATE.format(query=query)
        try:
            hypothesis: str = self._inferencer.infer(prompt)
        except Exception as exc:
            _LOG.warning("HyDE inference failed, falling back to original query: %s", exc)
            return query

        if not hypothesis or not hypothesis.strip():
            _LOG.warning("HyDE returned empty output, falling back to original query")
            return query

        _LOG.info(
            "HyDE: %d-char query → %d-char hypothesis (preview: %.80r)",
            len(query),
            len(hypothesis),
            hypothesis,
        )
        return hypothesis
