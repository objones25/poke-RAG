"""Query transformation strategies for retrieval pre-processing."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

from src.retrieval.types import EmbeddingOutput

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
            hypothesis: str = self._inferencer.infer(prompt, max_new_tokens=self._max_new_tokens)
        except Exception as exc:
            _LOG.warning(
                "HyDE inference failed, falling back to original query: %s",
                exc,
                exc_info=True,
            )
            return query

        if not hypothesis or not hypothesis.strip():
            _LOG.warning("HyDE returned empty output, falling back to original query")
            return query

        _LOG.info(
            "HyDE: %d-char query → %d-char hypothesis (preview: %r)",
            len(query),
            len(hypothesis),
            hypothesis[:80],
        )
        return hypothesis


class MultiDraftHyDETransformer:
    """Multi-draft HyDE: generate k hypothetical passages, embed all, average the vectors.

    Dense vectors are element-wise mean; sparse weights use per-token maximum across drafts.
    Falls back to embedding the raw query when all drafts fail.
    """

    def __init__(
        self,
        inferencer: Any,
        embedder: Any,
        *,
        num_drafts: int = 3,
        max_new_tokens: int = 150,
    ) -> None:
        self._inferencer = inferencer
        self._embedder = embedder
        self._num_drafts = num_drafts
        self._max_new_tokens = max_new_tokens

    def transform(self, query: str) -> str:
        """Return first successful draft or the original query on failure."""
        prompt = _HYDE_PROMPT_TEMPLATE.format(query=query)
        try:
            hypothesis: str = self._inferencer.infer(prompt, max_new_tokens=self._max_new_tokens)
        except Exception as exc:
            _LOG.warning("MultiDraftHyDE single transform failed, falling back: %s", exc)
            return query
        if not hypothesis or not hypothesis.strip():
            return query
        return hypothesis

    def transform_to_embedding(self, query: str) -> EmbeddingOutput:
        """Generate num_drafts hypotheses, encode as a batch, and fuse the embeddings."""
        drafts: list[str] = []
        prompt = _HYDE_PROMPT_TEMPLATE.format(query=query)
        for i in range(self._num_drafts):
            try:
                result: str = self._inferencer.infer(
                    prompt, max_new_tokens=self._max_new_tokens
                )
                if result and result.strip():
                    drafts.append(result)
            except Exception as exc:
                _LOG.warning("MultiDraftHyDE draft %d failed: %s", i, exc)

        if not drafts:
            _LOG.warning("All MultiDraftHyDE drafts failed; embedding raw query")
            return cast(EmbeddingOutput, self._embedder.encode([query]))

        _LOG.info(
            "MultiDraftHyDE: %d/%d drafts succeeded (preview: %r)",
            len(drafts),
            self._num_drafts,
            drafts[0][:80],
        )
        output: EmbeddingOutput = self._embedder.encode(drafts)

        dense_mean: list[float] = np.mean(np.array(output.dense), axis=0).tolist()

        sparse_merged: dict[int, float] = {}
        for sv in output.sparse:
            for token_id, weight in sv.items():
                if token_id not in sparse_merged or weight > sparse_merged[token_id]:
                    sparse_merged[token_id] = weight

        return EmbeddingOutput(dense=[dense_mean], sparse=[sparse_merged])
