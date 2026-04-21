from __future__ import annotations

import json
import time
from typing import Protocol, runtime_checkable

from google import genai

from scripts.training.schemas import JudgeScore

_JUDGE_PROMPT = """\
You are an expert Pokémon knowledge judge. Evaluate the following response to a Pokémon question.

Question: {question}

Response to evaluate: {response}

Retrieved context (what the model saw): {retrieved_chunks}

Reference material (authoritative ground truth): {reference_chunks}

Score the response on three criteria, each 0-100:
- accuracy: factual correctness relative to the reference material
- groundedness: response is grounded in the retrieved context, not hallucinated
- domain_correctness: Pokémon domain knowledge applied correctly

Return ONLY valid JSON in this exact format:
{{"accuracy": <int>, "groundedness": <int>, "domain_correctness": <int>}}
"""


@runtime_checkable
class JudgeProtocol(Protocol):
    def score_candidate(
        self,
        *,
        question: str,
        response: str,
        retrieved_chunks: list[str],
        reference_chunks: list[str],
    ) -> JudgeScore | None: ...


class GeminiJudge:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        max_retries: int = 5,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_retries = max_retries

    def score_candidate(
        self,
        *,
        question: str,
        response: str,
        retrieved_chunks: list[str],
        reference_chunks: list[str],
    ) -> JudgeScore | None:
        contents = _JUDGE_PROMPT.format(
            question=question,
            response=response,
            retrieved_chunks="\n".join(retrieved_chunks),
            reference_chunks="\n".join(reference_chunks),
        )
        for attempt in range(self._max_retries):
            try:
                api_response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                )
                return JudgeScore.model_validate(json.loads(api_response.text))
            except Exception as exc:
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    backoff = 2.0**attempt
                    time.sleep(backoff)
                elif attempt < self._max_retries - 1:
                    time.sleep(1.0)
                else:
                    break
        return None
