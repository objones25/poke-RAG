from __future__ import annotations

import os
from dataclasses import dataclass, field

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

_MODEL = "gemini-3.1-flash-lite-preview"
_TIMEOUT = 60


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def check_claim(answer: str, claim: str) -> bool:
    client = _client()
    prompt = (
        f"Does the following answer contain the claim?\n\n"
        f"Answer: {answer}\n\nClaim: {claim}\n\n"
        f"Reply with exactly one word: yes or no."
    )
    response = client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=GenerateContentConfig(http_options=HttpOptions(timeout=_TIMEOUT)),
    )
    return (response.text or "").strip().lower().startswith("yes")


def check_hallucination(answer: str, context: str) -> list[str]:
    client = _client()
    prompt = (
        f"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
        f"List factual claims in the answer that are NOT supported by the context, "
        f"one per line. If all claims are supported by the context, output exactly: SUPPORTED"
    )
    response = client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=GenerateContentConfig(http_options=HttpOptions(timeout=_TIMEOUT)),
    )
    text = (response.text or "").strip()
    if text.upper() == "SUPPORTED":
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


@dataclass
class JudgeResult:
    claim_hits: list[bool] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)


def judge_answer(answer: str, claims: list[str], context: str) -> JudgeResult:
    claim_hits = [check_claim(answer, claim) for claim in claims]
    unsupported_claims = check_hallucination(answer, context)
    return JudgeResult(claim_hits=claim_hits, unsupported_claims=unsupported_claims)
