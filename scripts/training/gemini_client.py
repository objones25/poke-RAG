from __future__ import annotations

import json
import logging
import random
import re
import time

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from pydantic import ValidationError

from scripts.training.schemas import GeminiQAPair

logger = logging.getLogger(__name__)

_SOURCE_HINTS: dict[str, str] = {
    "pokeapi": "Ask about: base stats, types, abilities, evolution levels, move power/accuracy/PP.",
    "smogon": (
        "Ask about: competitive movesets (include moves/item/ability/EVs/role when"
        " present), tier placement, counters/checks."
    ),
    "bulbapedia": (
        "Ask about: move effects, item effects, ability mechanics, lore, evolution methods."
    ),
}

_PROMPT_TEMPLATE = """\
You are creating training data for a Pokémon Q&A assistant.

Given the following chunk of text from a Pokémon knowledge base (source: {source}), \
generate ONE natural question and the ideal answer. The question must be answerable \
using ONLY the information in the chunk. The answer must be grounded in the chunk's facts.

RULES:
- Do NOT use outside knowledge.
- Do NOT fabricate stats, move names, tier placements, item effects, or evolution data.
- Question: conversational, 20-150 characters.
- Answer: concise and factual, 40-500 characters.
- If the chunk is only a title, a bare name, or lacks enough facts to generate a meaningful \
question and answer, return {{"question": "", "answer": ""}}.
{source_hint}

CHUNK:
{chunk}

Respond with ONLY valid JSON:
{{"question": "...", "answer": "..."}}"""

# Patterns indicating a useless answer from Gemini
_BAD_ANSWER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bprovided (text|context|chunk)\b", re.IGNORECASE),
    re.compile(r"\bdoes not (contain|include|provide|offer)\b", re.IGNORECASE),
    re.compile(r"\bnot enough information\b", re.IGNORECASE),
    re.compile(r"\bno (further |additional )?information\b", re.IGNORECASE),
    re.compile(r"\bonly (the )?(title|name)\b", re.IGNORECASE),
    re.compile(r"\bidentifies .{0,40} as the title\b", re.IGNORECASE),
    re.compile(r"\bno (descriptive|further|additional) content\b", re.IGNORECASE),
    re.compile(r"\bnot (included|provided)\b", re.IGNORECASE),
    re.compile(r"\bno further detail\b", re.IGNORECASE),
    re.compile(r"\bno (content|data|details?) (is |are )?provided\b", re.IGNORECASE),
    re.compile(r"\bcannot (be |)answered\b", re.IGNORECASE),
    re.compile(r"\binsufficient (information|data|context)\b", re.IGNORECASE),
]

# Bare "X is a Pokémon." answer with no real content
_BARE_POKEMON_RE = re.compile(r"^[\w][\w\s\-']+\s+is an? [Pp]ok[eé]mon\.?\s*$")

_MIN_ANSWER_LEN = 40


def _is_quality_pair(pair: GeminiQAPair) -> bool:
    if not pair.question.strip() or not pair.answer.strip():
        return False
    if len(pair.answer.strip()) < _MIN_ANSWER_LEN:
        return False
    if _BARE_POKEMON_RE.match(pair.answer.strip()):
        return False
    return all(not pat.search(pair.answer) for pat in _BAD_ANSWER_PATTERNS)


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite-preview") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate_qa_pair(
        self, chunk: str, source: str, max_retries: int = 3
    ) -> GeminiQAPair | None:
        """Return a validated Q&A pair, or None if the chunk produces no useful content."""
        prompt = _PROMPT_TEMPLATE.format(
            source=source,
            source_hint=_SOURCE_HINTS.get(source, ""),
            chunk=chunk[:2000],
        )
        config = GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=GeminiQAPair.model_json_schema(),
            http_options=HttpOptions(timeout=60),
        )
        response = None
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                if response is None or not response.text:
                    raise RuntimeError("Gemini API returned empty response")
                pair = GeminiQAPair.model_validate_json(response.text)
                return pair if _is_quality_pair(pair) else None
            except ValidationError as exc:
                logger.warning("JSON validation failed (attempt %d): %s", attempt + 1, exc)
                if response is not None and response.text:
                    try:
                        data = json.loads(response.text)
                        pair = GeminiQAPair(**data)
                        return pair if _is_quality_pair(pair) else None
                    except (json.JSONDecodeError, KeyError, ValidationError) as e:
                        logger.debug("Fallback JSON parsing failed: %s", e)
            except RuntimeError:
                # Empty response is fatal, re-raise immediately
                raise
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        "Rate limited. Waiting %.1fs before retry %d.", wait, attempt + 1
                    )
                    time.sleep(wait)
                else:
                    logger.warning("API call failed (attempt %d): %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(1.0)
        raise RuntimeError(f"Failed to generate Q&A pair after {max_retries} attempts")
