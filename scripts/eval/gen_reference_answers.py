"""One-shot helper: draft reference_answer and required_claims for gen_questions.yaml.

Run once, review output, paste back into gen_questions.yaml.

Usage:
    uv run python scripts/eval/gen_reference_answers.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

_MODEL = "gemini-3.1-flash-lite-preview"
_TIMEOUT = 60
_QUESTIONS_PATH = Path(__file__).parent / "gen_questions.yaml"


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def draft_reference(query: str) -> tuple[str, list[str]]:
    client = _client()
    prompt = (
        f"You are an expert on Pokémon. For the following query, provide:\n"
        f"1. A concise reference answer (1-2 sentences).\n"
        f"2. A list of 1-3 required claims — short, exact strings that must appear in a correct"
        f" answer.\n\n"
        f"Query: {query}\n\n"
        f"Respond in this exact YAML format:\n"
        f'reference_answer: "..."\n'
        f"required_claims:\n"
        f'  - "..."\n'
    )
    response = client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=GenerateContentConfig(http_options=HttpOptions(timeout=_TIMEOUT)),
    )
    parsed = yaml.safe_load(response.text)
    return parsed["reference_answer"], parsed["required_claims"]


def main() -> None:
    questions = yaml.safe_load(_QUESTIONS_PATH.read_text())
    output = []
    for q in questions:
        if q.get("adversarial"):
            output.append(q)
            continue
        ref, claims = draft_reference(q["query"])
        q["reference_answer"] = ref
        q["required_claims"] = claims
        output.append(q)
        print(f"  {q['id']}: drafted", file=sys.stderr)
    print(yaml.dump(output, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
