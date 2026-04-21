from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.generation.prompt_builder import build_prompt
from src.retrieval.protocols import RetrieverProtocol
from src.types import RetrievalError, RetrievedChunk


@dataclass
class CandidateWithContext:
    response: str
    retrieved_chunks: list[str]


def format_context(chunks: list[RetrievedChunk]) -> list[str]:
    return [chunk.text for chunk in chunks]


def fetch_reference_context(query: str, retriever: RetrieverProtocol) -> list[str]:
    try:
        result = retriever.retrieve(query, sources=["bulbapedia", "pokeapi"])
        return format_context(list(result.documents))
    except RetrievalError:
        return []


def generate_candidates(
    question: str,
    retriever: RetrieverProtocol,
    model: Any,
    processor: Any,
    *,
    k: int = 5,
    max_new_tokens: int = 512,
) -> list[CandidateWithContext]:
    result = retriever.retrieve(question)
    chunks = list(result.documents)
    retrieved_texts = format_context(chunks)
    prompt_text = build_prompt(question, tuple(chunks))
    messages = [{"role": "user", "content": prompt_text}]

    candidates: list[CandidateWithContext] = []
    for _ in range(k):
        templated = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = processor(text=templated, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        input_len = input_ids.shape[-1]
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )
        response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        candidates.append(CandidateWithContext(response=response, retrieved_chunks=retrieved_texts))

    return candidates
