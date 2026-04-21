from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from scripts.training.inference_runner import (
    CandidateWithContext,
    fetch_reference_context,
    format_context,
    generate_candidates,
)
from src.types import RetrievalError, RetrievedChunk, RetrievalResult


def _make_chunk(text: str, source: str = "pokeapi") -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=0.9,
        source=source,  # type: ignore[arg-type]
        entity_name="Pikachu",
        entity_type="pokemon",
        chunk_index=0,
        original_doc_id="doc_0",
    )


def _make_retrieval_result(texts: list[str], source: str = "pokeapi") -> RetrievalResult:
    return RetrievalResult(
        documents=tuple(_make_chunk(t, source) for t in texts),
        query="test",
    )


def _make_processor(response: str = "Pikachu is Electric-type.") -> MagicMock:
    processor = MagicMock()
    processor.apply_chat_template.return_value = "<tmpl>prompt</tmpl>"
    encoded = MagicMock()
    encoded.__contains__ = MagicMock(return_value=False)
    input_ids = torch.zeros((1, 10), dtype=torch.long)
    encoded.__getitem__ = MagicMock(return_value=input_ids)
    processor.return_value = encoded
    processor.decode.return_value = response
    return processor


def _make_model() -> MagicMock:
    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.zeros((1, 15), dtype=torch.long)
    return model


@pytest.mark.unit
class TestFormatContext:
    def test_returns_text_list(self) -> None:
        chunks = [_make_chunk("chunk A"), _make_chunk("chunk B")]
        assert format_context(chunks) == ["chunk A", "chunk B"]

    def test_empty_input(self) -> None:
        assert format_context([]) == []


@pytest.mark.unit
class TestFetchReferenceContext:
    def test_retrieves_from_bulbapedia_and_pokeapi_only(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_retrieval_result(["ref chunk"])

        result = fetch_reference_context("What is Pikachu?", mock_retriever)

        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert set(call_kwargs["sources"]) == {"bulbapedia", "pokeapi"}
        assert result == ["ref chunk"]

    def test_returns_empty_on_retrieval_error(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = RetrievalError("no docs")

        result = fetch_reference_context("q", mock_retriever)

        assert result == []


@pytest.mark.unit
class TestGenerateCandidates:
    def test_returns_k_candidates(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_retrieval_result(["Pikachu chunk"])

        candidates = generate_candidates(
            "What type is Pikachu?",
            mock_retriever,
            _make_model(),
            _make_processor(),
            k=3,
        )

        assert len(candidates) == 3
        assert all(isinstance(c, CandidateWithContext) for c in candidates)

    def test_candidate_has_response_and_retrieved_chunks(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_retrieval_result(["chunk text"])

        candidates = generate_candidates(
            "What type is Pikachu?",
            mock_retriever,
            _make_model(),
            _make_processor("Pikachu is an Electric-type Pokémon."),
            k=1,
        )

        assert candidates[0].response == "Pikachu is an Electric-type Pokémon."
        assert candidates[0].retrieved_chunks == ["chunk text"]

    def test_retriever_called_once_for_all_candidates(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_retrieval_result(["chunk"])

        generate_candidates("q", mock_retriever, _make_model(), _make_processor(), k=5)

        mock_retriever.retrieve.assert_called_once()

    def test_uses_sampling_for_diversity(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_retrieval_result(["chunk"])
        model = _make_model()

        generate_candidates("q", mock_retriever, model, _make_processor(), k=1)

        gen_kwargs = model.generate.call_args.kwargs
        assert gen_kwargs.get("do_sample") is True
