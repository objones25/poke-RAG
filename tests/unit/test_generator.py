"""Unit tests for loader, inference, and generator orchestration.

All external model/tokenizer dependencies are mocked — no GPU required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.generation.models import GenerationConfig
from src.types import GenerationResult, RetrievedChunk, Source
from tests.conftest import make_chunk as _chunk

_GEMMA_ID = "google/gemma-2-2b-it"


# ---------------------------------------------------------------------------
# ModelLoader tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelLoader:
    def test_get_model_raises_before_load(self) -> None:
        from src.generation.loader import ModelLoader

        loader = ModelLoader(config=GenerationConfig(model_id=_GEMMA_ID), device="cpu")
        with pytest.raises(RuntimeError, match="load"):
            loader.get_model()

    def test_get_tokenizer_raises_before_load(self) -> None:
        from src.generation.loader import ModelLoader

        loader = ModelLoader(config=GenerationConfig(model_id=_GEMMA_ID), device="cpu")
        with pytest.raises(RuntimeError, match="load"):
            loader.get_tokenizer()

    def test_load_is_idempotent(self) -> None:
        from src.generation.loader import ModelLoader

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()

        loader = ModelLoader(config=GenerationConfig(model_id=_GEMMA_ID), device="cpu")
        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ) as mock_model,
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ) as mock_tok,
        ):
            loader.load()
            loader.load()

        mock_model.assert_called_once()
        mock_tok.assert_called_once()

    def test_get_model_returns_loaded_model(self) -> None:
        from src.generation.loader import ModelLoader

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()

        loader = ModelLoader(config=GenerationConfig(model_id=_GEMMA_ID), device="cpu")
        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            loader.load()

        assert loader.get_model() is fake_model

    def test_get_tokenizer_returns_loaded_tokenizer(self) -> None:
        from src.generation.loader import ModelLoader

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()

        loader = ModelLoader(config=GenerationConfig(model_id=_GEMMA_ID), device="cpu")
        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            loader.load()

        assert loader.get_tokenizer() is fake_tokenizer

    def test_load_passes_model_id(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        loader = ModelLoader(
            config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cpu"
        )
        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=MagicMock(),
            ) as mock_model,
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=MagicMock(),
            ),
            patch("src.generation.loader.torch.cuda.is_available", return_value=False),
        ):
            loader.load()

        mock_model.assert_called_once()
        assert mock_model.call_args[0][0] == "google/gemma-2-2b-it"


# ---------------------------------------------------------------------------
# Inferencer tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInferencer:
    def _make_inferencer(self) -> Any:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()

        # tokenizer returns a dict-like object; model.generate returns token ids
        encoded = MagicMock()
        encoded.__getitem__ = MagicMock(return_value=MagicMock())
        encoded.to = MagicMock(return_value=encoded)
        fake_tokenizer.return_value = encoded
        fake_model.device = "cpu"

        output_ids = MagicMock()
        fake_model.generate.return_value = output_ids
        fake_tokenizer.decode.return_value = "  Generated answer.  "

        config = GenerationConfig(model_id=_GEMMA_ID)
        return Inferencer(fake_model, fake_tokenizer, config), fake_model, fake_tokenizer

    def test_raises_on_empty_prompt(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        inferencer = Inferencer(MagicMock(), MagicMock(), GenerationConfig(model_id=_GEMMA_ID))
        with pytest.raises(ValueError, match="prompt"):
            inferencer.infer("")

    def test_raises_on_whitespace_prompt(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        inferencer = Inferencer(MagicMock(), MagicMock(), GenerationConfig(model_id=_GEMMA_ID))
        with pytest.raises(ValueError, match="prompt"):
            inferencer.infer("   ")

    def test_calls_model_generate(self) -> None:
        inferencer, fake_model, fake_tokenizer = self._make_inferencer()
        inferencer.infer("Tell me about Pikachu.")
        fake_model.generate.assert_called_once()

    def test_decode_called_with_skip_special_tokens(self) -> None:
        inferencer, fake_model, fake_tokenizer = self._make_inferencer()
        inferencer.infer("Tell me about Pikachu.")
        fake_tokenizer.decode.assert_called_once()
        _, kwargs = fake_tokenizer.decode.call_args
        assert kwargs.get("skip_special_tokens") is True

    def test_result_is_stripped_string(self) -> None:
        inferencer, _, _ = self._make_inferencer()
        result = inferencer.infer("Tell me about Pikachu.")
        assert result == "Generated answer."

    def test_generate_called_with_config_params(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        encoded = MagicMock()
        encoded.to = MagicMock(return_value=encoded)
        fake_tokenizer.return_value = encoded
        fake_model.device = "cpu"
        fake_tokenizer.decode.return_value = "answer"

        config = GenerationConfig(
            model_id=_GEMMA_ID, temperature=0.3, max_new_tokens=256, top_p=0.8
        )
        inferencer = Inferencer(fake_model, fake_tokenizer, config)
        inferencer.infer("Some prompt.")

        _, kwargs = fake_model.generate.call_args
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_new_tokens"] == 256
        assert kwargs["top_p"] == 0.8

    def test_tokenizer_called_with_tokenizer_config_params(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig, TokenizerConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        encoded = MagicMock()
        encoded.to = MagicMock(return_value=encoded)
        fake_tokenizer.return_value = encoded
        fake_model.device = "cpu"
        fake_tokenizer.decode.return_value = "answer"

        tok_config = TokenizerConfig(max_length=512, truncation=True, return_tensors="pt")
        inferencer = Inferencer(
            fake_model,
            fake_tokenizer,
            GenerationConfig(model_id=_GEMMA_ID),
            tokenizer_config=tok_config,
        )
        inferencer.infer("Some prompt.")

        _, kwargs = fake_tokenizer.call_args
        assert kwargs["max_length"] == 512
        assert kwargs["truncation"] is True
        assert kwargs["return_tensors"] == "pt"


# ---------------------------------------------------------------------------
# Generator orchestration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerator:
    def _make_generator(self, answer: str = "A great answer about Pikachu.") -> Any:
        from src.generation.generator import Generator
        from src.generation.models import GenerationConfig

        mock_loader = MagicMock()
        mock_loader.get_model.return_value = MagicMock()
        mock_loader.get_tokenizer.return_value = MagicMock()

        mock_prompt_builder = MagicMock(return_value="<built prompt>")

        mock_inferencer = MagicMock()
        mock_inferencer.infer.return_value = answer

        config = GenerationConfig(model_id=_GEMMA_ID)
        gen = Generator(
            loader=mock_loader,
            prompt_builder=mock_prompt_builder,
            inferencer=mock_inferencer,
            config=config,
        )
        return gen, mock_loader, mock_prompt_builder, mock_inferencer

    def test_raises_value_error_on_empty_chunks(self) -> None:
        gen, _, _, _ = self._make_generator()
        with pytest.raises(ValueError, match="chunks"):
            gen.generate("What type is Pikachu?", ())

    def test_calls_prompt_builder_with_query_and_chunks(self) -> None:
        gen, _, mock_pb, _ = self._make_generator()
        chunks = (_chunk(),)
        gen.generate("What type is Pikachu?", chunks)
        mock_pb.assert_called_once_with("What type is Pikachu?", chunks)

    def test_calls_inferencer_with_built_prompt(self) -> None:
        gen, _, _, mock_inf = self._make_generator()
        chunks = (_chunk(),)
        gen.generate("What type is Pikachu?", chunks)
        mock_inf.infer.assert_called_once_with("<built prompt>")

    def test_returns_generation_result(self) -> None:
        gen, _, _, _ = self._make_generator()
        result = gen.generate("What type is Pikachu?", (_chunk(),))
        assert isinstance(result, GenerationResult)

    def test_result_answer_matches_inferencer_output(self) -> None:
        gen, _, _, _ = self._make_generator(answer="Pikachu is Electric-type.")
        result = gen.generate("What type is Pikachu?", (_chunk(),))
        assert result.answer == "Pikachu is Electric-type."

    def test_sources_used_are_deduplicated_tuple(self) -> None:
        gen, _, _, _ = self._make_generator()
        chunks = (
            _chunk(source="bulbapedia", chunk_index=0),
            _chunk(source="bulbapedia", chunk_index=1),
            _chunk(source="smogon", chunk_index=2),
        )
        result = gen.generate("Question?", chunks)
        assert isinstance(result.sources_used, tuple)
        assert set(result.sources_used) == {"bulbapedia", "smogon"}
        assert len(result.sources_used) == 2

    def test_num_chunks_used_matches_input_length(self) -> None:
        gen, _, _, _ = self._make_generator()
        chunks = (_chunk(chunk_index=i) for i in range(3))
        result = gen.generate("Question?", tuple(chunks))
        assert result.num_chunks_used == 3

    def test_model_name_is_gemma_4(self) -> None:
        gen, _, _, _ = self._make_generator()
        result = gen.generate("Question?", (_chunk(),))
        assert result.model_name == "google/gemma-2-2b-it"

    def test_sources_used_are_sorted_alphabetically(self) -> None:
        gen, _, _, _ = self._make_generator()
        chunks = (
            _chunk(source="smogon", chunk_index=0),
            _chunk(source="bulbapedia", chunk_index=1),
            _chunk(source="pokeapi", chunk_index=2),
        )
        result = gen.generate("Question?", chunks)
        assert list(result.sources_used) == sorted(result.sources_used)

    def test_inferencer_failure_propagates(self) -> None:
        gen, _, _, mock_inf = self._make_generator()
        mock_inf.infer.side_effect = RuntimeError("GPU OOM")
        with pytest.raises(RuntimeError, match="GPU OOM"):
            gen.generate("Question?", (_chunk(),))
