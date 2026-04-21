"""Integration tests for src/retrieval/embedder.py — tests the from_pretrained factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.embedder import BGEEmbedder
from src.retrieval.types import EmbeddingOutput


def _make_mock_model(dense_dim: int = 1024, n: int = 1) -> MagicMock:
    """Return a MagicMock that mimics BGEM3FlagModel.encode() output."""
    mock = MagicMock()
    mock.encode.return_value = {
        "dense_vecs": [[0.1] * dense_dim for _ in range(n)],
        "lexical_weights": [{i: 0.5 for i in range(3)} for _ in range(n)],
    }
    return mock


@pytest.mark.integration
class TestBGEEmbedderFromPretrained:
    """Test BGEEmbedder.from_pretrained() factory method."""

    def test_from_pretrained_passes_model_name(self) -> None:
        """Verify that model_name is passed to BGEM3FlagModel."""
        mock_model_instance = MagicMock()
        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model_instance) as mock_cls:
            BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cpu")
        mock_cls.assert_called_once()
        assert mock_cls.call_args[1]["model_name_or_path"] == "BAAI/bge-m3"

    def test_from_pretrained_uses_fp16_on_cuda(self) -> None:
        """Verify that use_fp16=True when device is cuda."""
        mock_model_instance = MagicMock()
        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model_instance) as mock_cls:
            BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cuda")
        assert mock_cls.call_args[1]["use_fp16"] is True

    def test_from_pretrained_uses_fp16_on_mps(self) -> None:
        """Verify that use_fp16=True when device is mps (Apple Silicon)."""
        mock_model_instance = MagicMock()
        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model_instance) as mock_cls:
            BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="mps")
        assert mock_cls.call_args[1]["use_fp16"] is True

    def test_from_pretrained_no_fp16_on_cpu(self) -> None:
        """Verify that use_fp16=False when device is cpu."""
        mock_model_instance = MagicMock()
        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model_instance) as mock_cls:
            BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cpu")
        assert mock_cls.call_args[1]["use_fp16"] is False

    def test_from_pretrained_passes_device(self) -> None:
        """Verify that device is passed to BGEM3FlagModel."""
        mock_model_instance = MagicMock()
        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model_instance) as mock_cls:
            BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cuda:0")
        assert mock_cls.call_args[1]["device"] == "cuda:0"

    def test_from_pretrained_returns_bge_embedder_instance(self) -> None:
        """Verify that from_pretrained returns a BGEEmbedder instance."""
        mock_model_instance = MagicMock()
        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model_instance):
            result = BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cpu")
        assert isinstance(result, BGEEmbedder)

    def test_from_pretrained_wraps_model_instance(self) -> None:
        """Verify that the returned embedder wraps the instantiated model."""
        mock_model_instance = MagicMock()
        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model_instance):
            embedder = BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cpu")
        # The embedder should have the model accessible (test internal structure)
        assert embedder._model is mock_model_instance

    def test_from_pretrained_different_model_names(self) -> None:
        """Verify that from_pretrained works with different model names."""
        model_names = ["BAAI/bge-m3", "BAAI/bge-large-zh-v1.5", "custom/model"]
        for model_name in model_names:
            mock_model_instance = MagicMock()
            with patch(
                "FlagEmbedding.BGEM3FlagModel", return_value=mock_model_instance
            ) as mock_cls:
                BGEEmbedder.from_pretrained(model_name=model_name, device="cpu")
            assert mock_cls.call_args[1]["model_name_or_path"] == model_name

    def test_from_pretrained_different_devices(self) -> None:
        """Verify that from_pretrained works with different devices."""
        devices = ["cpu", "cuda", "cuda:0", "mps"]
        for device in devices:
            mock_model_instance = MagicMock()
            with patch(
                "FlagEmbedding.BGEM3FlagModel", return_value=mock_model_instance
            ) as mock_cls:
                BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device=device)
            assert mock_cls.call_args[1]["device"] == device


@pytest.mark.integration
class TestBGEEmbedderEncodeIntegration:
    """Integration tests for BGEEmbedder.encode() with mock model."""

    def test_encode_calls_model_with_correct_flags(self) -> None:
        """Verify that encode() calls model with correct return_* flags."""
        mock = _make_mock_model(n=1)
        embedder = BGEEmbedder(mock)
        embedder.encode(["test text"])
        call_kwargs = mock.encode.call_args[1]
        assert call_kwargs["return_dense"] is True
        assert call_kwargs["return_sparse"] is True
        assert call_kwargs["return_colbert_vecs"] is False

    def test_encode_passes_all_texts(self) -> None:
        """Verify that encode() passes all input texts to the model."""
        mock = _make_mock_model(n=3)
        embedder = BGEEmbedder(mock)
        input_texts = ["text a", "text b", "text c"]
        embedder.encode(input_texts)
        call_args = mock.encode.call_args[0][0]
        assert call_args == input_texts

    def test_output_dense_is_list_of_lists_of_float(self) -> None:
        """Verify that dense output is list[list[float]]."""
        mock = _make_mock_model(n=2, dense_dim=1024)
        embedder = BGEEmbedder(mock)
        result = embedder.encode(["text one", "text two"])
        assert isinstance(result.dense, list)
        assert len(result.dense) == 2
        for vec in result.dense:
            assert isinstance(vec, list)
            for val in vec:
                assert isinstance(val, float)

    def test_output_sparse_is_list_of_dict_int_float(self) -> None:
        """Verify that sparse output is list[dict[int, float]]."""
        mock = _make_mock_model(n=2)
        embedder = BGEEmbedder(mock)
        result = embedder.encode(["text one", "text two"])
        assert isinstance(result.sparse, list)
        assert len(result.sparse) == 2
        for weights_dict in result.sparse:
            assert isinstance(weights_dict, dict)
            for key, value in weights_dict.items():
                assert isinstance(key, int)
                assert isinstance(value, float)

    def test_tensor_keys_converted_to_int(self) -> None:
        """Verify that string keys in lexical_weights are converted to int."""
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": [[0.1] * 1024],
            "lexical_weights": [{"42": 0.9, "7": 0.3, "999": 0.1}],
        }
        embedder = BGEEmbedder(mock)
        result = embedder.encode(["test"])
        assert 42 in result.sparse[0]
        assert 7 in result.sparse[0]
        assert 999 in result.sparse[0]
        assert all(isinstance(k, int) for k in result.sparse[0])

    def test_tensor_values_converted_to_float(self) -> None:
        """Verify that sparse values are converted to float."""
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": [[0.1] * 1024],
            "lexical_weights": [{42: 0.9, 7: 1, 999: 0}],  # int and float values
        }
        embedder = BGEEmbedder(mock)
        result = embedder.encode(["test"])
        assert all(isinstance(v, float) for v in result.sparse[0].values())

    def test_multiple_texts_produce_matching_lengths(self) -> None:
        """Verify that encoding N texts produces N dense and N sparse outputs."""
        for n in [1, 2, 4, 10]:
            mock = _make_mock_model(n=n, dense_dim=1024)
            embedder = BGEEmbedder(mock)
            texts = [f"text {i}" for i in range(n)]
            result = embedder.encode(texts)
            assert len(result.dense) == n
            assert len(result.sparse) == n

    def test_encode_returns_embedding_output(self) -> None:
        """Verify that encode() returns an EmbeddingOutput instance."""
        mock = _make_mock_model(n=1)
        embedder = BGEEmbedder(mock)
        result = embedder.encode(["test"])
        assert isinstance(result, EmbeddingOutput)

    def test_encode_preserves_dense_dimension(self) -> None:
        """Verify that dense vector dimensions are preserved."""
        for dim in [256, 768, 1024]:
            mock = _make_mock_model(n=2, dense_dim=dim)
            embedder = BGEEmbedder(mock)
            result = embedder.encode(["a", "b"])
            assert all(len(vec) == dim for vec in result.dense)

    def test_encode_handles_large_batch(self) -> None:
        """Verify that encode() handles a large batch of texts."""
        n = 100
        mock = _make_mock_model(n=n, dense_dim=1024)
        embedder = BGEEmbedder(mock)
        texts = [f"text {i}" for i in range(n)]
        result = embedder.encode(texts)
        assert len(result.dense) == n
        assert len(result.sparse) == n

    def test_encode_with_empty_input(self) -> None:
        """Verify that encode() returns empty output for empty input without calling model."""
        mock = MagicMock()
        embedder = BGEEmbedder(mock)
        result = embedder.encode([])
        assert result.dense == []
        assert result.sparse == []
        mock.encode.assert_not_called()

    def test_encode_with_special_characters(self) -> None:
        """Verify that encode() handles texts with special characters."""
        mock = _make_mock_model(n=3)
        embedder = BGEEmbedder(mock)
        texts = [
            "Pokémon with café and naïve",
            "Text with emoji: 🔥⚡",
            "SQL injection attempt: '; DROP TABLE--",
        ]
        embedder.encode(texts)
        call_args = mock.encode.call_args[0][0]
        assert call_args == texts

    def test_encode_with_long_text(self) -> None:
        """Verify that encode() handles very long texts."""
        mock = _make_mock_model(n=1)
        embedder = BGEEmbedder(mock)
        long_text = "word " * 1000  # ~5000 characters
        result = embedder.encode([long_text])
        assert len(result.dense) == 1
        assert len(result.sparse) == 1

    def test_encode_with_empty_string(self) -> None:
        """Verify that encode() handles empty strings in input."""
        mock = _make_mock_model(n=2)
        embedder = BGEEmbedder(mock)
        texts = ["", "non-empty text"]
        result = embedder.encode(texts)
        assert len(result.dense) == 2
        assert len(result.sparse) == 2

    def test_encode_with_whitespace_only_text(self) -> None:
        """Verify that encode() handles whitespace-only texts."""
        mock = _make_mock_model(n=2)
        embedder = BGEEmbedder(mock)
        texts = ["   ", "\t\n"]
        result = embedder.encode(texts)
        assert len(result.dense) == 2
        assert len(result.sparse) == 2

    def test_encode_preserves_text_order(self) -> None:
        """Verify that the order of outputs matches the order of inputs."""
        mock = MagicMock()
        # Create mock that returns outputs in same order as inputs
        mock.encode.return_value = {
            "dense_vecs": [[float(i)] * 1024 for i in range(3)],
            "lexical_weights": [{i: float(i) * 0.1} for i in range(3)],
        }
        embedder = BGEEmbedder(mock)
        texts = ["first", "second", "third"]
        result = embedder.encode(texts)
        # Check that dense vectors are in order (0.0, 1.0, 2.0)
        assert result.dense[0][0] == 0.0
        assert result.dense[1][0] == 1.0
        assert result.dense[2][0] == 2.0

    def test_encode_model_not_called_for_empty_input(self) -> None:
        """Verify that the model is not called when input is empty."""
        mock = MagicMock()
        embedder = BGEEmbedder(mock)
        embedder.encode([])
        mock.encode.assert_not_called()

    def test_encode_with_mixed_empty_and_nonempty(self) -> None:
        """Verify encode with mix of empty and non-empty texts."""
        mock = _make_mock_model(n=3)
        embedder = BGEEmbedder(mock)
        texts = ["text1", "", "text3"]
        result = embedder.encode(texts)
        assert len(result.dense) == 3
        assert len(result.sparse) == 3


@pytest.mark.integration
class TestBGEEmbedderFromPretrainedIntegration:
    """Test the full from_pretrained → encode workflow."""

    def test_from_pretrained_embedder_can_encode(self) -> None:
        """Verify that embedder from from_pretrained can encode texts."""
        mock_model_instance = _make_mock_model(n=2)
        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model_instance):
            embedder = BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cpu")
        result = embedder.encode(["text one", "text two"])
        assert isinstance(result, EmbeddingOutput)
        assert len(result.dense) == 2
        assert len(result.sparse) == 2

    def test_from_pretrained_with_cuda_produces_valid_output(self) -> None:
        """Verify full workflow works with cuda device."""
        mock_model_instance = _make_mock_model(n=1)
        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model_instance) as mock_cls:
            embedder = BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cuda")
            result = embedder.encode(["test"])
        # Verify fp16 was used
        assert mock_cls.call_args[1]["use_fp16"] is True
        # Verify encode works
        assert isinstance(result, EmbeddingOutput)

    def test_from_pretrained_initialization_suppresses_warnings(self) -> None:
        """Verify that from_pretrained suppresses the fast tokenizer warning."""
        mock_model_instance = MagicMock()
        with (
            patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model_instance),
            patch("warnings.filterwarnings") as mock_warnings,
        ):
            BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cpu")
            # Verify that filterwarnings was called to ignore fast tokenizer warning
            mock_warnings.assert_called()
            call_args = mock_warnings.call_args
            assert "fast tokenizer" in call_args[1]["message"]
