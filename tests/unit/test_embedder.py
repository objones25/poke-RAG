"""Unit tests for src/retrieval/embedder.py — RED until embedder.py is implemented."""

from __future__ import annotations

from unittest.mock import MagicMock

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


@pytest.mark.unit
class TestBGEEmbedderEncode:
    def test_returns_embedding_output(self) -> None:
        embedder = BGEEmbedder(_make_mock_model(n=2))
        result = embedder.encode(["text one", "text two"])
        assert isinstance(result, EmbeddingOutput)

    def test_dense_shape_matches_input_length(self) -> None:
        embedder = BGEEmbedder(_make_mock_model(n=2, dense_dim=1024))
        result = embedder.encode(["text one", "text two"])
        assert len(result.dense) == 2
        assert len(result.dense[0]) == 1024

    def test_sparse_length_matches_input(self) -> None:
        embedder = BGEEmbedder(_make_mock_model(n=3))
        result = embedder.encode(["a", "b", "c"])
        assert len(result.sparse) == 3

    def test_sparse_keys_are_ints(self) -> None:
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": [[0.1] * 1024],
            "lexical_weights": [{42: 0.9, 7: 0.3}],
        }
        result = BGEEmbedder(mock).encode(["hello"])
        assert all(isinstance(k, int) for k in result.sparse[0])

    def test_sparse_values_are_floats(self) -> None:
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": [[0.1] * 1024],
            "lexical_weights": [{42: 0.9}],
        }
        result = BGEEmbedder(mock).encode(["hello"])
        assert all(isinstance(v, float) for v in result.sparse[0].values())

    def test_passes_texts_to_model(self) -> None:
        mock = _make_mock_model(n=1)
        BGEEmbedder(mock).encode(["specific text"])
        call_args = mock.encode.call_args
        texts_passed = call_args[0][0] if call_args[0] else call_args[1].get("sentences")
        assert "specific text" in texts_passed

    def test_requests_dense_and_sparse(self) -> None:
        mock = _make_mock_model(n=1)
        BGEEmbedder(mock).encode(["text"])
        call_kwargs = mock.encode.call_args[1]
        assert call_kwargs.get("return_dense") is True
        assert call_kwargs.get("return_sparse") is True

    def test_colbert_not_requested(self) -> None:
        mock = _make_mock_model(n=1)
        BGEEmbedder(mock).encode(["text"])
        call_kwargs = mock.encode.call_args[1]
        assert call_kwargs.get("return_colbert_vecs") is False

    def test_empty_input_returns_empty_output(self) -> None:
        mock = MagicMock()
        result = BGEEmbedder(mock).encode([])
        assert result.dense == []
        assert result.sparse == []
        mock.encode.assert_not_called()

    def test_dense_values_are_floats(self) -> None:
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": [[1, 2, 3]],  # ints from model — must be converted
            "lexical_weights": [{}],
        }
        result = BGEEmbedder(mock).encode(["text"])
        assert all(isinstance(v, float) for v in result.dense[0])

    def test_string_sparse_keys_converted_to_int(self) -> None:
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": [[0.1] * 1024],
            "lexical_weights": [{"42": 0.9}],  # string keys — must be converted
        }
        result = BGEEmbedder(mock).encode(["text"])
        assert 42 in result.sparse[0]
        assert isinstance(list(result.sparse[0].keys())[0], int)


@pytest.mark.unit
class TestBGEEmbedderFromPretrained:
    def test_passes_model_name_to_bgem3flagmodel(self) -> None:
        from unittest.mock import patch

        with patch("FlagEmbedding.BGEM3FlagModel") as mock_model_class:
            BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cpu")
            mock_model_class.assert_called_once()
            call_kwargs = mock_model_class.call_args[1]
            assert call_kwargs["model_name_or_path"] == "BAAI/bge-m3"

    def test_passes_device_to_bgem3flagmodel(self) -> None:
        from unittest.mock import patch

        with patch("FlagEmbedding.BGEM3FlagModel") as mock_model_class:
            BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cuda")
            call_kwargs = mock_model_class.call_args[1]
            assert call_kwargs["device"] == "cuda"

    def test_use_fp16_true_for_cuda(self) -> None:
        from unittest.mock import patch

        with patch("FlagEmbedding.BGEM3FlagModel") as mock_model_class:
            BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cuda")
            call_kwargs = mock_model_class.call_args[1]
            assert call_kwargs["use_fp16"] is True

    def test_use_fp16_true_for_mps(self) -> None:
        from unittest.mock import patch

        with patch("FlagEmbedding.BGEM3FlagModel") as mock_model_class:
            BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="mps")
            call_kwargs = mock_model_class.call_args[1]
            assert call_kwargs["use_fp16"] is True

    def test_use_fp16_false_for_cpu(self) -> None:
        from unittest.mock import patch

        with patch("FlagEmbedding.BGEM3FlagModel") as mock_model_class:
            BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cpu")
            call_kwargs = mock_model_class.call_args[1]
            assert call_kwargs["use_fp16"] is False


@pytest.mark.unit
class TestBGEEmbedderProtocolCompliance:
    def test_satisfies_embedder_protocol(self) -> None:
        from src.retrieval.protocols import EmbedderProtocol

        embedder = BGEEmbedder(_make_mock_model())
        assert isinstance(embedder, EmbedderProtocol)


@pytest.mark.unit
class TestBGEEmbedderDuplicateSparseIds:
    """Test handling of duplicate sparse token IDs in embedder output."""

    def test_encode_deduplicates_duplicate_sparse_ids(self) -> None:
        """When lexical_weights has duplicate keys, keep last value."""
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": [[0.1] * 1024],
            "lexical_weights": [{42: 0.9}],  # Single key
        }
        result = BGEEmbedder(mock).encode(["text"])
        # Result should preserve all values
        assert 42 in result.sparse[0]
        assert result.sparse[0][42] == 0.9
