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


class _DuplicateItems:
    """Iterable whose .items() yields the same key twice — simulates BGE-M3 duplicate token IDs."""

    def __init__(self, pairs: list[tuple[int, float]]) -> None:
        self._pairs = pairs

    def items(self) -> list[tuple[int, float]]:
        return self._pairs


@pytest.mark.unit
class TestBGEEmbedderDuplicateSparseIds:
    """Test handling of duplicate sparse token IDs in embedder output."""

    def test_encode_deduplicates_duplicate_sparse_ids(self) -> None:
        """When lexical_weights has a single key, value is preserved."""
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": [[0.1] * 1024],
            "lexical_weights": [{42: 0.9}],
        }
        result = BGEEmbedder(mock).encode(["text"])
        assert 42 in result.sparse[0]
        assert result.sparse[0][42] == 0.9

    def test_sparse_dedup_keeps_max_not_last(self) -> None:
        """When a token ID appears twice, the higher weight wins (not the last)."""
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": [[0.1] * 1024],
            "lexical_weights": [_DuplicateItems([(42, 0.5), (42, 0.3)])],
        }
        result = BGEEmbedder(mock).encode(["text"])
        assert result.sparse[0][42] == pytest.approx(0.5)


@pytest.mark.unit
class TestEmbeddingOutputValidation:
    def test_embedding_output_dense_dimension_validation(self) -> None:
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": [[0.1] * 512],
            "lexical_weights": [{}],
        }
        result = BGEEmbedder(mock).encode(["text"])
        assert len(result.dense[0]) == 512

    def test_embedding_output_dense_must_be_list_of_lists(self) -> None:
        output = EmbeddingOutput(dense=[[0.1, 0.2]], sparse=[{}])
        assert isinstance(output.dense, list)
        assert isinstance(output.dense[0], list)

    def test_embedding_output_sparse_must_be_list_of_dicts(self) -> None:
        output = EmbeddingOutput(dense=[], sparse=[{1: 0.5}])
        assert isinstance(output.sparse, list)
        assert isinstance(output.sparse[0], dict)

    def test_embedding_output_sparse_dict_keys_are_ints(self) -> None:
        output = EmbeddingOutput(dense=[], sparse=[{42: 0.9}])
        for k in output.sparse[0]:
            assert isinstance(k, int)

    def test_embedding_output_sparse_dict_values_are_floats(self) -> None:
        output = EmbeddingOutput(dense=[], sparse=[{42: 0.9}])
        for v in output.sparse[0].values():
            assert isinstance(v, float)

    def test_embedding_output_frozen_immutability(self) -> None:
        output = EmbeddingOutput(dense=[[0.1]], sparse=[{}])
        with pytest.raises(AttributeError):
            output.dense = [[0.2]]

    def test_embedding_output_multiple_sparse_dicts(self) -> None:
        output = EmbeddingOutput(dense=[[0.1], [0.2]], sparse=[{1: 0.5}, {2: 0.3}])
        assert len(output.sparse) == 2
        assert 1 in output.sparse[0]
        assert 2 in output.sparse[1]


@pytest.mark.unit
class TestBGEEmbedderErrorHandling:
    def test_encode_raises_when_model_encode_raises(self) -> None:
        mock = MagicMock()
        mock.encode.side_effect = RuntimeError("Model inference failed")
        embedder = BGEEmbedder(mock)
        with pytest.raises(RuntimeError, match="Model inference failed"):
            embedder.encode(["text"])

    def test_encode_raises_on_value_error_from_model(self) -> None:
        mock = MagicMock()
        mock.encode.side_effect = ValueError("Invalid input shape")
        embedder = BGEEmbedder(mock)
        with pytest.raises(ValueError, match="Invalid input shape"):
            embedder.encode(["text"])

    def test_encode_raises_on_cuda_error_from_model(self) -> None:
        mock = MagicMock()
        mock.encode.side_effect = RuntimeError("CUDA out of memory")
        embedder = BGEEmbedder(mock)
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            embedder.encode(["text"])

    def test_encode_handles_missing_dense_vecs_key(self) -> None:
        mock = MagicMock()
        mock.encode.return_value = {
            "lexical_weights": [{}],
        }
        embedder = BGEEmbedder(mock)
        with pytest.raises(KeyError):
            embedder.encode(["text"])

    def test_encode_handles_missing_lexical_weights_key(self) -> None:
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": [[0.1] * 1024],
        }
        embedder = BGEEmbedder(mock)
        with pytest.raises(KeyError):
            embedder.encode(["text"])

    def test_encode_with_fewer_sparse_than_dense_vecs(self) -> None:
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": [[0.1] * 1024, [0.2] * 1024],
            "lexical_weights": [{}],
        }
        embedder = BGEEmbedder(mock)
        result = embedder.encode(["text1", "text2"])
        assert len(result.dense) == 2
        assert len(result.sparse) == 1


@pytest.mark.unit
class TestBGEEmbedderEdgeCases:
    def test_single_item_input(self) -> None:
        mock = _make_mock_model(n=1)
        result = BGEEmbedder(mock).encode(["single text"])
        assert len(result.dense) == 1
        assert len(result.sparse) == 1

    def test_very_long_string_input(self) -> None:
        mock = _make_mock_model(n=1)
        long_text = "word " * 10000
        result = BGEEmbedder(mock).encode([long_text])
        assert len(result.dense) == 1
        assert len(result.sparse) == 1

    def test_many_items_input(self) -> None:
        n = 100
        mock = _make_mock_model(n=n)
        texts = [f"text_{i}" for i in range(n)]
        result = BGEEmbedder(mock).encode(texts)
        assert len(result.dense) == n
        assert len(result.sparse) == n

    def test_special_characters_in_text(self) -> None:
        mock = _make_mock_model(n=1)
        special_text = "Pokémon: Bulbasaur's type is Grass/Poison! 🔥"
        result = BGEEmbedder(mock).encode([special_text])
        assert len(result.dense) == 1

    def test_empty_sparse_weights(self) -> None:
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": [[0.1] * 1024],
            "lexical_weights": [{}],
        }
        result = BGEEmbedder(mock).encode(["text"])
        assert result.sparse[0] == {}

    def test_dense_values_within_valid_range(self) -> None:
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": [[0.5] * 1024],
            "lexical_weights": [{}],
        }
        result = BGEEmbedder(mock).encode(["text"])
        assert all(-1.0 <= v <= 1.0 for v in result.dense[0])


def _make_mock_model_colbert(n: int = 1, seq_len: int = 4, dense_dim: int = 1024) -> MagicMock:
    """Return a MagicMock that mimics BGEM3FlagModel.encode() output with colbert_vecs."""
    import numpy as np

    mock = MagicMock()
    mock.encode.return_value = {
        "dense_vecs": [[0.1] * dense_dim for _ in range(n)],
        "lexical_weights": [{i: 0.5 for i in range(3)} for _ in range(n)],
        "colbert_vecs": [np.ones((seq_len, dense_dim), dtype="float32") for _ in range(n)],
    }
    return mock


@pytest.mark.unit
class TestBGEEmbedderColBERT:
    def test_colbert_requested_when_enabled(self) -> None:
        mock = _make_mock_model_colbert(n=1)
        BGEEmbedder(mock, colbert_enabled=True).encode(["text"])
        call_kwargs = mock.encode.call_args[1]
        assert call_kwargs.get("return_colbert_vecs") is True

    def test_colbert_not_requested_when_disabled(self) -> None:
        mock = _make_mock_model(n=1)
        BGEEmbedder(mock, colbert_enabled=False).encode(["text"])
        call_kwargs = mock.encode.call_args[1]
        assert call_kwargs.get("return_colbert_vecs") is False

    def test_colbert_field_populated_when_enabled(self) -> None:
        mock = _make_mock_model_colbert(n=2)
        result = BGEEmbedder(mock, colbert_enabled=True).encode(["a", "b"])
        assert result.colbert is not None
        assert len(result.colbert) == 2

    def test_colbert_field_is_none_when_disabled(self) -> None:
        mock = _make_mock_model(n=1)
        result = BGEEmbedder(mock, colbert_enabled=False).encode(["text"])
        assert result.colbert is None

    def test_colbert_token_vectors_are_float_lists(self) -> None:
        mock = _make_mock_model_colbert(n=1, seq_len=5)
        result = BGEEmbedder(mock, colbert_enabled=True).encode(["text"])
        assert result.colbert is not None
        for doc_vecs in result.colbert:
            for token_vec in doc_vecs:
                assert all(isinstance(v, float) for v in token_vec)

    def test_colbert_token_vectors_have_correct_dim(self) -> None:
        mock = _make_mock_model_colbert(n=1, seq_len=3, dense_dim=1024)
        result = BGEEmbedder(mock, colbert_enabled=True).encode(["text"])
        assert result.colbert is not None
        assert len(result.colbert[0][0]) == 1024

    def test_colbert_empty_input_returns_empty_list(self) -> None:
        mock = MagicMock()
        result = BGEEmbedder(mock, colbert_enabled=True).encode([])
        assert result.colbert == []
        mock.encode.assert_not_called()

    def test_colbert_empty_input_disabled_returns_none(self) -> None:
        mock = MagicMock()
        result = BGEEmbedder(mock, colbert_enabled=False).encode([])
        assert result.colbert is None

    def test_from_pretrained_passes_colbert_enabled(self) -> None:
        from unittest.mock import patch

        with patch("FlagEmbedding.BGEM3FlagModel"):
            embedder = BGEEmbedder.from_pretrained(
                model_name="BAAI/bge-m3", device="cpu", colbert_enabled=True
            )
            assert embedder._colbert_enabled is True

    def test_from_pretrained_colbert_disabled_by_default(self) -> None:
        from unittest.mock import patch

        with patch("FlagEmbedding.BGEM3FlagModel"):
            embedder = BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cpu")
            assert embedder._colbert_enabled is False
