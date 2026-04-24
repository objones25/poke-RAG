"""Unit tests for src/config.py — Settings and device detection."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestParseFloatInRange:
    def test_returns_default_when_value_is_none(self) -> None:
        from src.config import _parse_float_in_range

        result = _parse_float_in_range(None, "TEST_VAR", 0.5, 0.0, 1.0)
        assert result == 0.5

    def test_parses_valid_float_string(self) -> None:
        from src.config import _parse_float_in_range

        result = _parse_float_in_range("0.75", "TEST_VAR", 0.5, 0.0, 1.0)
        assert result == pytest.approx(0.75)

    def test_raises_on_non_numeric_string(self) -> None:
        from src.config import _parse_float_in_range

        with pytest.raises(ValueError, match="TEST_VAR"):
            _parse_float_in_range("abc", "TEST_VAR", 0.5, 0.0, 1.0)

    def test_raises_when_below_min_val(self) -> None:
        from src.config import _parse_float_in_range

        with pytest.raises(ValueError, match="TEST_VAR.*between 0.0 and 1.0"):
            _parse_float_in_range("-0.1", "TEST_VAR", 0.5, 0.0, 1.0)

    def test_raises_when_above_max_val(self) -> None:
        from src.config import _parse_float_in_range

        with pytest.raises(ValueError, match="TEST_VAR.*between 0.0 and 1.0"):
            _parse_float_in_range("1.1", "TEST_VAR", 0.5, 0.0, 1.0)

    def test_accepts_value_at_min_boundary(self) -> None:
        from src.config import _parse_float_in_range

        result = _parse_float_in_range("0.0", "TEST_VAR", 0.5, 0.0, 1.0)
        assert result == 0.0

    def test_accepts_value_at_max_boundary(self) -> None:
        from src.config import _parse_float_in_range

        result = _parse_float_in_range("1.0", "TEST_VAR", 0.5, 0.0, 1.0)
        assert result == 1.0

    def test_error_message_includes_env_var_name(self) -> None:
        from src.config import _parse_float_in_range

        with pytest.raises(ValueError) as exc_info:
            _parse_float_in_range("abc", "CUSTOM_TEMP", 0.5, 0.0, 1.0)
        assert "CUSTOM_TEMP" in str(exc_info.value)

    def test_error_message_includes_bad_value(self) -> None:
        from src.config import _parse_float_in_range

        with pytest.raises(ValueError) as exc_info:
            _parse_float_in_range("not_float", "TEST_VAR", 0.5, 0.0, 1.0)
        assert "not_float" in str(exc_info.value)

    def test_handles_scientific_notation(self) -> None:
        from src.config import _parse_float_in_range

        result = _parse_float_in_range("1e-3", "TEST_VAR", 0.5, 0.0, 1.0)
        assert result == pytest.approx(0.001)

    def test_raises_on_empty_string(self) -> None:
        from src.config import _parse_float_in_range

        with pytest.raises(ValueError, match="TEST_VAR"):
            _parse_float_in_range("", "TEST_VAR", 0.5, 0.0, 1.0)


@pytest.mark.unit
class TestParseFloatInRangeOptional:
    def test_returns_none_when_value_is_none(self) -> None:
        from src.config import _parse_float_in_range_optional

        result = _parse_float_in_range_optional(None, "TEST_VAR", 0.0, 1.0)
        assert result is None

    def test_parses_valid_float_string(self) -> None:
        from src.config import _parse_float_in_range_optional

        result = _parse_float_in_range_optional("0.75", "TEST_VAR", 0.0, 1.0)
        assert result == pytest.approx(0.75)

    def test_raises_on_non_numeric_string(self) -> None:
        from src.config import _parse_float_in_range_optional

        with pytest.raises(ValueError, match="TEST_VAR"):
            _parse_float_in_range_optional("abc", "TEST_VAR", 0.0, 1.0)

    def test_raises_when_below_min_val(self) -> None:
        from src.config import _parse_float_in_range_optional

        with pytest.raises(ValueError, match="TEST_VAR.*between 0.0 and 1.0"):
            _parse_float_in_range_optional("-0.1", "TEST_VAR", 0.0, 1.0)

    def test_raises_when_above_max_val(self) -> None:
        from src.config import _parse_float_in_range_optional

        with pytest.raises(ValueError, match="TEST_VAR.*between 0.0 and 1.0"):
            _parse_float_in_range_optional("1.1", "TEST_VAR", 0.0, 1.0)

    def test_accepts_value_at_min_boundary(self) -> None:
        from src.config import _parse_float_in_range_optional

        result = _parse_float_in_range_optional("0.0", "TEST_VAR", 0.0, 1.0)
        assert result == 0.0

    def test_accepts_value_at_max_boundary(self) -> None:
        from src.config import _parse_float_in_range_optional

        result = _parse_float_in_range_optional("1.0", "TEST_VAR", 0.0, 1.0)
        assert result == 1.0


@pytest.mark.unit
class TestParseIntPositive:
    def test_returns_default_when_value_is_none(self) -> None:
        from src.config import _parse_int_positive

        result = _parse_int_positive(None, "TEST_VAR", 100)
        assert result == 100

    def test_parses_valid_int_string(self) -> None:
        from src.config import _parse_int_positive

        result = _parse_int_positive("42", "TEST_VAR", 100)
        assert result == 42

    def test_raises_on_float_string(self) -> None:
        from src.config import _parse_int_positive

        with pytest.raises(ValueError, match="TEST_VAR"):
            _parse_int_positive("1.5", "TEST_VAR", 100)

    def test_raises_on_non_numeric_string(self) -> None:
        from src.config import _parse_int_positive

        with pytest.raises(ValueError, match="TEST_VAR"):
            _parse_int_positive("abc", "TEST_VAR", 100)

    def test_raises_on_zero(self) -> None:
        from src.config import _parse_int_positive

        with pytest.raises(ValueError, match="TEST_VAR.*positive"):
            _parse_int_positive("0", "TEST_VAR", 100)

    def test_raises_on_negative(self) -> None:
        from src.config import _parse_int_positive

        with pytest.raises(ValueError, match="TEST_VAR.*positive"):
            _parse_int_positive("-1", "TEST_VAR", 100)

    def test_accepts_minimum_positive_value(self) -> None:
        from src.config import _parse_int_positive

        result = _parse_int_positive("1", "TEST_VAR", 100)
        assert result == 1

    def test_error_message_includes_env_var_name(self) -> None:
        from src.config import _parse_int_positive

        with pytest.raises(ValueError) as exc_info:
            _parse_int_positive("abc", "CUSTOM_INT", 100)
        assert "CUSTOM_INT" in str(exc_info.value)

    def test_error_message_on_invalid_int(self) -> None:
        from src.config import _parse_int_positive

        with pytest.raises(ValueError) as exc_info:
            _parse_int_positive("not_int", "TEST_VAR", 100)
        assert "not_int" in str(exc_info.value)


@pytest.mark.unit
class TestParseBool:
    def test_returns_default_when_value_is_none(self) -> None:
        from src.config import _parse_bool

        assert _parse_bool(None, "TEST_VAR", True) is True
        assert _parse_bool(None, "TEST_VAR", False) is False

    def test_parses_true_variants(self) -> None:
        from src.config import _parse_bool

        assert _parse_bool("true", "TEST_VAR", False) is True
        assert _parse_bool("True", "TEST_VAR", False) is True
        assert _parse_bool("TRUE", "TEST_VAR", False) is True
        assert _parse_bool("1", "TEST_VAR", False) is True
        assert _parse_bool("yes", "TEST_VAR", False) is True
        assert _parse_bool("YES", "TEST_VAR", False) is True

    def test_parses_false_variants(self) -> None:
        from src.config import _parse_bool

        assert _parse_bool("false", "TEST_VAR", True) is False
        assert _parse_bool("False", "TEST_VAR", True) is False
        assert _parse_bool("FALSE", "TEST_VAR", True) is False
        assert _parse_bool("0", "TEST_VAR", True) is False
        assert _parse_bool("no", "TEST_VAR", True) is False
        assert _parse_bool("NO", "TEST_VAR", True) is False

    def test_raises_on_invalid_value(self) -> None:
        from src.config import _parse_bool

        with pytest.raises(ValueError, match="TEST_VAR"):
            _parse_bool("maybe", "TEST_VAR", True)

    def test_error_message_includes_env_var_name(self) -> None:
        from src.config import _parse_bool

        with pytest.raises(ValueError) as exc_info:
            _parse_bool("invalid", "CUSTOM_BOOL", True)
        assert "CUSTOM_BOOL" in str(exc_info.value)

    def test_strips_whitespace(self) -> None:
        from src.config import _parse_bool

        assert _parse_bool("  true  ", "TEST_VAR", False) is True
        assert _parse_bool("  false  ", "TEST_VAR", True) is False


@pytest.mark.unit
class TestDetectDevice:
    def test_returns_cuda_when_available(self) -> None:
        from src.config import _detect_device

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            assert _detect_device() == "cuda"

    def test_returns_mps_when_cuda_unavailable_but_mps_available(self) -> None:
        from src.config import _detect_device

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            assert _detect_device() == "mps"

    def test_returns_cpu_when_neither_cuda_nor_mps_available(self) -> None:
        from src.config import _detect_device

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            assert _detect_device() == "cpu"

    def test_cuda_takes_precedence_over_mps(self) -> None:
        from src.config import _detect_device

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            assert _detect_device() == "cuda"


@pytest.mark.unit
class TestSettingsFromEnv:
    def test_reads_required_qdrant_url(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        settings = Settings.from_env()
        assert settings.qdrant_url == "http://localhost:6333"

    def test_raises_keyerror_when_qdrant_url_missing(self) -> None:
        from src.config import Settings

        with patch.dict(os.environ, {}, clear=True), pytest.raises(KeyError):
            Settings.from_env()

    def test_reads_optional_qdrant_api_key(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "secret_key_123")
        settings = Settings.from_env()
        assert settings.qdrant_api_key is not None
        assert settings.qdrant_api_key.get_secret_value() == "secret_key_123"

    def test_qdrant_api_key_none_when_not_set(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        settings = Settings.from_env()
        assert settings.qdrant_api_key is None

    def test_reads_embed_model_with_default(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("EMBED_MODEL", raising=False)
        settings = Settings.from_env()
        assert settings.embed_model == "BAAI/bge-m3"

    def test_reads_custom_embed_model(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("EMBED_MODEL", "custom/model")
        settings = Settings.from_env()
        assert settings.embed_model == "custom/model"

    def test_reads_rerank_model_with_default(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("RERANK_MODEL", raising=False)
        settings = Settings.from_env()
        assert settings.rerank_model == "BAAI/bge-reranker-v2-m3"

    def test_reads_gen_model_with_default(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("GEN_MODEL", raising=False)
        settings = Settings.from_env()
        assert settings.gen_model == "google/gemma-4-E4B-it"

    def test_reads_temperature_with_default(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("TEMPERATURE", raising=False)
        settings = Settings.from_env()
        assert settings.temperature == pytest.approx(0.7)

    def test_reads_custom_temperature(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TEMPERATURE", "0.3")
        settings = Settings.from_env()
        assert settings.temperature == pytest.approx(0.3)

    def test_raises_valueerror_on_invalid_temperature(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TEMPERATURE", "not_a_float")
        with pytest.raises(ValueError, match="TEMPERATURE"):
            Settings.from_env()

    def test_reads_max_new_tokens_with_default(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("MAX_NEW_TOKENS", raising=False)
        settings = Settings.from_env()
        assert settings.max_new_tokens == 512

    def test_reads_custom_max_new_tokens(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("MAX_NEW_TOKENS", "256")
        settings = Settings.from_env()
        assert settings.max_new_tokens == 256

    def test_raises_valueerror_on_invalid_max_new_tokens(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("MAX_NEW_TOKENS", "invalid")
        with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
            Settings.from_env()

    def test_reads_top_p_with_default(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("TOP_P", raising=False)
        settings = Settings.from_env()
        assert settings.top_p == pytest.approx(0.9)

    def test_reads_custom_top_p(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TOP_P", "0.5")
        settings = Settings.from_env()
        assert settings.top_p == pytest.approx(0.5)

    def test_raises_valueerror_on_invalid_top_p(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TOP_P", "invalid")
        with pytest.raises(ValueError, match="TOP_P"):
            Settings.from_env()

    def test_reads_do_sample_true_by_default(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("DO_SAMPLE", raising=False)
        settings = Settings.from_env()
        assert settings.do_sample is True

    def test_reads_do_sample_false_when_set(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("DO_SAMPLE", "false")
        settings = Settings.from_env()
        assert settings.do_sample is False

    def test_reads_tokenizer_max_length_with_default(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("TOKENIZER_MAX_LENGTH", raising=False)
        settings = Settings.from_env()
        assert settings.tokenizer_max_length == 8192

    def test_reads_custom_tokenizer_max_length(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TOKENIZER_MAX_LENGTH", "4096")
        settings = Settings.from_env()
        assert settings.tokenizer_max_length == 4096

    def test_raises_valueerror_on_invalid_tokenizer_max_length(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TOKENIZER_MAX_LENGTH", "not_int")
        with pytest.raises(ValueError, match="TOKENIZER_MAX_LENGTH"):
            Settings.from_env()

    def test_reads_return_tensors_with_default(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("RETURN_TENSORS", raising=False)
        settings = Settings.from_env()
        assert settings.return_tensors == "pt"

    def test_reads_truncation_true_by_default(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("TRUNCATION", raising=False)
        settings = Settings.from_env()
        assert settings.truncation is True

    def test_reads_truncation_false_when_set(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TRUNCATION", "false")
        settings = Settings.from_env()
        assert settings.truncation is False

    def test_device_from_env_overrides_detection(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("DEVICE", "cuda")
        with patch("src.config._detect_device", return_value="cpu"):
            settings = Settings.from_env()
            assert settings.device == "cuda"

    def test_raises_valueerror_on_invalid_device(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("DEVICE", "gpu")
        with pytest.raises(ValueError, match="DEVICE"):
            Settings.from_env()

    def test_device_uses_detection_when_env_not_set(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("DEVICE", raising=False)
        with patch("src.config._detect_device", return_value="mps"):
            settings = Settings.from_env()
            assert settings.device == "mps"

    def test_settings_is_frozen(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        settings = Settings.from_env()
        with pytest.raises((AttributeError, TypeError)):
            settings.temperature = 0.5  # type: ignore[misc]

    def test_reads_log_level_with_default(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        settings = Settings.from_env()
        assert settings is not None

    def test_raises_valueerror_on_invalid_log_level(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("LOG_LEVEL", "INVALID_LEVEL")
        with pytest.raises(ValueError, match="LOG_LEVEL"):
            Settings.from_env()

    def test_all_settings_fields_present(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        settings = Settings.from_env()
        assert hasattr(settings, "qdrant_url")
        assert hasattr(settings, "qdrant_api_key")
        assert hasattr(settings, "embed_model")
        assert hasattr(settings, "rerank_model")
        assert hasattr(settings, "gen_model")
        assert hasattr(settings, "temperature")
        assert hasattr(settings, "max_new_tokens")
        assert hasattr(settings, "top_p")
        assert hasattr(settings, "do_sample")
        assert hasattr(settings, "tokenizer_max_length")
        assert hasattr(settings, "return_tensors")
        assert hasattr(settings, "truncation")
        assert hasattr(settings, "device")
        assert hasattr(settings, "lora_adapter_path")

    def test_lora_adapter_path_from_env_variable(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("LORA_ADAPTER_PATH", "models/pokesage-lora")
        settings = Settings.from_env()
        assert settings.lora_adapter_path == "models/pokesage-lora"

    def test_lora_adapter_path_defaults_to_none(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("LORA_ADAPTER_PATH", raising=False)
        settings = Settings.from_env()
        assert settings.lora_adapter_path is None

    def test_hyde_enabled_defaults_to_false(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("HYDE_ENABLED", raising=False)
        settings = Settings.from_env()
        assert settings.hyde_enabled is False

    def test_hyde_enabled_true_when_set(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("HYDE_ENABLED", "true")
        settings = Settings.from_env()
        assert settings.hyde_enabled is True

    def test_hyde_max_tokens_defaults_to_150(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("HYDE_MAX_TOKENS", raising=False)
        settings = Settings.from_env()
        assert settings.hyde_max_tokens == 150

    def test_hyde_max_tokens_reads_from_env(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("HYDE_MAX_TOKENS", "200")
        settings = Settings.from_env()
        assert settings.hyde_max_tokens == 200

    def test_all_settings_fields_include_hyde(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        settings = Settings.from_env()
        assert hasattr(settings, "hyde_enabled")
        assert hasattr(settings, "hyde_max_tokens")


@pytest.mark.unit
class TestConfigParameterBounds:
    def test_temperature_must_be_in_range_0_to_2(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TEMPERATURE", "-0.1")
        with pytest.raises(ValueError, match="TEMPERATURE"):
            Settings.from_env()

    def test_temperature_upper_bound_2_0(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TEMPERATURE", "2.1")
        with pytest.raises(ValueError, match="TEMPERATURE"):
            Settings.from_env()

    def test_temperature_valid_at_0(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TEMPERATURE", "0.0")
        settings = Settings.from_env()
        assert settings.temperature == 0.0

    def test_temperature_valid_at_2(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TEMPERATURE", "2.0")
        settings = Settings.from_env()
        assert settings.temperature == 2.0

    def test_top_p_must_be_in_range_0_to_1(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TOP_P", "-0.1")
        with pytest.raises(ValueError, match="TOP_P"):
            Settings.from_env()

    def test_top_p_upper_bound_1_0(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TOP_P", "1.1")
        with pytest.raises(ValueError, match="TOP_P"):
            Settings.from_env()

    def test_top_p_valid_at_0(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TOP_P", "0.0")
        settings = Settings.from_env()
        assert settings.top_p == 0.0

    def test_top_p_valid_at_1(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TOP_P", "1.0")
        settings = Settings.from_env()
        assert settings.top_p == 1.0

    def test_max_new_tokens_must_be_positive(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("MAX_NEW_TOKENS", "0")
        with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
            Settings.from_env()

    def test_max_new_tokens_must_be_greater_than_zero(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("MAX_NEW_TOKENS", "-1")
        with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
            Settings.from_env()

    def test_max_new_tokens_valid_at_1(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("MAX_NEW_TOKENS", "1")
        settings = Settings.from_env()
        assert settings.max_new_tokens == 1

    def test_hyde_max_tokens_zero_raises_valueerror(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("HYDE_MAX_TOKENS", "0")
        with pytest.raises(ValueError, match="HYDE_MAX_TOKENS"):
            Settings.from_env()

    def test_hyde_max_tokens_negative_raises_valueerror(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("HYDE_MAX_TOKENS", "-1")
        with pytest.raises(ValueError, match="HYDE_MAX_TOKENS"):
            Settings.from_env()

    def test_hyde_max_tokens_invalid_int_raises_valueerror(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("HYDE_MAX_TOKENS", "abc")
        with pytest.raises(ValueError, match="HYDE_MAX_TOKENS"):
            Settings.from_env()

    def test_hyde_num_drafts_zero_raises_valueerror(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("HYDE_NUM_DRAFTS", "0")
        with pytest.raises(ValueError, match="HYDE_NUM_DRAFTS"):
            Settings.from_env()

    def test_hyde_num_drafts_negative_raises_valueerror(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("HYDE_NUM_DRAFTS", "-5")
        with pytest.raises(ValueError, match="HYDE_NUM_DRAFTS"):
            Settings.from_env()

    def test_hyde_confidence_threshold_above_1_raises_valueerror(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("HYDE_CONFIDENCE_THRESHOLD", "1.5")
        with pytest.raises(ValueError, match="HYDE_CONFIDENCE_THRESHOLD"):
            Settings.from_env()

    def test_hyde_confidence_threshold_below_0_raises_valueerror(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("HYDE_CONFIDENCE_THRESHOLD", "-0.1")
        with pytest.raises(ValueError, match="HYDE_CONFIDENCE_THRESHOLD"):
            Settings.from_env()

    def test_hyde_confidence_threshold_invalid_float_raises_valueerror(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("HYDE_CONFIDENCE_THRESHOLD", "not_a_float")
        with pytest.raises(ValueError, match="HYDE_CONFIDENCE_THRESHOLD"):
            Settings.from_env()

    def test_tokenizer_max_length_zero_raises_valueerror(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TOKENIZER_MAX_LENGTH", "0")
        with pytest.raises(ValueError, match="TOKENIZER_MAX_LENGTH"):
            Settings.from_env()

    def test_tokenizer_max_length_negative_raises_valueerror(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("TOKENIZER_MAX_LENGTH", "-1")
        with pytest.raises(ValueError, match="TOKENIZER_MAX_LENGTH"):
            Settings.from_env()

    def test_query_timeout_seconds_defaults_to_120(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("QUERY_TIMEOUT_SECONDS", raising=False)
        settings = Settings.from_env()
        assert settings.query_timeout_seconds == 120.0

    def test_query_timeout_seconds_reads_custom_value(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QUERY_TIMEOUT_SECONDS", "30")
        settings = Settings.from_env()
        assert settings.query_timeout_seconds == 30.0

    def test_query_timeout_seconds_invalid_float_raises_valueerror(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QUERY_TIMEOUT_SECONDS", "abc")
        with pytest.raises(ValueError, match="QUERY_TIMEOUT_SECONDS"):
            Settings.from_env()

    def test_query_timeout_seconds_below_minimum_raises_valueerror(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QUERY_TIMEOUT_SECONDS", "0")
        with pytest.raises(ValueError, match="QUERY_TIMEOUT_SECONDS"):
            Settings.from_env()

    def test_query_timeout_seconds_above_maximum_raises_valueerror(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QUERY_TIMEOUT_SECONDS", "9999")
        with pytest.raises(ValueError, match="QUERY_TIMEOUT_SECONDS"):
            Settings.from_env()

    def test_query_timeout_seconds_valid_at_minimum_boundary(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QUERY_TIMEOUT_SECONDS", "1.0")
        settings = Settings.from_env()
        assert settings.query_timeout_seconds == 1.0

    def test_query_timeout_seconds_valid_at_maximum_boundary(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QUERY_TIMEOUT_SECONDS", "600.0")
        settings = Settings.from_env()
        assert settings.query_timeout_seconds == 600.0

    def test_do_sample_true_accepts_explicit_true(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("DO_SAMPLE", "true")
        settings = Settings.from_env()
        assert settings.do_sample is True

    def test_do_sample_true_accepts_1(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("DO_SAMPLE", "1")
        settings = Settings.from_env()
        assert settings.do_sample is True

    def test_do_sample_true_accepts_yes(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("DO_SAMPLE", "yes")
        settings = Settings.from_env()
        assert settings.do_sample is True

    def test_do_sample_false_accepts_0(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("DO_SAMPLE", "0")
        settings = Settings.from_env()
        assert settings.do_sample is False

    def test_do_sample_false_accepts_no(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("DO_SAMPLE", "no")
        settings = Settings.from_env()
        assert settings.do_sample is False

    def test_do_sample_invalid_value_raises_valueerror(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("DO_SAMPLE", "maybe")
        with pytest.raises(ValueError, match="DO_SAMPLE"):
            Settings.from_env()
