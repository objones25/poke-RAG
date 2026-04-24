from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from google.genai.types import GenerateContentConfig, HttpOptions

from scripts.training.gemini_client import GeminiClient

_QUALITY_ANSWER = "Pikachu is an Electric-type Pokémon with high speed and moderate attack."


@pytest.mark.unit
class TestGeminiClientGenerateQAPair:
    """Test GeminiClient.generate_qa_pair() error handling and response parsing."""

    def test_successful_generation(self) -> None:
        """GREEN: Successful generation returns a valid pair."""
        client = GeminiClient(api_key="fake-key")
        response_text = '{"question": "What is Pikachu?", "answer": "' + _QUALITY_ANSWER + '"}'

        with patch.object(client._client.models, "generate_content") as mock_gen:
            mock_response = MagicMock()
            mock_response.text = response_text
            mock_gen.return_value = mock_response

            result = client.generate_qa_pair("Pikachu chunk", "pokeapi")
            assert result is not None
            assert result.question == "What is Pikachu?"
            assert result.answer == _QUALITY_ANSWER

    def test_response_text_is_none_raises_error(self) -> None:
        """RED: response.text is None should raise RuntimeError."""
        client = GeminiClient(api_key="fake-key")

        with patch.object(client._client.models, "generate_content") as mock_gen:
            mock_response = MagicMock()
            mock_response.text = None
            mock_gen.return_value = mock_response

            with pytest.raises(RuntimeError, match="empty response"):
                client.generate_qa_pair("Some chunk", "pokeapi")

    def test_response_text_is_empty_string_raises_error(self) -> None:
        """RED: response.text is empty string should raise RuntimeError."""
        client = GeminiClient(api_key="fake-key")

        with patch.object(client._client.models, "generate_content") as mock_gen:
            mock_response = MagicMock()
            mock_response.text = ""
            mock_gen.return_value = mock_response

            with pytest.raises(RuntimeError, match="empty response"):
                client.generate_qa_pair("Some chunk", "pokeapi")

    def test_validation_error_with_empty_response_text(self) -> None:
        """RED: ValidationError on None/empty response.text during fallback."""
        client = GeminiClient(api_key="fake-key")

        with patch.object(client._client.models, "generate_content") as mock_gen:
            mock_response = MagicMock()
            mock_response.text = None  # None will cause json.loads to fail
            mock_gen.return_value = mock_response

            with pytest.raises(RuntimeError, match="exhausted|empty response"):
                client.generate_qa_pair("Some chunk", "pokeapi", max_retries=1)

    def test_fallback_json_parsing_specific_exception(self) -> None:
        """RED: Fallback JSON parsing should catch specific exceptions, not bare Exception."""
        client = GeminiClient(api_key="fake-key")

        # First attempt: ValidationError triggers fallback
        # Fallback: json.loads fails with JSONDecodeError (not logged as generic Exception)
        with patch.object(client._client.models, "generate_content") as mock_gen:
            mock_response = MagicMock()
            # Invalid JSON that will fail pydantic validation, then json.loads
            mock_response.text = "not valid json at all {{"
            mock_gen.return_value = mock_response

            with pytest.raises(RuntimeError, match="Failed to generate"):
                client.generate_qa_pair("Some chunk", "pokeapi", max_retries=1)

    def test_fallback_json_parsing_keyerror(self) -> None:
        """RED: Fallback parsing should handle KeyError (missing expected fields) gracefully."""
        client = GeminiClient(api_key="fake-key")

        with patch.object(client._client.models, "generate_content") as mock_gen:
            mock_response = MagicMock()
            # Valid JSON but missing required fields for GeminiQAPair
            mock_response.text = '{"wrong_field": "value"}'
            mock_gen.return_value = mock_response

            with pytest.raises(RuntimeError, match="Failed to generate"):
                client.generate_qa_pair("Some chunk", "pokeapi", max_retries=1)

    def test_quality_gate_returns_none(self) -> None:
        """GREEN: Valid pair that fails quality gate returns None."""
        client = GeminiClient(api_key="fake-key")
        # Bare "X is a Pokémon." answer fails quality gate
        response_text = '{"question": "What is Pikachu?", "answer": "Pikachu is a Pokémon."}'

        with patch.object(client._client.models, "generate_content") as mock_gen:
            mock_response = MagicMock()
            mock_response.text = response_text
            mock_gen.return_value = mock_response

            result = client.generate_qa_pair("Pikachu chunk", "pokeapi")
            assert result is None

    def test_rate_limit_retry_logic(self) -> None:
        """GREEN: 429 rate limit triggers retry logic."""
        client = GeminiClient(api_key="fake-key")
        response_text = '{"question": "What is Pikachu?", "answer": "' + _QUALITY_ANSWER + '"}'

        with (
            patch.object(client._client.models, "generate_content") as mock_gen,
            patch("time.sleep"),
        ):
            call_count = [0]

            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise Exception("429 RESOURCE_EXHAUSTED Rate limit exceeded")
                mock_response = MagicMock()
                mock_response.text = response_text
                return mock_response

            mock_gen.side_effect = side_effect

            result = client.generate_qa_pair("Some chunk", "pokeapi", max_retries=3)
            assert result is not None
            assert call_count[0] == 2  # Failed once, succeeded on retry

    def test_config_is_proper_type(self) -> None:
        """GREEN: Config passed to generate_content is GenerateContentConfig, not dict."""
        client = GeminiClient(api_key="fake-key")
        response_text = '{"question": "What is Pikachu?", "answer": "' + _QUALITY_ANSWER + '"}'

        with patch.object(client._client.models, "generate_content") as mock_gen:
            mock_response = MagicMock()
            mock_response.text = response_text
            mock_gen.return_value = mock_response

            client.generate_qa_pair("Pikachu chunk", "pokeapi")

            # Verify generate_content was called
            assert mock_gen.called
            # Get the call arguments and verify config is the correct type
            call_args = mock_gen.call_args
            assert call_args is not None
            config = call_args.kwargs.get("config")
            assert config is not None
            assert isinstance(config, GenerateContentConfig)

    def test_config_has_timeout(self) -> None:
        """RED: Config passed to generate_content should have httpOptions with timeout set."""
        client = GeminiClient(api_key="fake-key")
        response_text = '{"question": "What is Pikachu?", "answer": "' + _QUALITY_ANSWER + '"}'

        with patch.object(client._client.models, "generate_content") as mock_gen:
            mock_response = MagicMock()
            mock_response.text = response_text
            mock_gen.return_value = mock_response

            client.generate_qa_pair("Pikachu chunk", "pokeapi")

            # Verify generate_content was called
            assert mock_gen.called
            # Get the call arguments and verify config has timeout
            call_args = mock_gen.call_args
            assert call_args is not None
            config = call_args.kwargs.get("config")
            assert config is not None
            assert isinstance(config, GenerateContentConfig)
            # Verify httpOptions has a timeout
            assert config.http_options is not None
            assert isinstance(config.http_options, HttpOptions)
            assert config.http_options.timeout is not None
            assert config.http_options.timeout > 0
