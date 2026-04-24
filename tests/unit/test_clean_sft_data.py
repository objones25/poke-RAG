from __future__ import annotations

import pytest

from scripts.training.clean_sft_data import _normalize_messages


@pytest.mark.unit
class TestNormalizeMessages:
    """Test _normalize_messages type annotation and behavior."""

    def test_normalize_system_user_assistant(self) -> None:
        """GREEN: system, user, assistant structure returns user and assistant only."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Pikachu?"},
            {"role": "assistant", "content": "Pikachu is an Electric-type."},
        ]
        result = _normalize_messages(messages)
        assert result is not None
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_normalize_user_assistant_passthrough(self) -> None:
        """GREEN: user, assistant structure passes through unchanged."""
        messages: list[dict[str, str]] = [
            {"role": "user", "content": "What is Pikachu?"},
            {"role": "assistant", "content": "Pikachu is an Electric-type."},
        ]
        result = _normalize_messages(messages)
        assert result == messages

    def test_normalize_unexpected_structure(self) -> None:
        """GREEN: Unexpected role structure returns None."""
        messages: list[dict[str, str]] = [
            {"role": "user", "content": "Question"},
        ]
        result = _normalize_messages(messages)
        assert result is None

    def test_normalize_empty_list(self) -> None:
        """GREEN: Empty messages returns None."""
        messages: list[dict[str, str]] = []
        result = _normalize_messages(messages)
        assert result is None


@pytest.mark.unit
class TestIsBadText:
    """Test _is_bad_text for detecting problematic text in questions/answers."""

    def test_clean_text_returns_false(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_text

        assert _is_bad_text("This is a normal Pokémon question about types.") is False

    def test_var_placeholder_returns_true(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_text

        assert _is_bad_text("This is a [VAR(name)] placeholder.") is True

    def test_var_placeholder_uppercase_returns_true(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_text

        assert _is_bad_text("Some text with [VAR(something)] here.") is True

    def test_db_id_pattern_returns_true(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_text

        assert _is_bad_text("Reference Item0042 in database.") is True

    def test_db_id_pattern_long_suffix_returns_true(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_text

        assert _is_bad_text("See Dra6688 for details.") is True

    def test_dynamax_crystal_lowercase_returns_true(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_text

        assert _is_bad_text("The dynamax crystal is important.") is True

    def test_dynamax_crystal_mixedcase_returns_true(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_text

        assert _is_bad_text("Use the Dynamax Crystal for raids.") is True

    def test_normal_alphanumeric_no_db_pattern_returns_false(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_text

        assert _is_bad_text("Pikachu has 95 HP") is False


@pytest.mark.unit
class TestIsBadAnswer:
    """Test _is_bad_answer for detecting low-quality answers."""

    def test_normal_detailed_answer_returns_false(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_answer

        answer = (
            "Pikachu is an Electric-type Pokémon known for its powerful "
            "electric attacks and iconic status in the franchise."
        )
        assert _is_bad_answer(answer) is False

    def test_answer_too_short_returns_true(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_answer

        assert _is_bad_answer("Pikachu") is True

    def test_bare_pokemon_definition_returns_true(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_answer

        assert _is_bad_answer("Pikachu is a Pokémon.") is True

    def test_bare_pokemon_definition_with_period_returns_true(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_answer

        assert _is_bad_answer("Charizard is a Pokémon") is True

    def test_bare_pokemon_an_variation_returns_true(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_answer

        assert _is_bad_answer("Eevee is an Pokémon.") is True

    def test_answer_with_insufficient_information_pattern_returns_true(
        self,
    ) -> None:
        from scripts.training.clean_sft_data import _is_bad_answer

        assert (
            _is_bad_answer(
                "Unfortunately there is insufficient information available "
                "about the Pokémon type system."
            )
            is True
        )

    def test_answer_with_not_enough_information_pattern_returns_true(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_answer

        assert _is_bad_answer("Unfortunately there is not enough information available.") is True

    def test_answer_with_cannot_be_answered_pattern_returns_true(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_answer

        assert _is_bad_answer("This question cannot be answered from the provided text.") is True

    def test_answer_with_no_further_detail_pattern_returns_true(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_answer

        assert (
            _is_bad_answer("There is no further detail available in the context about this topic.")
            is True
        )

    def test_answer_with_does_not_contain_pattern_returns_true(self) -> None:
        from scripts.training.clean_sft_data import _is_bad_answer

        assert _is_bad_answer("The provided text does not contain information about that.") is True


@pytest.mark.unit
class TestClean:
    """Test clean function for filtering and normalizing SFT data."""

    def test_clean_empty_input_returns_zero_kept(self, tmp_path) -> None:
        from scripts.training.clean_sft_data import clean

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        input_file.write_text("")

        total, kept, removed = clean(input_file, output_file)
        assert total == 0
        assert kept == 0
        assert removed == 0

    def test_clean_all_valid_entries(self, tmp_path) -> None:
        from scripts.training.clean_sft_data import clean

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        input_file.write_text(
            '{"messages": [{"role": "user", "content": "What is Pikachu?"}, '
            '{"role": "assistant", "content": "Pikachu is an Electric-type '
            'Pokémon with powerful attacks and excellent speed."}]}\n'
        )

        total, kept, removed = clean(input_file, output_file)
        assert total == 1
        assert kept == 1
        assert removed == 0
        assert output_file.exists()

    def test_clean_filters_bad_text_user_content(self, tmp_path) -> None:
        from scripts.training.clean_sft_data import clean

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        input_file.write_text(
            '{"messages": [{"role": "user", "content": "Tell me about [VAR(pokemon)]"}, '
            '{"role": "assistant", "content": "Pikachu is an Electric-type '
            'Pokémon with powerful attacks."}]}\n'
        )

        total, kept, removed = clean(input_file, output_file)
        assert total == 1
        assert kept == 0
        assert removed == 1

    def test_clean_filters_bad_text_assistant_content(self, tmp_path) -> None:
        from scripts.training.clean_sft_data import clean

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        input_file.write_text(
            '{"messages": [{"role": "user", "content": "What is Pikachu?"}, '
            '{"role": "assistant", "content": "Pikachu is a Pokémon with '
            'Item0042 special attack."}]}\n'
        )

        total, kept, removed = clean(input_file, output_file)
        assert total == 1
        assert kept == 0
        assert removed == 1

    def test_clean_filters_bad_answers(self, tmp_path) -> None:
        from scripts.training.clean_sft_data import clean

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        input_file.write_text(
            '{"messages": [{"role": "user", "content": "What is Pikachu?"}, '
            '{"role": "assistant", "content": "The provided text does not '
            'contain enough information about this."}]}\n'
        )

        total, kept, removed = clean(input_file, output_file)
        assert total == 1
        assert kept == 0
        assert removed == 1

    def test_clean_filters_invalid_json(self, tmp_path) -> None:
        from scripts.training.clean_sft_data import clean

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        input_file.write_text("not valid json\n")

        total, kept, removed = clean(input_file, output_file)
        assert total == 1
        assert kept == 0
        assert removed == 1

    def test_clean_filters_wrong_message_structure(self, tmp_path) -> None:
        from scripts.training.clean_sft_data import clean

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        input_file.write_text('{"messages": [{"role": "user", "content": "Hello"}]}\n')

        total, kept, removed = clean(input_file, output_file)
        assert total == 1
        assert kept == 0
        assert removed == 1

    def test_clean_strips_system_message(self, tmp_path) -> None:
        from scripts.training.clean_sft_data import clean

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        input_file.write_text(
            '{"messages": [{"role": "system", "content": "You are helpful."}, '
            '{"role": "user", "content": "What is Pikachu?"}, '
            '{"role": "assistant", "content": "Pikachu is an Electric-type '
            'Pokémon found in Viridian Forest."}]}\n'
        )

        total, kept, removed = clean(input_file, output_file)
        assert total == 1
        assert kept == 1
        assert removed == 0

        output_content = output_file.read_text()
        assert "system" not in output_content

    def test_clean_empty_lines_are_skipped(self, tmp_path) -> None:
        from scripts.training.clean_sft_data import clean

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        input_file.write_text(
            '{"messages": [{"role": "user", "content": "What is Pikachu?"}, '
            '{"role": "assistant", "content": "Pikachu is an Electric-type '
            'Pokémon with powerful attacks."}]}\n'
            "\n"
            "\n"
        )

        total, kept, removed = clean(input_file, output_file)
        assert total == 1
        assert kept == 1

    def test_clean_mix_of_valid_and_invalid(self, tmp_path) -> None:
        from scripts.training.clean_sft_data import clean

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        input_file.write_text(
            '{"messages": [{"role": "user", "content": "What is Pikachu?"}, '
            '{"role": "assistant", "content": "Pikachu is an Electric-type '
            'Pokémon with excellent speed."}]}\n'
            '{"messages": [{"role": "user", "content": "Tell [VAR(me)]"}, '
            '{"role": "assistant", "content": "This cannot be answered from '
            'the text."}]}\n'
            '{"messages": [{"role": "user", "content": "What is Charizard?"}, '
            '{"role": "assistant", "content": "Charizard is a Fire-type '
            "Pokémon that evolves from Charmeleon with strong fire-based "
            'attacks."}]}\n'
        )

        total, kept, removed = clean(input_file, output_file)
        assert total == 3
        assert kept == 2
        assert removed == 1
