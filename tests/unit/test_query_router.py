"""Unit tests for src/retrieval/query_router.py — RED until implemented."""

from __future__ import annotations

import pytest

from src.retrieval.query_router import QueryRouter


@pytest.mark.unit
class TestQueryRouterPokeapi:
    def test_routes_base_stats_query(self) -> None:
        router = QueryRouter()
        assert router.route("What are Charizard's base stats?") == ["pokeapi"]

    def test_routes_hp_query(self) -> None:
        router = QueryRouter()
        assert router.route("How much HP does Blissey have?") == ["pokeapi"]

    def test_routes_attack_stat_query(self) -> None:
        router = QueryRouter()
        assert router.route("What is Machamp's attack stat?") == ["pokeapi"]

    def test_routes_defense_query(self) -> None:
        router = QueryRouter()
        assert router.route("How high is Shuckle's defense?") == ["pokeapi"]

    def test_routes_speed_query(self) -> None:
        router = QueryRouter()
        assert router.route("What is Jolteon's speed stat?") == ["pokeapi"]

    def test_routes_special_attack_query(self) -> None:
        router = QueryRouter()
        assert router.route("What is Alakazam's special attack?") == ["pokeapi"]

    def test_routes_sp_atk_abbreviation(self) -> None:
        router = QueryRouter()
        assert router.route("How high is Gengar's sp. atk?") == ["pokeapi"]

    def test_routes_special_defense_query(self) -> None:
        router = QueryRouter()
        assert router.route("Does Blissey have high special defense?") == ["pokeapi"]

    def test_routes_special_defence_british_spelling(self) -> None:
        router = QueryRouter()
        assert router.route("What is Chansey's special defence?") == ["pokeapi"]

    def test_routes_sp_def_abbreviation(self) -> None:
        router = QueryRouter()
        assert router.route("Starmie sp. def stat?") == ["pokeapi"]

    def test_routes_bst_abbreviation(self) -> None:
        router = QueryRouter()
        assert router.route("Which Pokemon has the highest BST?") == ["pokeapi"]

    def test_routes_type_query(self) -> None:
        router = QueryRouter()
        assert router.route("What type is Garchomp?") == ["pokeapi"]

    def test_routes_typing_query(self) -> None:
        router = QueryRouter()
        assert router.route("What is Ferrothorn's typing?") == ["pokeapi"]

    def test_routes_weakness_query(self) -> None:
        router = QueryRouter()
        assert router.route("What are Gyarados weaknesses?") == ["pokeapi"]

    def test_routes_resist_query(self) -> None:
        router = QueryRouter()
        assert router.route("What does Ferrothorn resist?") == ["pokeapi"]

    def test_routes_resists_plural(self) -> None:
        router = QueryRouter()
        assert router.route("What types does Skarmory resists?") == ["pokeapi"]

    def test_routes_resistance_query(self) -> None:
        router = QueryRouter()
        assert router.route("What resistances does Heatran have?") == ["pokeapi"]

    def test_routes_immunity_query(self) -> None:
        router = QueryRouter()
        assert router.route("Does Gengar have an immunity to Normal?") == ["pokeapi"]

    def test_routes_move_query(self) -> None:
        router = QueryRouter()
        assert router.route("What moves can Gengar learn?") == ["pokeapi"]

    def test_routes_learnset_query(self) -> None:
        router = QueryRouter()
        assert router.route("Show me Charizard's learnset") == ["pokeapi"]

    def test_routes_ability_query(self) -> None:
        router = QueryRouter()
        assert router.route("What is Intimidate ability?") == ["pokeapi"]

    def test_routes_hidden_ability_query(self) -> None:
        router = QueryRouter()
        assert router.route("What is Garchomp's hidden ability?") == ["pokeapi"]

    def test_routes_evolution_query(self) -> None:
        router = QueryRouter()
        assert router.route("How does Eevee evolve?") == ["pokeapi"]

    def test_routes_evolves_variant(self) -> None:
        router = QueryRouter()
        assert router.route("At what level does Magikarp evolve?") == ["pokeapi"]

    def test_routes_evolution_chain_query(self) -> None:
        router = QueryRouter()
        assert router.route("What is Ralts' evolution chain?") == ["pokeapi"]

    def test_routes_mega_form_query(self) -> None:
        router = QueryRouter()
        assert router.route("Does Charizard have a mega form?") == ["pokeapi"]

    def test_routes_regional_form_query(self) -> None:
        router = QueryRouter()
        assert router.route("What is the Galarian form of Ponyta?") == ["pokeapi"]

    def test_routes_egg_group_query(self) -> None:
        router = QueryRouter()
        assert router.route("What egg group is Ditto in?") == ["pokeapi"]

    def test_routes_height_weight_query(self) -> None:
        router = QueryRouter()
        assert router.route("What is Wailord's height and weight?") == ["pokeapi"]


@pytest.mark.unit
class TestQueryRouterSmogon:
    def test_routes_tier_query(self) -> None:
        router = QueryRouter()
        assert router.route("What tier is Landorus-Therian?") == ["smogon"]

    def test_routes_ou_tier(self) -> None:
        router = QueryRouter()
        # "generation" also triggers bulbapedia (history/gen patterns)
        assert router.route("Best OU Pokemon this generation?") == ["bulbapedia", "smogon"]

    def test_routes_uu_tier(self) -> None:
        router = QueryRouter()
        assert router.route("Is Scizor still UU?") == ["smogon"]

    def test_routes_ru_tier(self) -> None:
        router = QueryRouter()
        assert router.route("What's good in RU right now?") == ["smogon"]

    def test_routes_nu_tier(self) -> None:
        router = QueryRouter()
        assert router.route("Top threats in NU?") == ["smogon"]

    def test_routes_ubers_tier(self) -> None:
        router = QueryRouter()
        assert router.route("Is Zacian banned to Ubers?") == ["smogon"]

    def test_routes_competitive_query(self) -> None:
        router = QueryRouter()
        assert router.route("Is Garchomp good competitively?") == ["smogon"]

    def test_routes_counter_query(self) -> None:
        router = QueryRouter()
        assert router.route("What counters Toxapex?") == ["smogon"]

    def test_routes_checks_query(self) -> None:
        router = QueryRouter()
        assert router.route("What checks Volcarona?") == ["smogon"]

    def test_routes_coverage_query(self) -> None:
        router = QueryRouter()
        # "moves" also triggers pokeapi (factual move data); "run" triggers smogon
        assert router.route("What coverage moves does Dragonite run?") == ["pokeapi", "smogon"]

    def test_routes_teammates_query(self) -> None:
        router = QueryRouter()
        assert router.route("What are good teammates for Garchomp?") == ["smogon"]

    def test_routes_ev_spread_query(self) -> None:
        router = QueryRouter()
        assert router.route("What is the best EV spread for Garchomp?") == ["smogon"]

    def test_routes_ev_whole_word_not_substring(self) -> None:
        router = QueryRouter()
        # "level" and "evolve" contain 'ev' but must not trigger smogon
        result = router.route("At what level does Magikarp evolve?")
        assert "smogon" not in result

    def test_routes_iv_whole_word(self) -> None:
        router = QueryRouter()
        # "speed" also triggers pokeapi (stat pattern); "IV"/"max speed" trigger smogon
        assert router.route("What IV does Pikachu need for max speed?") == ["pokeapi", "smogon"]

    def test_routes_iv_whole_word_not_substring(self) -> None:
        router = QueryRouter()
        # "revival" contains 'iv' but must not trigger smogon
        result = router.route("Does revival herb restore a fainted Pokemon?")
        assert "smogon" not in result

    def test_routes_strategy_query(self) -> None:
        router = QueryRouter()
        assert router.route("What strategy works best for Clefable?") == ["smogon"]

    def test_routes_meta_query(self) -> None:
        router = QueryRouter()
        assert router.route("How has the meta shifted this season?") == ["smogon"]

    def test_routes_smogon_keyword_itself(self) -> None:
        router = QueryRouter()
        assert router.route("What does Smogon say about Garchomp?") == ["smogon"]

    def test_routes_vgc_query(self) -> None:
        router = QueryRouter()
        assert router.route("Which Pokemon are good in VGC doubles?") == ["smogon"]

    def test_routes_doubles_query(self) -> None:
        router = QueryRouter()
        assert router.route("Is Incineroar good in doubles?") == ["smogon"]

    def test_routes_moveset_query(self) -> None:
        router = QueryRouter()
        assert router.route("What moveset should I run on Clefable?") == ["smogon"]

    def test_routes_viability_query(self) -> None:
        router = QueryRouter()
        assert router.route("How is Rotom-Wash's viability in OU?") == ["smogon"]

    def test_routes_core_synergy_query(self) -> None:
        router = QueryRouter()
        assert router.route("What core has good synergy with Toxapex?") == ["smogon"]


@pytest.mark.unit
class TestQueryRouterBulbapedia:
    def test_routes_lore_query(self) -> None:
        router = QueryRouter()
        assert router.route("What is the lore behind Mewtwo?") == ["bulbapedia"]

    def test_routes_flavor_text_query(self) -> None:
        router = QueryRouter()
        assert router.route("What is Gengar's flavor text?") == ["bulbapedia"]

    def test_routes_flavour_british_spelling(self) -> None:
        router = QueryRouter()
        assert router.route("What is Gengar's flavour text?") == ["bulbapedia"]

    def test_routes_pokedex_entry_query(self) -> None:
        router = QueryRouter()
        assert router.route("What does the pokedex entry say about Cubone?") == ["bulbapedia"]

    def test_routes_dex_entry_query(self) -> None:
        router = QueryRouter()
        # "dex entry" matches both bulbapedia (flavor text) and pokeapi (factual data)
        assert router.route("Read me the dex entry for Haunter") == ["bulbapedia", "pokeapi"]

    def test_routes_origin_query(self) -> None:
        router = QueryRouter()
        assert router.route("What is the origin of Charizard's design?") == ["bulbapedia"]

    def test_routes_design_inspiration_query(self) -> None:
        router = QueryRouter()
        assert router.route("What animal is Arcanine's design based on?") == ["bulbapedia"]

    def test_routes_anime_query(self) -> None:
        router = QueryRouter()
        assert router.route("Does Pikachu appear in the anime?") == ["bulbapedia"]

    def test_routes_manga_query(self) -> None:
        router = QueryRouter()
        assert router.route("Which manga features Mewtwo prominently?") == ["bulbapedia"]

    def test_routes_history_query(self) -> None:
        router = QueryRouter()
        assert router.route("What generation was Gengar introduced?") == ["bulbapedia"]

    def test_routes_mythology_query(self) -> None:
        router = QueryRouter()
        assert router.route("What mythology is Xerneas based on?") == ["bulbapedia"]

    def test_routes_backstory_query(self) -> None:
        router = QueryRouter()
        assert router.route("What is the backstory of the Regi trio?") == ["bulbapedia"]

    def test_routes_debut_query(self) -> None:
        router = QueryRouter()
        assert router.route("When did Togepi debut in the anime?") == ["bulbapedia"]


@pytest.mark.unit
class TestQueryRouterMultiSource:
    def test_routes_competitive_stats_to_pokeapi_and_smogon(self) -> None:
        router = QueryRouter()
        result = router.route("What EV spread suits Garchomp's base stats?")
        assert set(result) == {"pokeapi", "smogon"}

    def test_routes_lore_and_stats_to_bulbapedia_and_pokeapi(self) -> None:
        router = QueryRouter()
        result = router.route("What are Mewtwo's base stats and origin?")
        assert set(result) == {"pokeapi", "bulbapedia"}

    def test_routes_all_keywords_to_all_sources(self) -> None:
        router = QueryRouter()
        result = router.route("stats tier lore")
        assert set(result) == {"pokeapi", "smogon", "bulbapedia"}

    def test_multi_source_result_is_sorted(self) -> None:
        router = QueryRouter()
        result = router.route("competitive stats for Garchomp")
        assert result == sorted(result)


@pytest.mark.unit
class TestQueryRouterFallback:
    def test_empty_query_returns_all_sources(self) -> None:
        router = QueryRouter()
        assert set(router.route("")) == {"bulbapedia", "pokeapi", "smogon"}

    def test_whitespace_only_returns_all_sources(self) -> None:
        router = QueryRouter()
        assert set(router.route("   ")) == {"bulbapedia", "pokeapi", "smogon"}

    def test_pokemon_name_alone_returns_all_sources(self) -> None:
        router = QueryRouter()
        assert set(router.route("Pikachu")) == {"bulbapedia", "pokeapi", "smogon"}

    def test_generic_question_returns_all_sources(self) -> None:
        router = QueryRouter()
        assert set(router.route("Tell me about Snorlax")) == {"bulbapedia", "pokeapi", "smogon"}

    def test_fallback_returns_canonical_source_order(self) -> None:
        router = QueryRouter()
        result = router.route("Pikachu")
        assert result == ["bulbapedia", "pokeapi", "smogon"]


@pytest.mark.unit
class TestQueryRouterCaseInsensitivity:
    def test_uppercase_stat_keyword_matches_pokeapi(self) -> None:
        router = QueryRouter()
        assert router.route("STATS for Pikachu") == ["pokeapi"]

    def test_uppercase_tier_keyword_matches_smogon(self) -> None:
        router = QueryRouter()
        assert router.route("What TIER is Mewtwo?") == ["smogon"]

    def test_uppercase_lore_keyword_matches_bulbapedia(self) -> None:
        router = QueryRouter()
        assert router.route("LORE of Arceus") == ["bulbapedia"]

    def test_mixed_case_matches(self) -> None:
        router = QueryRouter()
        assert router.route("Base Stats and Weakness") == ["pokeapi"]

    def test_uppercase_ev_matches_smogon(self) -> None:
        router = QueryRouter()
        assert router.route("Best EV spread for Tyranitar?") == ["smogon"]


@pytest.mark.unit
class TestQueryRouterRegexEdgeCases:
    def test_pokemon_name_with_regex_metacharacters(self) -> None:
        """Query containing Pokemon name with regex special chars should not crash."""
        router = QueryRouter()
        # Nidoran has special chars, but query mentions stats (pokeapi match)
        result = router.route("Does Nidoran♂ have high stats?")
        assert result == ["pokeapi"]

    def test_query_with_tab_character(self) -> None:
        """Tab character in query should not prevent regex matching."""
        router = QueryRouter()
        result = router.route("what are the stats\t")
        assert result == ["pokeapi"]

    def test_query_with_newline_character(self) -> None:
        """Newline character in query should not prevent regex matching."""
        router = QueryRouter()
        result = router.route("what are the stats\n")
        assert result == ["pokeapi"]

    def test_query_with_mixed_whitespace(self) -> None:
        """Mixed whitespace (tabs, newlines, spaces) should be handled."""
        router = QueryRouter()
        result = router.route("  what\t\nare the\rstats  ")
        assert result == ["pokeapi"]

    def test_query_only_special_characters(self) -> None:
        """Query with only special characters should fall back to all sources."""
        router = QueryRouter()
        result = router.route("!!!")
        assert set(result) == {"bulbapedia", "pokeapi", "smogon"}

    def test_query_with_unicode_special_chars(self) -> None:
        """Unicode special characters should not crash regex."""
        router = QueryRouter()
        result = router.route("What about Pokémon stats? (2024)")
        assert result == ["pokeapi"]

    def test_very_long_query_with_keyword(self) -> None:
        """Very long query (2000+ chars) containing keyword should match efficiently."""
        long_query = "x" * 1000 + " stats " + "y" * 1000
        router = QueryRouter()
        result = router.route(long_query)
        assert result == ["pokeapi"]

    def test_very_long_query_no_keyword(self) -> None:
        """Very long query without keywords should fall back to all sources."""
        long_query = "x" * 2500
        router = QueryRouter()
        result = router.route(long_query)
        assert set(result) == {"bulbapedia", "pokeapi", "smogon"}

    def test_query_with_multiple_newlines_and_keyword(self) -> None:
        """Multiple newlines should not prevent keyword matching."""
        router = QueryRouter()
        result = router.route("what\n\n\nare\n\nstats\n\n")
        assert result == ["pokeapi"]


@pytest.mark.unit
class TestQueryRouterReturnValueConsistency:
    def test_empty_query_vs_no_match_same_sources(self) -> None:
        """Both empty query and no-match query return same sources."""
        router = QueryRouter()
        empty_result = set(router.route(""))
        no_match_result = set(router.route("xyz_no_match_abc"))
        assert empty_result == no_match_result == {"bulbapedia", "pokeapi", "smogon"}

    def test_fallback_always_sorted(self) -> None:
        """Fallback (all sources) should always be in sorted order."""
        router = QueryRouter()
        result = router.route("no match here")
        assert result == ["bulbapedia", "pokeapi", "smogon"]
        assert result == sorted(result)

    def test_pokeapi_single_source_no_duplicates(self) -> None:
        """Single source match should not have duplicates."""
        router = QueryRouter()
        result = router.route("What are the stats?")
        assert result == ["pokeapi"]
        assert len(result) == len(set(result))

    def test_multi_source_result_no_duplicates(self) -> None:
        """Multi-source result should have no duplicates."""
        router = QueryRouter()
        result = router.route("stats and tier")
        assert len(result) == len(set(result))

    def test_multi_source_always_sorted(self) -> None:
        """Multi-source result should always be sorted."""
        router = QueryRouter()
        # These keywords trigger different order without sorting
        result = router.route("smogon tier and pokeapi stats")
        assert result == sorted(result)

    def test_return_type_always_list(self) -> None:
        """Return value should always be a list."""
        router = QueryRouter()
        result = router.route("stats")
        assert isinstance(result, list)
        result = router.route("")
        assert isinstance(result, list)

    def test_return_elements_always_source_type(self) -> None:
        """All elements in return list should be valid Source strings."""
        router = QueryRouter()
        valid_sources = {"bulbapedia", "pokeapi", "smogon"}
        for query in ["stats", "tier", "lore", "no match", ""]:
            result = router.route(query)
            for source in result:
                assert source in valid_sources


@pytest.mark.unit
class TestQueryRouterWholeWordBoundaries:
    def test_revival_does_not_match_iv(self) -> None:
        """'revival' contains 'iv' but should not trigger smogon."""
        router = QueryRouter()
        result = router.route("Does revival herb restore a fainted Pokemon?")
        assert "smogon" not in result
        # Should route to pokeapi (item, faint keywords)
        assert "pokeapi" in result

    def test_level_does_not_match_ev(self) -> None:
        """'level' contains 'ev' but should not trigger smogon."""
        router = QueryRouter()
        result = router.route("At what level does Magikarp evolve?")
        assert "smogon" not in result
        # Should route to pokeapi (level, evolve keywords)
        assert "pokeapi" in result

    def test_standing_iv_uppercase_matches(self) -> None:
        """Standalone uppercase 'IV' should match smogon."""
        router = QueryRouter()
        result = router.route("What IV does Pikachu need?")
        assert "smogon" in result

    def test_standing_ev_uppercase_matches(self) -> None:
        """Standalone uppercase 'EV' should match smogon."""
        router = QueryRouter()
        result = router.route("Best EV spread?")
        assert "smogon" in result

    def test_iv_surrounded_by_punctuation_matches(self) -> None:
        """'IV' surrounded by punctuation should match."""
        router = QueryRouter()
        result = router.route("Perfect IV's for competitive play")
        assert "smogon" in result

    def test_ev_at_word_boundary_with_numbers(self) -> None:
        """'EV' followed by numbers should still match (word boundary)."""
        router = QueryRouter()
        result = router.route("252 EV in Special Attack")
        assert "smogon" in result


@pytest.mark.unit
class TestQueryRouterIdempotency:
    def test_same_query_twice_returns_equal_results(self) -> None:
        """Calling route() twice with same query should return equal lists."""
        router = QueryRouter()
        result1 = router.route("What are Charizard's stats?")
        result2 = router.route("What are Charizard's stats?")
        assert result1 == result2

    def test_multiple_instances_give_same_routing(self) -> None:
        """Two separate QueryRouter instances should give same results."""
        router1 = QueryRouter()
        router2 = QueryRouter()
        query = "competitive stats and tier"
        result1 = router1.route(query)
        result2 = router2.route(query)
        assert result1 == result2

    def test_same_query_across_100_calls(self) -> None:
        """Routing should be consistent across many calls."""
        router = QueryRouter()
        results = [router.route("stats and lore") for _ in range(100)]
        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_different_sources_consistent(self) -> None:
        """Each source keyword should consistently route to same source."""
        router = QueryRouter()
        for _ in range(10):
            assert router.route("stats") == ["pokeapi"]
            assert router.route("tier") == ["smogon"]
            assert router.route("lore") == ["bulbapedia"]
