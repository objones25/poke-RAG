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
        assert router.route("Best OU Pokemon this generation?") == ["smogon"]

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
        assert router.route("What coverage moves does Dragonite run?") == ["smogon"]

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
        assert router.route("What IV does Pikachu need for max speed?") == ["smogon"]

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
        assert router.route("Read me the dex entry for Haunter") == ["bulbapedia"]

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
