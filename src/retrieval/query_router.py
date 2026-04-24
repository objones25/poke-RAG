"""Keyword-based query router: classify a query into one or more retrieval sources."""

from __future__ import annotations

import re

from src.types import Source


def _w(kw: str) -> re.Pattern[str]:
    """Whole-word match (word boundaries on both sides)."""
    return re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)


def _p(phrase: str) -> re.Pattern[str]:
    """Prefix phrase match (word boundary at start only, phrase must appear verbatim)."""
    return re.compile(r"\b" + re.escape(phrase), re.IGNORECASE)


def _prefix(stem: str) -> re.Pattern[str]:
    """Stem + optional word chars (matches 'competitive', 'competitively', etc.)."""
    return re.compile(r"\b" + re.escape(stem) + r"\w*", re.IGNORECASE)


_POKEAPI_PATTERNS: tuple[re.Pattern[str], ...] = (
    # stats
    _w("stat"),
    _w("stats"),
    _w("base"),
    _w("bst"),
    _w("hp"),
    _w("attack"),
    _w("defense"),
    _w("defence"),
    _p("sp. atk"),
    _p("sp. def"),
    _p("special attack"),
    _p("special defense"),
    _p("special defence"),
    _p("special atk"),
    _p("special def"),
    # types
    _w("type"),
    _w("typing"),
    _w("weakness"),
    _w("weaknesses"),
    _w("weak"),
    _w("resist"),
    _w("resists"),
    _w("resistance"),
    _w("resistances"),
    _w("immunity"),
    _w("immunities"),
    # moves — use learn/learnset to avoid conflict with competitive "coverage moves"
    _w("learn"),
    _w("learns"),
    _w("learnset"),
    # abilities
    _w("ability"),
    _w("abilities"),
    _p("hidden ability"),
    # evolution
    _w("evolve"),
    _w("evolves"),
    _w("evolution"),
    _p("evolution chain"),
    # forms
    _w("form"),
    _w("forms"),
    _w("mega"),
    _w("galarian"),
    _w("alolan"),
    _w("hisuian"),
    # breeding / physical
    _p("egg group"),
    _w("height"),
    _w("weight"),
    _w("level"),
    # game mechanics — items, status (prevents fallback on non-smogon game-mechanic queries)
    _w("item"),
    _w("items"),
    _w("restore"),
    _w("restores"),
    _w("faint"),
    _w("fainted"),
    _w("nature"),
    _w("natures"),
    _w("berry"),
    _w("berries"),
)

_SMOGON_PATTERNS: tuple[re.Pattern[str], ...] = (
    _w("tier"),
    _w("tiers"),
    _w("ou"),
    _w("uu"),
    _w("ru"),
    _w("nu"),
    _w("pu"),
    _w("lc"),
    _w("ubers"),
    _prefix("competitive"),  # matches "competitive" and "competitively"
    _w("counter"),
    _w("counters"),
    _w("check"),
    _w("checks"),
    _w("coverage"),
    _w("teammate"),
    _w("teammates"),
    # EV and IV must be whole-word only (not substrings of "level", "evolve", "revival")
    _w("ev"),
    _w("evs"),
    _p("ev spread"),
    _w("iv"),
    _w("ivs"),
    _w("strategy"),
    _w("strategies"),
    _w("meta"),
    _w("smogon"),
    _w("vgc"),
    _w("doubles"),
    _w("singles"),
    _w("moveset"),
    _w("viability"),
    _w("synergy"),
    _w("core"),
    _w("set"),
)

_BULBAPEDIA_PATTERNS: tuple[re.Pattern[str], ...] = (
    _w("lore"),
    _p("flavor text"),
    _p("flavour text"),
    _w("pokedex"),
    _p("dex entry"),
    _w("origin"),
    _w("design"),
    _w("anime"),
    _w("manga"),
    _w("history"),
    _w("introduced"),  # "What generation was X introduced?" triggers here
    _w("mythology"),
    _w("backstory"),
    _w("debut"),
)

_ALL_SOURCES: list[Source] = ["bulbapedia", "pokeapi", "smogon"]

_SOURCE_PATTERNS: dict[Source, tuple[re.Pattern[str], ...]] = {
    "pokeapi": _POKEAPI_PATTERNS,
    "smogon": _SMOGON_PATTERNS,
    "bulbapedia": _BULBAPEDIA_PATTERNS,
}


class QueryRouter:
    def route(self, query: str) -> list[Source]:
        stripped = query.strip()
        if not stripped:
            return list(_ALL_SOURCES)

        matched: list[Source] = []
        for source, patterns in _SOURCE_PATTERNS.items():
            if any(p.search(stripped) for p in patterns):
                matched.append(source)

        if not matched:
            return list(_ALL_SOURCES)

        return sorted(matched)
