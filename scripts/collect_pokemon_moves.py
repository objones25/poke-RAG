"""
Collect per-Pokémon learnset data from PokéAPI and write one line per learn method.

Output location: data/processed/pokeapi/pokemon_moves.txt  (staging; user moves to processed/)
Cache location:  data/raw/pokeapi/pokemon/{name}.json

Each non-empty Pokémon produces up to four output lines:
  - level-up moves  (with level numbers, from the most recent version group)
  - machine moves   (TM / HM, deduplicated across all version groups)
  - egg moves       (deduplicated across all version groups)
  - tutor moves     (deduplicated across all version groups)

Lines are omitted entirely when a Pokémon has no moves of that type.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import httpx
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_LOG = logging.getLogger(__name__)

_BASE_URL = "https://pokeapi.co/api/v2"
_CACHE_DIR = Path("data/raw/pokeapi/pokemon")
_OUTPUT_PATH = Path("data/processed/pokeapi/pokemon_moves.txt")
_CONCURRENCY = 5
_PAGE_SIZE = 200


def _fmt_name(hyphenated: str) -> str:
    """'shadow-ball' → 'Shadow Ball'"""
    return " ".join(part.capitalize() for part in hyphenated.split("-"))


def _pokemon_display_name(raw: str) -> str:
    """Convert API name to display form: 'mr-mime' → 'Mr. Mime', 'ho-oh' → 'Ho-Oh'."""
    return " ".join(part.capitalize() for part in raw.split("-"))


def _vg_id(url: str) -> int:
    """Extract numeric version-group ID from its URL."""
    return int(url.rstrip("/").rsplit("/", 1)[-1])


def _build_lines(name: str, moves_payload: list[dict]) -> list[str]:  # type: ignore[type-arg]
    """
    Parse PokéAPI moves payload into up to four natural-language lines.

    Returns a (possibly empty) list of strings, one per non-empty learn method.
    """
    display = _pokemon_display_name(name)

    # For level-up: pick the most recent version group per move, then aggregate.
    # We store {move_name: level} using the most recent VG's level for each move.
    levelup: dict[str, int] = {}  # move_name → level (0 = "any")
    machine: set[str] = set()
    egg: set[str] = set()
    tutor: set[str] = set()

    for entry in moves_payload:
        move_name = entry["move"]["name"]
        for vgd in entry["version_group_details"]:
            method = vgd["move_learn_method"]["name"]
            vg_url = vgd["version_group"]["url"]
            vid = _vg_id(vg_url)

            if method == "level-up":
                level = vgd["level_learned_at"]
                # Keep the entry from the highest (most recent) version group
                if move_name not in levelup:
                    levelup[move_name] = (level, vid)  # type: ignore[assignment]
                else:
                    cur_level, cur_vid = levelup[move_name]  # type: ignore[assignment]
                    if vid > cur_vid:
                        levelup[move_name] = (level, vid)  # type: ignore[assignment]
            elif method == "machine":
                machine.add(move_name)
            elif method == "egg":
                egg.add(move_name)
            elif method == "tutor":
                tutor.add(move_name)

    lines: list[str] = []

    if levelup:
        # Sort by level (ascending), then alphabetically for ties.
        sorted_moves = sorted(
            ((mv, lv) for mv, (lv, _) in levelup.items()),  # type: ignore[misc]
            key=lambda x: (x[1], x[0]),
        )
        parts = [f"{_fmt_name(mv)} (level {lv})" for mv, lv in sorted_moves]
        lines.append(f"{display} learns the following moves by leveling up: {', '.join(parts)}.")

    if machine:
        sorted_tm = sorted(_fmt_name(m) for m in machine)
        lines.append(f"{display} can learn the following TM and HM moves: {', '.join(sorted_tm)}.")

    if egg:
        sorted_egg = sorted(_fmt_name(m) for m in egg)
        lines.append(f"{display} can hatch with the following egg moves: {', '.join(sorted_egg)}.")

    if tutor:
        sorted_tutor = sorted(_fmt_name(m) for m in tutor)
        lines.append(
            f"{display} can learn the following moves from a move tutor: {', '.join(sorted_tutor)}."
        )

    return lines


async def _fetch_json(client: httpx.AsyncClient, url: str) -> dict:  # type: ignore[type-arg]
    response = await client.get(url, timeout=30.0)
    response.raise_for_status()
    return response.json()


async def _fetch_pokemon(
    name: str,
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> dict | None:  # type: ignore[type-arg]
    """Return cached or freshly fetched /pokemon/{name}/ JSON, None on error."""
    cache_path = _CACHE_DIR / f"{name}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass  # re-fetch if cache is corrupt

    url = f"{_BASE_URL}/pokemon/{name}/"
    async with semaphore:
        try:
            data = await _fetch_json(client, url)
        except Exception as exc:
            _LOG.warning("Failed to fetch %s: %s", url, exc)
            return None

    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


async def _paginate_names(client: httpx.AsyncClient) -> list[str]:
    """Return all Pokémon names from the list endpoint in national-dex order."""
    names: list[str] = []
    url: str | None = f"{_BASE_URL}/pokemon?limit={_PAGE_SIZE}&offset=0"
    while url is not None:
        page = await _fetch_json(client, url)
        names.extend(r["name"] for r in page["results"])
        url = page.get("next")
    return names


async def run(output_path: Path = _OUTPUT_PATH) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        _LOG.info("Fetching Pokémon name list…")
        names = await _paginate_names(client)
        _LOG.info("Found %d Pokémon", len(names))

        semaphore = asyncio.Semaphore(_CONCURRENCY)

        tasks = [_fetch_pokemon(name, client=client, semaphore=semaphore) for name in names]

        results: list[dict | None] = []  # type: ignore[type-arg]
        with tqdm(total=len(tasks), desc="fetching", unit="mon") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

    # Re-pair names with results in original order so the output is stable.
    # asyncio.as_completed returns in completion order; re-fetch from cache to get ordered output.
    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for name in tqdm(names, desc="writing", unit="mon"):
            cache_path = _CACHE_DIR / f"{name}.json"
            if not cache_path.exists():
                skipped += 1
                continue
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                _LOG.warning("Skipping %s (cache read error): %s", name, exc)
                skipped += 1
                continue

            lines = _build_lines(name, data.get("moves", []))
            for line in lines:
                fh.write(line + "\n")
                written += 1

    _LOG.info(
        "Done. %d lines written, %d Pokémon skipped. Output: %s",
        written,
        skipped,
        output_path,
    )


if __name__ == "__main__":
    asyncio.run(run())
