"""
Collect per-Pokémon wild encounter locations from PokéAPI and write one line per Pokémon.

Output location: data/processed/pokeapi/pokemon_encounters.txt  (staging; user moves to processed/)
Cache location:  data/raw/pokeapi/encounters/{name}.json

Each Pokémon with at least one wild encounter location produces one line:
  "Gengar is a wild Pokémon found in: Old Chateau 2F Right Room (Diamond/Pearl/Platinum,
   level 16, walk); Thrifty Megamart Abandoned Site (Sun/Moon/Ultra Sun/Ultra Moon,
   levels 27-29, walk)."

Pokémon with no encounter locations (starters, legendaries, trade-only) are skipped entirely.
The "is a wild Pokémon" phrasing is intentional — it matches the existing chunker regex for
entity_name extraction without any code changes.
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
_CACHE_DIR = Path("data/raw/pokeapi/encounters")
_OUTPUT_PATH = Path("data/processed/pokeapi/pokemon_encounters.txt")
_CONCURRENCY = 5
_PAGE_SIZE = 200


def _fmt_slug(slug: str) -> str:
    """'old-chateau-2f-right-room' → 'Old Chateau 2F Right Room'"""
    return slug.replace("-", " ").title()


def _pokemon_display_name(raw: str) -> str:
    """'mr-mime' → 'Mr. Mime', 'ho-oh' → 'Ho-Oh'"""
    return " ".join(part.capitalize() for part in raw.split("-"))


def _build_line(name: str, encounters: list[dict]) -> str | None:  # type: ignore[type-arg]
    """
    Build one encounter line for a Pokémon, or None if it has no wild locations.

    Each location area entry is formatted as:
      Location Name (Game1/Game2/..., level X or levels X-Y, method1/method2)

    Versions are deduplicated and kept in original (approximately release) order.
    """
    if not encounters:
        return None

    display = _pokemon_display_name(name)
    parts: list[str] = []

    for entry in encounters:
        location_name = _fmt_slug(entry["location_area"]["name"])

        seen_versions: set[str] = set()
        versions: list[str] = []
        min_level = 100
        max_level = 0
        methods: set[str] = set()

        for vd in entry["version_details"]:
            v = _fmt_slug(vd["version"]["name"])
            if v not in seen_versions:
                seen_versions.add(v)
                versions.append(v)
            for ed in vd["encounter_details"]:
                min_level = min(min_level, ed["min_level"])
                max_level = max(max_level, ed["max_level"])
                methods.add(_fmt_slug(ed["method"]["name"]))

        if min_level == max_level:
            level_str = f"level {min_level}"
        else:
            level_str = f"levels {min_level}-{max_level}"
        version_str = "/".join(versions)
        method_str = "/".join(sorted(methods))

        parts.append(f"{location_name} ({version_str}, {level_str}, {method_str})")

    if not parts:
        return None

    return f"{display} is a wild Pokémon found in: {'; '.join(parts)}."


async def _fetch_json(client: httpx.AsyncClient, url: str) -> list | dict:  # type: ignore[type-arg]
    response = await client.get(url, timeout=30.0)
    response.raise_for_status()
    return response.json()


async def _fetch_encounters(
    name: str,
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> list | None:  # type: ignore[type-arg]
    """Return cached or freshly fetched /pokemon/{name}/encounters JSON, None on error."""
    cache_path = _CACHE_DIR / f"{name}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    url = f"{_BASE_URL}/pokemon/{name}/encounters"
    async with semaphore:
        try:
            data = await _fetch_json(client, url)
        except Exception as exc:
            _LOG.warning("Failed to fetch %s: %s", url, exc)
            return None

    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data  # type: ignore[return-value]


async def _paginate_names(client: httpx.AsyncClient) -> list[str]:
    names: list[str] = []
    url: str | None = f"{_BASE_URL}/pokemon?limit={_PAGE_SIZE}&offset=0"
    while url is not None:
        page = await _fetch_json(client, url)
        names.extend(r["name"] for r in page["results"])  # type: ignore[index]
        url = page.get("next")  # type: ignore[union-attr]
    return names


async def run(output_path: Path = _OUTPUT_PATH) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        _LOG.info("Fetching Pokémon name list…")
        names = await _paginate_names(client)
        _LOG.info("Found %d Pokémon", len(names))

        semaphore = asyncio.Semaphore(_CONCURRENCY)
        tasks = [_fetch_encounters(name, client=client, semaphore=semaphore) for name in names]

        with tqdm(total=len(tasks), desc="fetching", unit="mon") as pbar:
            for coro in asyncio.as_completed(tasks):
                await coro
                pbar.update(1)

    written = 0
    skipped_no_encounters = 0
    skipped_error = 0

    with output_path.open("w", encoding="utf-8") as fh:
        for name in tqdm(names, desc="writing", unit="mon"):
            cache_path = _CACHE_DIR / f"{name}.json"
            if not cache_path.exists():
                skipped_error += 1
                continue
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                _LOG.warning("Skipping %s (cache read error): %s", name, exc)
                skipped_error += 1
                continue

            line = _build_line(name, data)
            if line is None:
                skipped_no_encounters += 1
            else:
                fh.write(line + "\n")
                written += 1

    _LOG.info(
        "Done. %d lines written, %d Pokémon with no encounters (skipped), %d errors. Output: %s",
        written,
        skipped_no_encounters,
        skipped_error,
        output_path,
    )


if __name__ == "__main__":
    asyncio.run(run())
