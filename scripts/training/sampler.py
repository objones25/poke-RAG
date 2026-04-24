from __future__ import annotations

import random
from pathlib import Path


def extract_entity_name(line: str, source: str) -> str | None:
    separators = [" (", " is a ", " is an ", ": "]
    for sep in separators:
        idx = line.find(sep)
        if 0 < idx <= 60:
            return line[:idx].strip()
    return None


class ChunkSampler:
    def __init__(
        self,
        processed_dir: Path,
        weights: dict[str, float],
        include_aug: bool = False,
        seed: int = 42,
    ) -> None:
        self._rng = random.Random(seed)
        self._lines: dict[str, list[str]] = {}
        self._indices: dict[str, int] = {}

        for source in ("bulbapedia", "pokeapi", "smogon"):
            source_dir = processed_dir / source
            if not source_dir.exists():
                continue
            lines: list[str] = []
            for path in sorted(source_dir.glob("*.txt")):
                if not include_aug and "_aug" in path.stem:
                    continue
                with open(path, encoding="utf-8") as f:
                    for raw in f:
                        stripped = raw.strip()
                        if stripped:
                            lines.append(stripped)
            self._rng.shuffle(lines)
            self._lines[source] = lines
            self._indices[source] = 0

        self._weights = {k: v for k, v in weights.items() if k in self._lines}

    def total_available(self) -> dict[str, int]:
        return {src: len(self._lines[src]) - self._indices.get(src, 0) for src in self._lines}

    def sample(self) -> tuple[str, str] | None:
        available = {
            src: w
            for src, w in self._weights.items()
            if self._indices.get(src, 0) < len(self._lines.get(src, []))
        }
        if not available:
            return None
        total_w = sum(available.values())
        if total_w == 0:
            return None
        sources = list(available.keys())
        probs = [available[s] / total_w for s in sources]
        source = self._rng.choices(sources, weights=probs, k=1)[0]
        idx = self._indices[source]
        line = self._lines[source][idx]
        self._indices[source] = idx + 1
        return line, source
