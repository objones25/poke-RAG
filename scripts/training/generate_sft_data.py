from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.training.gemini_client import GeminiClient  # noqa: E402
from scripts.training.sampler import ChunkSampler, extract_entity_name  # noqa: E402
from scripts.training.schemas import SFTDatapoint, SFTMessage  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS: dict[str, float] = {"bulbapedia": 0.4, "pokeapi": 0.4, "smogon": 0.2}

_MIN_CHUNK_LEN = 80
_VAR_RE = re.compile(r"\[VAR\s*\(", re.IGNORECASE)
_DB_ID_RE = re.compile(r"\b\w+\d{3,}\b")  # catches "Dra6688", "Item0042", etc.


def _is_useful_chunk(chunk: str) -> bool:
    if len(chunk) < _MIN_CHUNK_LEN:
        return False
    if _VAR_RE.search(chunk):
        return False
    return not _DB_ID_RE.search(chunk)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run(
    goal: int,
    output: Path,
    processed_dir: Path,
    api_key: str,
    model: str,
    seed: int,
    delay: float,
    max_per_entity: int,
    source_weights: dict[str, float],
    include_aug: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    start_count = _count_lines(output)
    if start_count >= goal:
        logger.info("Goal already reached (%d pairs). Nothing to do.", start_count)
        return
    logger.info("Starting from %d / %d pairs.", start_count, goal)

    sampler = ChunkSampler(processed_dir, source_weights, include_aug=include_aug, seed=seed)
    logger.info("Loaded lines per source: %s", sampler.total_available())

    client = GeminiClient(api_key=api_key, model=model)
    entity_counts: dict[str, int] = defaultdict(int)
    generated = start_count
    attempted = 0
    skipped_chunk = 0
    skipped_dedup = 0
    skipped_quality = 0

    with open(output, "a", encoding="utf-8", buffering=1) as f:
        while generated < goal:
            result = sampler.sample()
            if result is None:
                logger.warning("Sampler exhausted all available lines at %d pairs.", generated)
                break
            chunk, source = result

            if not _is_useful_chunk(chunk):
                skipped_chunk += 1
                continue

            entity = extract_entity_name(chunk, source)
            if entity and entity_counts[entity] >= max_per_entity:
                skipped_dedup += 1
                continue

            attempted += 1
            try:
                pair = client.generate_qa_pair(chunk, source)
            except Exception as exc:
                logger.warning("Skipping chunk due to error: %s", exc)
                continue

            if pair is None:
                skipped_quality += 1
                continue

            datapoint = SFTDatapoint(
                messages=[
                    SFTMessage(role="user", content=pair.question),
                    SFTMessage(role="assistant", content=pair.answer),
                ]
            )
            f.write(datapoint.model_dump_json() + "\n")
            if entity:
                entity_counts[entity] += 1
            generated += 1
            if generated % 100 == 0:
                logger.info(
                    "Progress: %d / %d (attempted=%d, skipped_chunk=%d,"
                    " skipped_quality=%d, dedup=%d)",
                    generated,
                    goal,
                    attempted,
                    skipped_chunk,
                    skipped_quality,
                    skipped_dedup,
                )
            time.sleep(delay)

    logger.info(
        "Done. Generated %d new pairs. Total: %d / %d. "
        "Attempted=%d, skipped_chunk=%d, skipped_quality=%d, dedup=%d.",
        generated - start_count,
        generated,
        goal,
        attempted,
        skipped_chunk,
        skipped_quality,
        skipped_dedup,
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Generate SFT Q&A training data using Gemini Flash Lite."
    )
    parser.add_argument("--goal", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("data/sft/train.jsonl"))
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--max-per-entity", type=int, default=5)
    parser.add_argument("--source-weights", type=json.loads, default=_DEFAULT_WEIGHTS)
    parser.add_argument("--include-aug", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        parser.error("GEMINI_API_KEY not set.")

    try:
        run(
            goal=args.goal,
            output=args.output,
            processed_dir=args.processed_dir,
            api_key=api_key,
            model=args.model,
            seed=args.seed,
            delay=args.delay,
            max_per_entity=args.max_per_entity,
            source_weights=args.source_weights,
            include_aug=args.include_aug,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
