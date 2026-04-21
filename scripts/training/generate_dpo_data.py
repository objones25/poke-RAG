from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.training.inference_runner import (  # noqa: E402
    fetch_reference_context,
    generate_candidates,
)
from scripts.training.judge_protocol import GeminiJudge, JudgeProtocol  # noqa: E402
from scripts.training.schemas import DPODatapoint  # noqa: E402
from src.retrieval.protocols import RetrieverProtocol  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _load_questions(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def run(
    goal: int,
    output: Path,
    questions: list[str],
    retriever: RetrieverProtocol,
    model: Any,
    processor: Any,
    judge: JudgeProtocol,
    *,
    k: int = 5,
    delay: float = 0.5,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    start_count = _count_lines(output)
    if start_count >= goal:
        logger.info("Goal already reached (%d pairs). Nothing to do.", start_count)
        return

    question_offset = start_count
    logger.info(
        "Starting from %d / %d pairs (question offset=%d).",
        start_count,
        goal,
        question_offset,
    )

    generated = start_count
    skipped = 0

    with open(output, "a", encoding="utf-8", buffering=1) as f:
        for question in questions[question_offset:]:
            if generated >= goal:
                break

            candidates = generate_candidates(question, retriever, model, processor, k=k)
            reference_chunks = fetch_reference_context(question, retriever)

            scored: list[tuple[str, int]] = []
            for c in candidates:
                score = judge.score_candidate(
                    question=question,
                    response=c.response,
                    retrieved_chunks=c.retrieved_chunks,
                    reference_chunks=reference_chunks,
                )
                if score is not None:
                    scored.append((c.response, score.total))

            if len(scored) < 2:
                skipped += 1
                logger.warning(
                    "Skipping question (only %d scored candidates): %.60s",
                    len(scored),
                    question,
                )
                continue

            scored.sort(key=lambda x: x[1])
            rejected, chosen = scored[0][0], scored[-1][0]

            datapoint = DPODatapoint(prompt=question, chosen=chosen, rejected=rejected)
            f.write(datapoint.model_dump_json() + "\n")
            generated += 1
            if generated % 50 == 0 or generated == goal:
                logger.info("Progress: %d / %d (skipped=%d).", generated, goal, skipped)
            time.sleep(delay)

    logger.info(
        "Done. Total pairs: %d / %d. Skipped: %d.",
        generated,
        goal,
        skipped,
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs using LLM-as-judge (RLAIF)."
    )
    parser.add_argument("--goal", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("data/dpo/train.jsonl"))
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/dpo/questions.txt"),
        help="One question per line.",
    )
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--judge-model", default="gemini-2.0-flash")
    parser.add_argument("--k", type=int, default=5, help="Candidates per question.")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        parser.error("GEMINI_API_KEY not set.")

    questions = _load_questions(args.questions)
    if not questions:
        parser.error(f"No questions found in {args.questions}")

    from qdrant_client import QdrantClient

    from src.generation.loader import ModelLoader
    from src.retrieval.retriever import Retriever

    qdrant = QdrantClient(url=args.qdrant_url)
    retriever = Retriever(qdrant_client=qdrant)

    loader = ModelLoader(model_id=args.model_id)
    model, processor = loader.load()

    judge = GeminiJudge(api_key=gemini_key, model=args.judge_model)

    try:
        run(
            goal=args.goal,
            output=args.output,
            questions=questions,
            retriever=retriever,
            model=model,
            processor=processor,
            judge=judge,
            k=args.k,
            delay=args.delay,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
