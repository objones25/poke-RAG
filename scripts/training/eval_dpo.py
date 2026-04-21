"""Win-rate evaluation for DPO-trained model vs SFT baseline.

Generates one response from each model for every question, has Gemini judge
pick the winner, and reports the win rate.

Usage (RunPod, after both models are trained):
    python scripts/training/eval_dpo.py \
        --questions data/dpo/eval_questions.txt \
        --sft-adapter models/pokesage-lora/lora_adapter \
        --dpo-adapter models/pokesage-dpo/lora_adapter \
        --output results/dpo_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_EVAL_PROMPT = """\
You are evaluating two answers to a Pokémon question. Pick the better one.

Question: {question}

Retrieved context: {context}

Answer A: {answer_a}

Answer B: {answer_b}

Which answer is better overall (accuracy, groundedness, and domain correctness)?
Reply with ONLY "A" or "B".
"""


def _load_questions(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _generate_response(
    question: str,
    retriever: Any,
    model: Any,
    processor: Any,
) -> tuple[str, list[str]]:
    from scripts.training.inference_runner import generate_candidates

    candidates = generate_candidates(question, retriever, model, processor, k=1)
    c = candidates[0]
    return c.response, c.retrieved_chunks


def _judge_winner(
    question: str,
    answer_a: str,
    answer_b: str,
    context: list[str],
    judge_client: Any,
    judge_model: str,
) -> str | None:
    contents = _EVAL_PROMPT.format(
        question=question,
        context="\n".join(context),
        answer_a=answer_a,
        answer_b=answer_b,
    )
    try:
        resp = judge_client.models.generate_content(model=judge_model, contents=contents)
        verdict = resp.text.strip().upper()
        return verdict if verdict in {"A", "B"} else None
    except Exception as exc:
        log.warning("Judge call failed: %s", exc)
        return None


def evaluate(
    questions: list[str],
    sft_model: Any,
    dpo_model: Any,
    processor: Any,
    retriever: Any,
    judge_client: Any,
    judge_model: str,
    output: Path,
    delay: float = 0.5,
) -> dict[str, Any]:
    import time

    output.parent.mkdir(parents=True, exist_ok=True)
    wins_dpo = 0
    wins_sft = 0
    skipped = 0

    with output.open("w", encoding="utf-8") as f:
        for i, question in enumerate(questions):
            log.info("Evaluating question %d / %d", i + 1, len(questions))

            sft_resp, context = _generate_response(question, retriever, sft_model, processor)
            dpo_resp, _ = _generate_response(question, retriever, dpo_model, processor)

            verdict = _judge_winner(
                question, sft_resp, dpo_resp, context, judge_client, judge_model
            )

            record: dict[str, Any] = {
                "question": question,
                "sft": sft_resp,
                "dpo": dpo_resp,
                "verdict": verdict,
            }
            f.write(json.dumps(record) + "\n")

            if verdict == "A":
                wins_sft += 1
            elif verdict == "B":
                wins_dpo += 1
            else:
                skipped += 1

            time.sleep(delay)

    scored = len(questions) - skipped
    win_rate = wins_dpo / scored if scored > 0 else 0.0
    summary = {
        "total": len(questions),
        "scored": scored,
        "skipped": skipped,
        "dpo_wins": wins_dpo,
        "sft_wins": wins_sft,
        "dpo_win_rate": win_rate,
    }
    log.info("DPO win rate: %.1f%% (%d/%d)", 100 * win_rate, wins_dpo, scored)
    return summary


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Win-rate evaluation: DPO vs SFT baseline.")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--dpo-adapter", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--judge-model", default="gemini-2.0-flash")
    parser.add_argument("--output", type=Path, default=Path("results/dpo_eval.jsonl"))
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        parser.error("GEMINI_API_KEY not set.")

    if not torch_cuda_available():
        parser.error("CUDA required for model inference.")

    questions = _load_questions(args.questions)
    if not questions:
        parser.error(f"No questions found in {args.questions}")

    import importlib.util

    if importlib.util.find_spec("unsloth") is None:
        parser.error("Unsloth required. Run runpod_setup.sh first.")

    from google import genai
    from peft import PeftModel  # type: ignore[import]
    from qdrant_client import QdrantClient

    from src.generation.loader import ModelLoader
    from src.retrieval.embedder import BGEEmbedder
    from src.retrieval.reranker import BGEReranker
    from src.retrieval.retriever import Retriever
    from src.retrieval.vector_store import QdrantVectorStore

    qdrant = QdrantClient(
        url=args.qdrant_url,
        api_key=os.environ.get("QDRANT_API_KEY"),
    )
    embedder = BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cuda")
    reranker = BGEReranker.from_pretrained(model_name="BAAI/bge-reranker-v2-m3", device="cuda")
    vector_store = QdrantVectorStore(qdrant)
    retriever = Retriever(embedder=embedder, vector_store=vector_store, reranker=reranker)

    loader = ModelLoader(model_id=args.model)
    base_model, processor = loader.load()

    sft_model = PeftModel.from_pretrained(base_model, str(args.sft_adapter))
    dpo_model = PeftModel.from_pretrained(base_model, str(args.dpo_adapter))

    judge_client = genai.Client(api_key=gemini_key)

    summary = evaluate(
        questions=questions,
        sft_model=sft_model,
        dpo_model=dpo_model,
        processor=processor,
        retriever=retriever,
        judge_client=judge_client,
        judge_model=args.judge_model,
        output=args.output,
        delay=args.delay,
    )

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Summary written to %s", summary_path)


def torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
