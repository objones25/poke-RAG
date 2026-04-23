"""Win-rate evaluation for DPO-trained model vs SFT baseline.

Generates one response from each model for every question, scores both with
GeminiJudge (same judge used during DPO data generation), and reports the win rate.

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
    if not candidates:
        return "", []
    c = candidates[0]
    return c.response, c.retrieved_chunks


def evaluate(
    questions: list[str],
    peft_model: Any,
    processor: Any,
    retriever: Any,
    judge: Any,
    output: Path,
    delay: float = 0.5,
) -> dict[str, Any]:
    import time

    from scripts.training.inference_runner import fetch_reference_context

    output.parent.mkdir(parents=True, exist_ok=True)
    wins_dpo = 0
    wins_sft = 0
    ties = 0
    skipped = 0

    with output.open("w", encoding="utf-8") as f:
        for i, question in enumerate(questions):
            log.info("Evaluating question %d / %d", i + 1, len(questions))

            peft_model.set_adapter("sft")
            sft_resp, context = _generate_response(question, retriever, peft_model, processor)

            peft_model.set_adapter("dpo")
            dpo_resp, _ = _generate_response(question, retriever, peft_model, processor)

            reference_chunks = fetch_reference_context(question, retriever)

            sft_score = judge.score_candidate(
                question=question,
                response=sft_resp,
                retrieved_chunks=context,
                reference_chunks=reference_chunks,
            )
            dpo_score = judge.score_candidate(
                question=question,
                response=dpo_resp,
                retrieved_chunks=context,
                reference_chunks=reference_chunks,
            )

            if sft_score is None or dpo_score is None:
                verdict = None
                skipped += 1
            elif dpo_score.total > sft_score.total:
                verdict = "DPO"
                wins_dpo += 1
            elif sft_score.total > dpo_score.total:
                verdict = "SFT"
                wins_sft += 1
            else:
                verdict = "TIE"
                ties += 1

            record: dict[str, Any] = {
                "question": question,
                "sft": sft_resp,
                "dpo": dpo_resp,
                "sft_score": sft_score.model_dump() if sft_score else None,
                "dpo_score": dpo_score.model_dump() if dpo_score else None,
                "verdict": verdict,
            }
            f.write(json.dumps(record) + "\n")

            time.sleep(delay)

    scored = len(questions) - skipped
    win_rate = wins_dpo / scored if scored > 0 else 0.0
    summary = {
        "total": len(questions),
        "scored": scored,
        "skipped": skipped,
        "dpo_wins": wins_dpo,
        "sft_wins": wins_sft,
        "ties": ties,
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
    parser.add_argument("--judge-model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--output", type=Path, default=Path("results/dpo_eval.jsonl"))
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
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

    from peft import PeftModel  # type: ignore[import]
    from qdrant_client import QdrantClient

    from scripts.training.judge_protocol import GeminiJudge
    from src.generation.loader import ModelLoader
    from src.generation.models import GenerationConfig
    from src.retrieval.embedder import BGEEmbedder
    from src.retrieval.retriever import Retriever
    from src.retrieval.vector_store import QdrantVectorStore
    from src.types import RetrievedChunk

    class _PassthroughReranker:
        def rerank(
            self, query: str, documents: list[RetrievedChunk], top_k: int
        ) -> list[RetrievedChunk]:
            return documents[:top_k]

    qdrant = QdrantClient(
        url=args.qdrant_url,
        api_key=os.environ.get("QDRANT_API_KEY"),
    )
    embedder = BGEEmbedder.from_pretrained(model_name="BAAI/bge-m3", device="cuda")
    vector_store = QdrantVectorStore(qdrant)
    retriever = Retriever(
        embedder=embedder, vector_store=vector_store, reranker=_PassthroughReranker()
    )

    gen_config = GenerationConfig(model_id=args.model)
    loader = ModelLoader(config=gen_config, device="cuda")
    loader.load()
    base_model = loader.get_model()
    processor = loader.get_tokenizer()

    peft_model = PeftModel.from_pretrained(base_model, str(args.sft_adapter), adapter_name="sft")
    peft_model.load_adapter(str(args.dpo_adapter), adapter_name="dpo")

    judge = GeminiJudge(api_key=gemini_key, model=args.judge_model)

    summary = evaluate(
        questions=questions,
        peft_model=peft_model,
        processor=processor,
        retriever=retriever,
        judge=judge,
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
