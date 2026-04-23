"""QLoRA DPO training script for google/gemma-4-E4B-it via Unsloth + TRL.

Usage (RunPod):
    python scripts/training/train_dpo.py \
        --data data/dpo/train.jsonl \
        --sft-adapter models/pokesage-lora/lora_adapter \
        --output-dir models/pokesage-dpo \
        --run-name pokesage-dpo-v1

Requirements:
    - CUDA GPU with compute capability >= 8.0 (A100, H100)
    - Unsloth installed from GitHub main — see runpod_setup.sh
    - SFT adapter already trained via train_sft.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QLoRA DPO fine-tuning of gemma-4-E4B-it via Unsloth + TRL"
    )
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument(
        "--sft-adapter",
        type=Path,
        default=None,
        help="Path to SFT LoRA adapter to load before DPO (optional).",
    )
    p.add_argument("--data", required=True, type=Path, help="DPO JSONL (prompt/chosen/rejected)")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1, help="DPO KL penalty coefficient.")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--run-name", default=None)
    return p


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("Skipping malformed JSON at line %d: %s", lineno, e)
    return records


def _check_environment() -> None:
    if not torch.cuda.is_available():
        log.error("No CUDA device found. This script requires a CUDA GPU.")
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    compute_cap = props.major + props.minor / 10
    log.info("GPU: %s  (compute capability %.1f)", props.name, compute_cap)

    if compute_cap < 8.0:
        log.error(
            "GPU compute capability %.1f < 8.0. "
            "bfloat16 and efficient QLoRA require A100 (sm_80) or newer.",
            compute_cap,
        )
        sys.exit(1)

    vram_gb = props.total_memory / (1024**3)
    log.info("VRAM: %.1f GB", vram_gb)


def train(args: argparse.Namespace) -> None:
    _check_environment()

    # Validate cheap preconditions before the expensive model load.
    if args.sft_adapter is not None and not args.sft_adapter.exists():
        log.error("SFT adapter path does not exist: %s", args.sft_adapter)
        sys.exit(1)

    log.info("Loading DPO dataset from %s", args.data)
    records = _load_jsonl(args.data)
    if not records:
        log.error("No records found in %s", args.data)
        sys.exit(1)
    required_cols = {"prompt", "chosen", "rejected"}
    missing = required_cols - records[0].keys()
    if missing:
        log.error("DPO dataset missing required columns: %s", missing)
        sys.exit(1)
    log.info("Dataset OK: %d DPO pairs", len(records))

    try:
        from unsloth import FastModel  # type: ignore[import]
    except ImportError:
        log.error("Unsloth is not installed. Run runpod_setup.sh first.")
        sys.exit(1)

    try:
        from datasets import Dataset  # type: ignore[import]
        from trl import DPOConfig, DPOTrainer  # type: ignore[import]
    except ImportError as e:
        log.error("Missing dependency: %s. Run runpod_setup.sh first.", e)
        sys.exit(1)

    run_name = args.run_name or args.output_dir.name
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"
    else:
        os.environ.setdefault("WANDB_PROJECT", "pokesage-dpo")
        os.environ.setdefault("WANDB_RUN_NAME", run_name)
        report_to = "wandb"
    log.info("W&B: %s  run=%s", report_to, run_name)

    log.info("Loading model: %s", args.model)
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=not args.no_4bit,
        full_finetuning=False,
    )

    if args.sft_adapter is not None:
        log.info("Merging SFT adapter from %s into base weights", args.sft_adapter)
        from peft import PeftModel  # type: ignore[import]

        model = PeftModel.from_pretrained(model, str(args.sft_adapter))
        model = model.merge_and_unload()
        log.info("SFT adapter merged — will serve as implicit DPO reference")

    log.info("Attaching DPO LoRA adapter (r=%d, alpha=%d)", args.lora_r, args.lora_alpha)
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )

    raw = Dataset.from_list(records)
    splits = raw.train_test_split(test_size=args.val_frac, seed=args.seed)
    train_ds, val_ds = splits["train"], splits["test"]
    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dpo_config = DPOConfig(
        output_dir=str(args.output_dir),
        run_name=run_name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_seq_length // 2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        report_to=report_to,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # None uses the frozen base as implicit reference via PEFT
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    log.info("Starting DPO training …")
    trainer.train()

    adapter_path = args.output_dir / "lora_adapter"
    log.info("Saving DPO LoRA adapter to %s", adapter_path)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    log.info("Done.")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
