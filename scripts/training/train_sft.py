"""QLoRA SFT training script for google/gemma-4-E4B-it via Unsloth + TRL.

Usage (RunPod):
    python scripts/training/train_sft.py \
        --data data/sft/train.jsonl \
        --output-dir models/pokesage-lora \
        --epochs 3 \
        --run-name pokesage-v1

Requirements:
    - CUDA GPU with compute capability >= 8.0 (A100, H100)
    - Unsloth installed from GitHub main (NOT PyPI) — see runpod_setup.sh
    - Run runpod_setup.sh before this script
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QLoRA SFT fine-tuning of gemma-4-E4B-it via Unsloth + TRL"
    )
    p.add_argument(
        "--model",
        default="google/gemma-4-E4B-it",
        help="HuggingFace model ID or local path (default: google/gemma-4-E4B-it)",
    )
    p.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to training JSONL (messages format)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to save LoRA adapter and checkpoints",
    )
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.1, help="Fraction held out for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (uses bfloat16 full precision)",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B experiment tracking",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="W&B run name (defaults to output-dir basename)",
    )
    return p


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


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


def _apply_chat_template(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    """Pre-apply the Gemma chat template so SFTTrainer gets plain text strings."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# GPU / environment checks
# ---------------------------------------------------------------------------


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
    if vram_gb < 40:
        log.warning(
            "%.1f GB VRAM detected. Gemma-4-E4B 4-bit + LoRA typically needs ~40 GB. "
            "Reduce --batch-size or --max-seq-length if you hit OOM.",
            vram_gb,
        )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    _check_environment()

    # Lazy imports — these must only be present on the RunPod training machine
    try:
        from unsloth import FastModel  # type: ignore[import]
    except ImportError:
        log.error(
            "Unsloth is not installed. Run runpod_setup.sh first.\n"
            "  bash scripts/training/runpod_setup.sh"
        )
        sys.exit(1)

    try:
        from datasets import Dataset  # type: ignore[import]
        from trl import SFTConfig, SFTTrainer  # type: ignore[import]
    except ImportError as e:
        log.error("Missing dependency: %s. Run runpod_setup.sh first.", e)
        sys.exit(1)

    # ------------------------------------------------------------------
    # W&B setup
    # ------------------------------------------------------------------
    run_name = args.run_name or args.output_dir.name
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"
        log.info("W&B disabled")
    else:
        wandb_key = os.environ.get("WANDB_API_KEY")
        if not wandb_key:
            log.warning(
                "WANDB_API_KEY not set. W&B logging will be anonymous or may fail. "
                "Set it via: export WANDB_API_KEY=<your-key>"
            )
        report_to = "wandb"
        os.environ.setdefault("WANDB_PROJECT", "pokesage-sft")
        os.environ.setdefault("WANDB_RUN_NAME", run_name)
        log.info("W&B project=pokesage-sft  run=%s", run_name)

    # ------------------------------------------------------------------
    # Load model + tokenizer via Unsloth FastModel
    # ------------------------------------------------------------------
    log.info("Loading model: %s", args.model)
    log.info(
        "  4-bit=%s  max_seq_length=%d  dtype=None (auto bfloat16)",
        not args.no_4bit,
        args.max_seq_length,
    )

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # auto-detect; bfloat16 on A100/H100
        load_in_4bit=not args.no_4bit,
        full_finetuning=False,
    )

    # ------------------------------------------------------------------
    # Attach LoRA adapter
    # Freeze vision layers — text-only SFT on a multimodal base model.
    # ------------------------------------------------------------------
    log.info("Attaching LoRA adapter (r=%d, alpha=%d)", args.lora_r, args.lora_alpha)
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

    # ------------------------------------------------------------------
    # Load and prepare dataset
    # ------------------------------------------------------------------
    log.info("Loading dataset from %s", args.data)
    records = _load_jsonl(args.data)
    if not records:
        log.error("No records found in %s", args.data)
        sys.exit(1)
    log.info("Loaded %d records", len(records))

    raw = Dataset.from_list(records)

    splits = raw.train_test_split(test_size=args.val_frac, seed=args.seed)
    train_ds = splits["train"]
    val_ds = splits["test"]
    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    # Pre-apply chat template — converts messages list → plain text string
    def _fmt(example: dict[str, Any]) -> dict[str, str]:
        return _apply_chat_template(example, tokenizer)

    train_ds = train_ds.map(_fmt, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(_fmt, remove_columns=val_ds.column_names)

    # ------------------------------------------------------------------
    # SFTConfig
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        run_name=run_name,
        # Batching
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # Sequence length
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=True,  # pack short samples for efficiency
        # Optimization
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        # Precision — bfloat16 on A100/H100
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        # Evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_steps=10,
        report_to=report_to,
        # Reproducibility
        seed=args.seed,
        # Speed
        dataloader_num_workers=4,
        remove_unused_columns=True,
    )

    # ------------------------------------------------------------------
    # SFTTrainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # trl >= 0.12 uses processing_class
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    log.info("Starting training …")
    trainer.train()

    # ------------------------------------------------------------------
    # Save LoRA adapter
    # ------------------------------------------------------------------
    adapter_path = args.output_dir / "lora_adapter"
    log.info("Saving LoRA adapter to %s", adapter_path)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    log.info("Done. Best eval_loss saved to %s", args.output_dir)


# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
