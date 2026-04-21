# SFT Implementation: pokesage-lora

**Last Updated:** 2026-04-21

This document covers the Supervised Fine-Tuning (SFT) pipeline for the `pokesage-v1` LoRA adapter, a low-rank fine-tune of `google/gemma-4-E4B-it` on Pokémon expert Q&A data.

## 1. Overview

**Goal**: Fine-tune Gemma 4 on Pokémon knowledge from three sources (Bulbapedia, PokéAPI, Smogon) using LoRA (PEFT) to produce better, more grounded answers aligned with competitive and PokéDex-accurate information.

**Deployment**: The trained adapter (`pokesage-v1`) is published to HuggingFace Hub as [`objones25/pokesage-lora`](https://huggingface.co/objones25/pokesage-lora) and integrated into inference via `src/generation/loader.py`.

**Key Properties**:

- **Model**: `google/gemma-4-E4B-it` (4B parameters, multimodal)
- **Adapter size**: ~147 MB (adapter_model.safetensors + tokenizer.json)
- **Adapter config**: r=16, alpha=16, targeting all linear layers, dropout=0
- **Training framework**: Unsloth (2x speedup) + TRL + PEFT + 4-bit quantization
- **Hardware**: H100 (RunPod), ~60 GB VRAM with packing
- **Data**: 2000 Q&A pairs in chat format (user/assistant turns, no system message)

## 2. SFT Data Generation Pipeline

### 2.1 Overview

The data pipeline samples chunks from `processed/bulbapedia/`, `processed/pokeapi/`, and `processed/smogon/`, generates synthetic Q&A pairs via Gemini Flash, and deduplicates/filters the results.

```
processed/bulbapedia/*.txt
  ↓
ChunkSampler (weighted sampling)
  ↓
GeminiClient (generate_sft_data.py)
  ↓
clean_sft_data.py (deduplicate, filter)
  ↓
data/sft/train.jsonl (2000 examples)
```

### 2.2 Scripts

| Script                                  | Purpose                                                                         |
| --------------------------------------- | ------------------------------------------------------------------------------- |
| `scripts/training/generate_sft_data.py` | Orchestrates the pipeline: samples chunks, calls Gemini, writes JSONL           |
| `scripts/training/gemini_client.py`     | Wraps Google Generative AI SDK; generates Q&A pairs with retry + quality filter |
| `scripts/training/schemas.py`           | Pydantic models: `GeminiQAPair`, `SFTMessage`, `SFTDatapoint`                   |
| `scripts/training/sampler.py`           | Loads and samples chunks from `processed/` with weighted source distribution    |
| `scripts/training/pokesage_system.py`   | System prompt used during fine-tuning                                           |
| `scripts/training/clean_sft_data.py`    | Removes bad Q&A pairs, normalizes message structure                             |

### 2.3 Chunk Sampling

**ChunkSampler** (`sampler.py`):

- Loads all lines from `processed/{source}/*.txt` into memory
- Applies source weights: `{"bulbapedia": 0.4, "pokeapi": 0.4, "smogon": 0.2}` by default
- Returns (chunk, source) tuples uniformly from weighted sources
- Optionally includes `*_aug.txt` paraphrased variants via `--include-aug`

**Entity tracking**:

- Extracts entity name (e.g., "Charizard", "Earthquake") from chunk via pattern matching
- Limits Q&A pairs per entity via `--max-per-entity` (default: 5) to prevent overfitting to common Pokémon

### 2.4 Q&A Generation

**GeminiClient** (`gemini_client.py`):

1. Samples a chunk from ChunkSampler
2. Builds prompt from template with source-specific hints
3. Calls `models.generate_content()` with Gemini Flash (e.g., `gemini-3.1-flash-lite-preview`)
4. Uses JSON schema constraint to ensure structured output
5. Validates response with regex filters against common bad-answer patterns

**Quality filters**:

- Rejects empty questions/answers
- Removes answers shorter than 40 chars
- Filters answers matching "is a Pokémon" pattern with no details
- Rejects answers containing placeholder phrases like "provided text does not contain", "not enough information", "cannot be answered"

**Retry logic**:

- Max 3 attempts per chunk
- Exponential backoff (2^attempt seconds) for rate limit (429, RESOURCE_EXHAUSTED)
- Falls back to lenient JSON parse if Pydantic validation fails
- Raises RuntimeError if all attempts fail

**Output format**:

```json
{
  "messages": [
    { "role": "user", "content": "What is Earthquake's base power?" },
    {
      "role": "assistant",
      "content": "Earthquake has a base power of 100 and 100% accuracy."
    }
  ]
}
```

### 2.5 Data Cleaning

**clean_sft_data.py**:

- Strips system messages (keeps only user/assistant)
- Removes entries with invalid JSON or unexpected role ordering
- Filters assistant responses with VAR placeholders, internal DB IDs, or Dynamax Crystal references
- Reapplies all Q&A quality filters
- Atomic file writing via temp file + atomic replace
- Outputs stats: total, kept, removed, keep %

**Usage**:

```bash
uv run python scripts/training/clean_sft_data.py data/sft/raw.jsonl --inplace
```

### 2.6 Running the Pipeline

```bash
export GEMINI_API_KEY="your-api-key"
uv run python scripts/training/generate_sft_data.py \
  --goal 2000 \
  --output data/sft/train.jsonl \
  --processed-dir processed \
  --model gemini-3.1-flash-lite-preview \
  --max-per-entity 5 \
  --delay 0.5 \
  --seed 42
```

**Parameters**:

- `--goal`: Target number of Q&A pairs (default: 2000)
- `--output`: Path to JSONL file (appends if exists)
- `--processed-dir`: Root of processed/ data
- `--model`: Gemini model ID (default: `gemini-3.1-flash-lite-preview`)
- `--max-per-entity`: Max Q&A pairs per Pokémon/move (default: 5)
- `--delay`: Sleep between API calls in seconds (default: 0.5)
- `--include-aug`: Include `*_aug.txt` paraphrased variants (off by default)

## 3. Training Setup (RunPod H100)

### 3.1 Prerequisites

- **GPU**: H100 or A100 (compute capability ≥ 8.0)
- **VRAM**: ~60 GB with packing, ~40 GB minimum
- **Driver**: 580+ (RunPod default); check `nvidia-smi`
- **Python**: 3.10+

### 3.2 Environment Setup

See `scripts/training/RUNPOD_SETUP_NOTES.md` for the full hard-won lessons. TL;DR:

**Critical dependency chain** (must be installed in this exact order):

```bash
pip install "setuptools==80.9.0"
pip install "packaging>=24.2"
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git
pip install "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git" \
  --no-build-isolation
```

**Why from GitHub, not PyPI**:

- PyPI Unsloth **lacks** the Gemma 4 gradient accumulation fix
- PyPI unsloth_zoo unconditionally pulls torchao 0.17.0, which requires torch 2.7.0 and breaks with 2.6.0
- Installing with `--no-deps` avoids this dependency hell

**Do NOT install xformers** — it will silently pull torch 2.11.0+ and break unsloth_zoo.

### 3.3 Training Script

**Path**: `scripts/training/train_sft.py`

**Key points**:

- Uses Unsloth's `FastLanguageModel` for model loading and LoRA attachment
- 4-bit quantization via Unsloth (optional: `--no-4bit` for full bfloat16)
- LoRA frozen to language layers only (`finetune_vision_layers=False`)
- Packing enabled (`packing=True`), though Gemma 4's multimodal processor reports as skipped
- Flash Attention 2 disabled by default (Gemma 4 incompatibility); uses SDPA instead
- Gradient checkpointing enabled for memory efficiency
- W&B logging optional (`--no-wandb` to disable)

### 3.4 Training Command

```bash
python scripts/training/train_sft.py \
  --data data/sft/train.jsonl \
  --output-dir models/pokesage-lora \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-4 \
  --warmup-steps 10 \
  --max-seq-length 2048 \
  --lora-r 16 \
  --lora-alpha 16 \
  --val-frac 0.1 \
  --run-name pokesage-v1
```

**Key hyperparameters**:

- **Batch size**: 4 per device (H100 VRAM allows up to 8)
- **Gradient accumulation**: 4 steps (effective batch 16)
- **Learning rate**: 2e-4 (cosine decay with 10 warmup steps)
- **LoRA**: r=16, alpha=16 (16:1 scaling), dropout=0, all linear layers
- **Validation**: 10% held out, evaluation per epoch
- **Save strategy**: Save best checkpoint by eval_loss

### 3.5 Expected Warnings (Not Errors)

The following appear on every clean Gemma 4 run. They are expected:

1. **Flash Attention 2 reports as broken** — Training uses SDPA instead, performance is normal
2. **Sample packing skipped** — Gemma 4 is a multimodal model; Unsloth detects and skips packing
3. **KV cache disabled during gradient checkpointing** — Expected Gemma 4 behavior; does not affect training

## 4. Training Results (pokesage-v1, April 2026)

### 4.1 Metrics

| Epoch | Train loss (end) | Eval loss | Notes                       |
| ----- | ---------------- | --------- | --------------------------- |
| 1     | 1.93             | 2.92      | Baseline                    |
| 2     | 1.54             | **2.82**  | **Best checkpoint** (saved) |
| 3     | 1.34             | 2.901     | Mild overfitting            |

**Best checkpoint**: Epoch 2 (eval_loss 2.82), saved to `models/pokesage-lora/lora_adapter/`

**Observations**:

- Steady loss decrease through epoch 2
- Mild overfitting at epoch 3 (expected with 2000 pairs)
- Training time: ~4 hours on H100

### 4.2 Artifact Locations

| Artifact     | Path                                                          |
| ------------ | ------------------------------------------------------------- |
| LoRA weights | `models/pokesage-lora/lora_adapter/adapter_model.safetensors` |
| Tokenizer    | `models/pokesage-lora/lora_adapter/tokenizer.json`            |
| HF Hub       | https://huggingface.co/objones25/pokesage-lora                |
| W&B run      | https://wandb.ai/objones25/pokesage-sft/runs/ht1h2qpd         |

### 4.3 Model Size

- Base model: ~4.0 GB (Gemma 4 4B-it)
- LoRA adapter: ~147 MB
- Combined at inference: ~4.1 GB (adapter merged/loaded alongside base)

## 5. Inference Integration

### 5.1 ModelLoader Integration

**File**: `src/generation/loader.py`

The adapter is wired into inference via the `ModelLoader` class:

```python
class ModelLoader:
    def __init__(
        self,
        config: GenerationConfig,
        device: str,
        lora_adapter_path: str | None = None,
    ) -> None:
        self._lora_adapter_path = lora_adapter_path
        ...

    def load(self) -> None:
        raw_model = AutoModelForImageTextToText.from_pretrained(...)
        self._model = self._apply_lora_adapter(raw_model)

    def _apply_lora_adapter(self, model: PreTrainedModel) -> PreTrainedModel:
        if self._lora_adapter_path is None:
            return model
        source = (
            self._lora_adapter_path
            if Path(self._lora_adapter_path).exists()
            else "objones25/pokesage-lora"  # HF Hub fallback
        )
        return PeftModel.from_pretrained(model, source)
```

### 5.2 Configuration

**Environment variable**: `LORA_ADAPTER_PATH`

Set via `.env` or shell:

```bash
LORA_ADAPTER_PATH=models/pokesage-lora
```

**Behavior**:

- If unset: base model only (no LoRA)
- If set and path exists locally: load from local path
- If set but path missing: fall back to HF Hub (`objones25/pokesage-lora`)
- If HF Hub load fails: raise `RuntimeError` (fail-fast, no silent fallback to base)

### 5.3 Usage in API

The `src/api/app.py` initializes ModelLoader via FastAPI startup:

```python
# In lifespan setup
lora_path = os.getenv("LORA_ADAPTER_PATH")
loader = ModelLoader(config, device=device, lora_adapter_path=lora_path)
loader.load()
```

Both `PipelineResult` and `QueryResponse` include an optional `confidence_score` field (future use for RLHF filtering).

### 5.4 Local Testing

Download the adapter from HF Hub:

```bash
uv run pip install huggingface-hub
hf download objones25/pokesage-lora --local-dir models/pokesage-lora
```

Set the env var and run the API:

```bash
export LORA_ADAPTER_PATH=models/pokesage-lora
uv run python -m src.api.app
```

Query the endpoint:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Earthquake", "sources": ["pokeapi"]}'
```

## 6. Reproducing the Pipeline

### 6.1 Full Workflow

**Phase 1: Data Generation** (local, ~1 GPU hour)

```bash
export GEMINI_API_KEY="your-api-key"
uv sync --all-extras
uv run python scripts/training/generate_sft_data.py \
  --goal 2000 --output data/sft/train.jsonl --seed 42
```

**Phase 2: Data Cleaning**

```bash
uv run python scripts/training/clean_sft_data.py data/sft/train.jsonl --inplace
```

**Phase 3: Training** (RunPod H100, ~4 hours)

1. Spin up H100 pod on RunPod
2. Run `bash scripts/training/runpod_setup.sh`
3. Upload `data/sft/train.jsonl` to pod
4. Run training command (see §3.4)
5. Upload adapter to HF Hub (see RUNPOD_SETUP_NOTES.md)

### 6.2 Checkpoints and Resumption

Training saves one checkpoint per epoch in `--output-dir`:

```
models/pokesage-lora/
├── checkpoint-125/        # Epoch 1 final
├── checkpoint-250/        # Epoch 2 final (best)
└── checkpoint-375/        # Epoch 3 final
```

To resume from a checkpoint, edit `train_sft.py` to pass `resume_from_checkpoint`:

```python
trainer.train(resume_from_checkpoint="models/pokesage-lora/checkpoint-125")
```

The best checkpoint is automatically loaded at the end if `load_best_model_at_end=True` (default).

## 7. Next Steps: RL / DPO

The SFT phase establishes a strong base model. Future phases:

1. **RLHF (Reinforcement Learning from Human Feedback)**: Collect preference pairs from RAG query logs, reward Pokémon-accurate answers via a trained reward model
2. **DPO (Direct Preference Optimization)**: Simpler alternative to RLHF; directly optimize preferences without a separate reward model

Both require:

- Preference data (human or synthetic from RAG logs)
- Reward model training or DPO loss implementation
- Longer training runs on H100

Placeholder scripts will be added to `scripts/training/` when this phase begins.

## Related Documentation

- `CLAUDE.md` — Overall project architecture and no-hallucination constraints
- `RUNPOD_SETUP_NOTES.md` — Hard-won lessons on Unsloth + Gemma 4 setup
- `src/generation/loader.py` — Model loading and LoRA adapter integration
- `src/api/app.py` — FastAPI initialization and endpoints
