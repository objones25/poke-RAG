#!/usr/bin/env bash
# RunPod environment setup for Gemma-4 QLoRA SFT.
#
# What this does:
#   1. Checks CUDA version and GPU compute capability (must be >= 8.0)
#   2. Installs Unsloth from GitHub main (NOT PyPI — PyPI build lacks Gemma 4 fix)
#   3. Installs all other training deps at pinned minimum versions
#   4. Runs a smoke test to confirm Unsloth loads without error
#
# Usage:
#   bash scripts/training/runpod_setup.sh
#
# Verified working on:
#   - A100 80GB SXM4 (sm_80) + CUDA 12.1 / 12.4
#   - H100 80GB SXM5 (sm_90) + CUDA 12.4

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()   { error "$*"; exit 1; }

# ---------------------------------------------------------------------------
# 1. CUDA version check
# ---------------------------------------------------------------------------
info "Checking CUDA version …"

if ! command -v nvcc &>/dev/null; then
    die "nvcc not found. Ensure CUDA toolkit is installed on this pod."
fi

CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
info "CUDA version: ${CUDA_VERSION}"

if [[ "$CUDA_MAJOR" -lt 12 ]]; then
    die "CUDA ${CUDA_VERSION} detected. Unsloth Ampere builds require CUDA >= 12.x."
fi

# Map CUDA minor → wheel suffix used by Unsloth extras
if [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -le 1 ]]; then
    UNSLOTH_CUDA="cu121-ampere"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -le 3 ]]; then
    UNSLOTH_CUDA="cu123-ampere"
else
    UNSLOTH_CUDA="cu124-ampere"   # 12.4 and newer
fi
info "Unsloth CUDA variant: ${UNSLOTH_CUDA}"

# ---------------------------------------------------------------------------
# 2. GPU compute capability check
# ---------------------------------------------------------------------------
info "Checking GPU compute capability …"

if ! command -v python3 &>/dev/null; then
    die "python3 not found."
fi

COMPUTE_CAP=$(python3 - <<'PYEOF'
import sys
try:
    import torch
    props = torch.cuda.get_device_properties(0)
    cap = props.major + props.minor / 10
    print(f"{cap:.1f}")
    print(f"  GPU: {props.name}", file=sys.stderr)
    print(f"  VRAM: {props.total_memory / (1024**3):.1f} GB", file=sys.stderr)
except Exception as e:
    print(f"Could not detect GPU: {e}", file=sys.stderr)
    print("0.0")
PYEOF
)

info "Compute capability: ${COMPUTE_CAP}"
COMPUTE_INT=$(python3 -c "print(int(float('${COMPUTE_CAP}') * 10))")

if [[ "$COMPUTE_INT" -lt 80 ]]; then
    die "GPU compute capability ${COMPUTE_CAP} < 8.0. \
Need A100 (sm_80) or H100 (sm_90) for bfloat16 and efficient QLoRA."
fi

# bfloat16 sanity
info "GPU passes compute capability check (>= 8.0 required for bfloat16 + efficient CUDA kernels)"

# ---------------------------------------------------------------------------
# 3. Upgrade pip + PyTorch
# ---------------------------------------------------------------------------
info "Upgrading pip …"
python3 -m pip install --upgrade pip --quiet

# unsloth_zoo calls torch._inductor.config which was added in PyTorch 2.5.
# RunPod base images sometimes ship 2.4.x — upgrade to avoid AttributeError.
# torch/torchvision/torchaudio must be upgraded together — mismatched versions
# cause import errors. Pinned to 2.6.0/0.21.0/2.6.0 (latest cu124 release per
# https://pytorch.org/get-started/previous-versions/).
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "0.0.0")
TORCH_MINOR=$(echo "$TORCH_VERSION" | cut -d. -f2)
if [[ "$(echo "$TORCH_VERSION" | cut -d. -f1)" -lt 2 ]] || [[ "$(echo "$TORCH_VERSION" | cut -d. -f1)" -eq 2 && "$TORCH_MINOR" -lt 5 ]]; then
    warn "PyTorch ${TORCH_VERSION} detected — upgrading torch + torchvision + torchaudio to 2.6.0 …"
    pip install \
        torch==2.6.0 \
        torchvision==0.21.0 \
        torchaudio==2.6.0 \
        --index-url "https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}" \
        --quiet
else
    info "PyTorch ${TORCH_VERSION} OK (>= 2.5.1)"
fi

# ---------------------------------------------------------------------------
# 4. Install Unsloth from GitHub main
# ---------------------------------------------------------------------------
# PyPI Unsloth does NOT have the Gemma 4 gradient accumulation fix (landed in
# v0.1.36-beta, only available from GitHub main as of April 2026).
# Installing from PyPI will cause silent gradient corruption or "not supported"
# errors when fine-tuning gemma-4-E4B-it (see Unsloth issue #4942).
#
# unsloth_zoo must be installed first — it is a required dep that pip does not
# resolve automatically when installing Unsloth from a git URL.
# The cu1xx-ampere extras are only defined on PyPI releases, not the git HEAD,
# so we install without extras and rely on the pre-installed CUDA env.
# ---------------------------------------------------------------------------
info "Installing unsloth_zoo (required Unsloth dependency) …"
pip install unsloth_zoo --quiet

info "Installing Unsloth from GitHub main (required for Gemma 4 support) …"
pip install "git+https://github.com/unslothai/unsloth.git" --quiet

# ---------------------------------------------------------------------------
# 5. Core training deps
# ---------------------------------------------------------------------------
info "Installing training dependencies …"

pip install \
    "transformers>=4.43.0" \
    "trl>=0.12.0" \
    "peft>=0.11.0" \
    "bitsandbytes>=0.43.0" \
    "datasets>=2.19.0" \
    "accelerate>=0.29.0" \
    "wandb>=0.17.0" \
    --quiet

# ---------------------------------------------------------------------------
# 6. Smoke test — confirm Unsloth imports and recognises Gemma 4
# ---------------------------------------------------------------------------
info "Running Unsloth smoke test …"

python3 - <<'PYEOF'
import sys

try:
    from unsloth import FastModel
    print("  ✓ FastModel imported successfully")
except ImportError as e:
    print(f"  ✗ FastModel import failed: {e}", file=sys.stderr)
    sys.exit(1)

# Confirm Gemma 4 is in the supported model list without downloading weights
try:
    from unsloth.models._utils import SUPPORTED_MODELS  # type: ignore[import]
    gemma4_supported = any("gemma-4" in m.lower() for m in SUPPORTED_MODELS)
    if gemma4_supported:
        print("  ✓ Gemma 4 found in Unsloth SUPPORTED_MODELS")
    else:
        # Not all Unsloth versions expose this — soft warning only
        print("  ⚠  Could not verify Gemma 4 in SUPPORTED_MODELS (may still work)")
except Exception:
    print("  ⚠  Could not inspect SUPPORTED_MODELS (non-fatal)")

# Confirm bitsandbytes is functional
try:
    import bitsandbytes as bnb
    print(f"  ✓ bitsandbytes {bnb.__version__}")
except ImportError as e:
    print(f"  ✗ bitsandbytes import failed: {e}", file=sys.stderr)
    sys.exit(1)

# Confirm TRL SFTConfig exists (trl >= 0.12 renamed TrainingArguments subclass)
try:
    from trl import SFTConfig, SFTTrainer  # noqa: F401
    print("  ✓ trl SFTConfig + SFTTrainer available")
except ImportError as e:
    print(f"  ✗ trl import failed: {e}", file=sys.stderr)
    sys.exit(1)

print("Smoke test passed.")
PYEOF

# ---------------------------------------------------------------------------
# 7. Print usage hint
# ---------------------------------------------------------------------------
info "Setup complete. To start training:"
echo ""
echo "  python scripts/training/train_sft.py \\"
echo "      --data data/sft/train.jsonl \\"
echo "      --output-dir models/pokesage-lora \\"
echo "      --epochs 3 \\"
echo "      --run-name pokesage-v1"
echo ""
echo "Set WANDB_API_KEY before training if you want W&B tracking:"
echo "  export WANDB_API_KEY=<your-key>"
