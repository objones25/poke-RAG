#!/usr/bin/env bash
# RunPod environment setup for Gemma-4 QLoRA SFT.
#
# READ RUNPOD_SETUP_NOTES.md BEFORE EDITING THIS FILE.
#
# Key facts learned the hard way:
#   - torchao >= 0.13.0 needs torch 2.7.0+ (register_constant in pytree)
#   - torch 2.7.0 does NOT exist for cu124 — use cu126 index instead
#   - NVIDIA driver CUDA version (nvidia-smi) governs wheel selection, not nvcc
#   - RunPod H100 driver 580+ supports CUDA 13.0 → cu126 wheels work
#   - setuptools must be 80.9.0 and packaging >= 24.2 before building Unsloth
#   - torch/torchvision/torchaudio must always be upgraded together
#   - Unsloth must be installed from GitHub main (PyPI lacks Gemma 4 fix)
#
# Usage:
#   bash scripts/training/runpod_setup.sh
#
# Verified working on:
#   - H100 80GB HBM3 (sm_90) + CUDA toolkit 12.4 + driver 580 (CUDA 13.0)

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
info "CUDA toolkit version: ${CUDA_VERSION}"

if [[ "$CUDA_MAJOR" -lt 12 ]]; then
    die "CUDA ${CUDA_VERSION} detected. Unsloth Ampere builds require CUDA >= 12.x."
fi

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

info "GPU passes compute capability check (>= 8.0 required for bfloat16 + efficient CUDA kernels)"

# ---------------------------------------------------------------------------
# 3. Upgrade pip + setuptools + packaging
# ---------------------------------------------------------------------------
info "Upgrading pip …"
python3 -m pip install --upgrade pip --quiet

# Unsloth pyproject.toml requires setuptools==80.9.0 for its build backend.
# setuptools 80.9.0 also requires packaging>=24.2 to parse SPDX license strings.
# These must be installed before Unsloth — order matters.
info "Pinning setuptools==80.9.0 and packaging>=24.2 (required by Unsloth build) …"
pip install "setuptools==80.9.0" "packaging>=24.2" --quiet

# ---------------------------------------------------------------------------
# 4. Upgrade PyTorch to 2.7.0 via cu126
# ---------------------------------------------------------------------------
# CRITICAL: torch 2.7.0 does NOT exist on the cu124 index.
# torchao >= 0.13.0 (required by unsloth_zoo) calls
# torch.utils._pytree.register_constant which was added in torch 2.7.0.
# The H100 driver (580+) supports CUDA 13.0, so cu126 wheels are compatible
# even though the installed toolkit reports 12.4.
# torch/torchvision/torchaudio MUST be upgraded together — mismatched versions
# cause import errors.
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "0.0.0")
TORCH_MINOR=$(echo "$TORCH_VERSION" | cut -d. -f2)
if [[ "$(echo "$TORCH_VERSION" | cut -d. -f1)" -lt 2 ]] || \
   [[ "$(echo "$TORCH_VERSION" | cut -d. -f1)" -eq 2 && "$TORCH_MINOR" -lt 7 ]]; then
    warn "PyTorch ${TORCH_VERSION} detected — upgrading torch + torchvision + torchaudio to 2.7.0 (cu126) …"
    pip install \
        torch==2.7.0 \
        torchvision==0.22.0 \
        torchaudio==2.7.0 \
        --index-url "https://download.pytorch.org/whl/cu126" \
        --quiet
else
    info "PyTorch ${TORCH_VERSION} OK (>= 2.7.0)"
fi

# ---------------------------------------------------------------------------
# 5. Install unsloth_zoo from GitHub (no-deps avoids torchao conflict)
# ---------------------------------------------------------------------------
# The PyPI unsloth_zoo pulls in torchao 0.17.0 unconditionally. Installing
# from GitHub with --no-deps avoids this and lets the existing torchao
# (satisfied transitively) stay put.
info "Installing unsloth_zoo from GitHub (--no-deps to avoid torchao conflict) …"
pip install --no-deps "git+https://github.com/unslothai/unsloth-zoo.git" --quiet

# ---------------------------------------------------------------------------
# 6. Install Unsloth from GitHub main
# ---------------------------------------------------------------------------
# PyPI Unsloth lacks the Gemma 4 gradient accumulation fix.
# --no-build-isolation uses the setuptools==80.9.0 we pinned above instead of
# pip's isolated build environment which may ship a different setuptools.
info "Installing Unsloth from GitHub main (required for Gemma 4 support) …"
pip install \
    "unsloth[cu126-torch270] @ git+https://github.com/unslothai/unsloth.git" \
    --no-build-isolation \
    --quiet

# ---------------------------------------------------------------------------
# 7. Core training deps
# ---------------------------------------------------------------------------
# DO NOT install xformers here. xformers 0.0.35+ requires torch>=2.10, so
# `pip install --upgrade xformers` will silently pull in torch 2.11.0 and
# break unsloth_zoo (requires torch<2.11.0). H100 uses Flash Attention 2 /
# SDPA natively — xformers is not needed.
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
# 8. Smoke test — confirm Unsloth imports and recognises Gemma 4
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

try:
    from unsloth.models._utils import SUPPORTED_MODELS  # type: ignore[import]
    gemma4_supported = any("gemma-4" in m.lower() for m in SUPPORTED_MODELS)
    if gemma4_supported:
        print("  ✓ Gemma 4 found in Unsloth SUPPORTED_MODELS")
    else:
        print("  ⚠  Could not verify Gemma 4 in SUPPORTED_MODELS (may still work)")
except Exception:
    print("  ⚠  Could not inspect SUPPORTED_MODELS (non-fatal)")

try:
    import bitsandbytes as bnb
    print(f"  ✓ bitsandbytes {bnb.__version__}")
except ImportError as e:
    print(f"  ✗ bitsandbytes import failed: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from trl import SFTConfig, SFTTrainer  # noqa: F401
    print("  ✓ trl SFTConfig + SFTTrainer available")
except ImportError as e:
    print(f"  ✗ trl import failed: {e}", file=sys.stderr)
    sys.exit(1)

print("Smoke test passed.")
PYEOF

# ---------------------------------------------------------------------------
# 9. Print usage hint
# ---------------------------------------------------------------------------
info "Setup complete. To start training:"
echo ""
echo "  python scripts/training/train_sft.py \\"
echo "      --data data/sft/train.jsonl \\"
echo "      --output-dir /workspace/models/pokesage-lora \\"
echo "      --epochs 3 \\"
echo "      --run-name pokesage-v1"
echo ""
echo "Set WANDB_API_KEY before training if you want W&B tracking:"
echo "  export WANDB_API_KEY=<your-key>"
