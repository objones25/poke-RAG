# RunPod Setup: Hard-Won Lessons

Every mistake below was made live on a RunPod H100. Do not repeat them.

---

## The One Thing That Kills Every Install

**`torchao >= 0.10.0` calls `torch.utils._pytree.register_constant`, which only exists in PyTorch 2.7.0+.**

`unsloth_zoo` requires `torchao >= 0.13.0`. Every version of torchao that satisfies that constraint needs torch 2.7.0. If you have torch 2.6.0, the whole stack explodes on import.

---

## CUDA Wheel Selection: Use `nvidia-smi`, Not `nvcc`

`nvcc --version` tells you the CUDA **toolkit** installed in the container. That is irrelevant for pip wheel selection.

`nvidia-smi` tells you the **driver's** maximum CUDA version. That is what matters.

| What you see in nvidia-smi | Highest CUDA wheel you can use |
| -------------------------- | ------------------------------ |
| CUDA Version: 12.4         | cu124                          |
| CUDA Version: 12.6         | cu126                          |
| CUDA Version: 13.0         | cu128 (and lower)              |

RunPod H100 pods (as of April 2026) ship with driver 580.x → CUDA 13.0. This means:

- **cu124 torch max = 2.6.0** — NOT enough for torchao >= 0.13.0
- **cu126 torch 2.7.0 is available and works** — use this

---

## Correct Install Sequence (H100, Driver 580+)

```bash
# 1. Fix setuptools — Unsloth's pyproject.toml requires exactly 80.9.0
pip install "setuptools==80.9.0"

# 2. Fix packaging — setuptools 80.9.0 needs this to parse SPDX licenses
pip install "packaging>=24.2"

# 3. Upgrade PyTorch to 2.7.0 via cu126 (NOT cu124 — 2.7.0 doesn't exist there)
#    All three MUST be upgraded together. Mismatched versions cause import errors.
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

# 4. Install unsloth_zoo from GitHub with --no-deps
#    The PyPI version of unsloth_zoo pulls in torchao 0.17.0 which then
#    needs register_constant. Installing from GitHub with --no-deps avoids this.
pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git

# 5. Install Unsloth from GitHub main with --no-build-isolation
#    PyPI Unsloth does NOT have the Gemma 4 gradient accumulation fix.
#    --no-build-isolation uses the setuptools we just installed instead of
#    pip's isolated build env which may have a different setuptools version.
pip install "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git" \
    --no-build-isolation

# 6. Verify
python -c "from unsloth import FastModel; print('OK')"
```

---

## DO NOT INSTALL XFORMERS

**Never run `pip install xformers` or `pip install --upgrade xformers` on this stack.**

xformers 0.0.35+ requires `torch>=2.10`. Running `pip install --upgrade xformers` will
silently pull in torch 2.11.0+cu126 and replace your carefully-pinned 2.7.0. This breaks:

- unsloth_zoo (requires `torch<2.11.0`)
- torchaudio 2.7.0 (requires `torch==2.7.0` exactly)
- torchvision 0.22.0 (requires `torch==2.7.0` exactly)

**xformers is not needed.** Unsloth on H100 uses Flash Attention 2 or SDPA natively.
Do not install it.

If you accidentally upgraded xformers and broke the stack, recover with:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126
pip uninstall xformers -y
python -c "from unsloth import FastModel; print('OK')"
```

---

## Errors and What They Mean

| Error                                                                               | Cause                                                                           | Fix                                                      |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `AttributeError: module 'torch.utils._pytree' has no attribute 'register_constant'` | torchao >= 0.10.0 needs torch 2.7.0+. You have 2.6.0.                           | Upgrade torch to 2.7.0 via cu126 index                   |
| `torch 2.6.0+cu124 not found` (when trying torch==2.7.0 on cu124 index)             | 2.7.0 was never built for cu124. Max for cu124 is 2.6.0.                        | Use `--index-url https://download.pytorch.org/whl/cu126` |
| `ImportError: Cannot import packaging.licenses`                                     | setuptools >= 77 needs packaging >= 24.2                                        | `pip install "packaging>=24.2"`                          |
| `configuration error: project.license must be valid exactly by one definition`      | Unsloth's pyproject.toml uses SPDX license string; system setuptools rejects it | `pip install "setuptools==80.9.0"` first                 |
| `Please install unsloth_zoo` on Unsloth import                                      | Unsloth git install doesn't resolve unsloth_zoo automatically                   | Install unsloth_zoo from GitHub before Unsloth           |
| `torchvision requires torch==2.4.1` (or similar)                                    | Only torch was upgraded; torchvision/torchaudio still pin the old version       | Always upgrade all three together                        |
| `torch._inductor has no attribute 'config'`                                         | unsloth_zoo uses `torch._inductor.config` added in torch 2.5                    | Upgrade torch (see sequence above)                       |
| `unsloth_zoo 0.x.x requires torch<2.11.0, but torch 2.11.0 is installed`            | `pip install --upgrade xformers` pulled in torch 2.11.0                         | Reinstall torch 2.7.0 + uninstall xformers (see above)   |

---

## Why GitHub Main, Not PyPI

- PyPI Unsloth **does not** have the Gemma 4 gradient accumulation fix (landed in the GitHub main branch).
- PyPI unsloth_zoo pulls in torchao 0.17.0 unconditionally. The GitHub version with `--no-deps` does not.
- Installing from PyPI will cause silent gradient corruption or runtime errors on gemma-4-E4B-it.

---

## Key Version Constraints (April 2026)

| Package     | Required version | Notes                                    |
| ----------- | ---------------- | ---------------------------------------- |
| torch       | 2.7.0            | cu126 wheel; 2.6.0 is not enough         |
| torchvision | 0.22.0           | Must match torch                         |
| torchaudio  | 2.7.0            | Must match torch                         |
| setuptools  | 80.9.0           | Unsloth pyproject.toml build requirement |
| packaging   | >= 24.2          | Required by setuptools 80.9.0            |
| unsloth_zoo | GitHub main      | Install with --no-deps                   |
| unsloth     | GitHub main      | Install with --no-build-isolation        |

---

## Training Command (single line)

```bash
python scripts/training/train_sft.py --data data/sft/train.jsonl --output-dir /workspace/models/pokesage-lora --epochs 3 --run-name pokesage-v1
```
