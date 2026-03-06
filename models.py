"""
models.py
=========
Loads both ViT and SigLIP models once at startup.
Applies hardware-aware optimizations via hardware.py:
  - CUDA GPU  → float16/bfloat16, model on GPU
  - CPU AVX2  → INT8 quantization via torchao, P-core pinned threads
  - CPU basic → INT8 quantization, default threads

Label contracts (normalized):
  Both predict() functions always return (fake_prob, real_prob)
  where fake_prob + real_prob == 1.0

ViT  raw:   class 0 = Real, class 1 = Fake  → normalize to (fake=probs[1], real=probs[0])
SigLIP raw: class 0 = Fake, class 1 = Real  → natural  (fake=probs[0], real=probs[1])

Public API
----------
  predict_vit(image)    → (fake_prob, real_prob)
  predict_siglip(image) → (fake_prob, real_prob)
  reload_models()       → reload + re-quantize both models from disk
                          Called by feedback.py after a fine-tune cycle.
"""

import os
import warnings
import threading
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    SiglipForImageClassification,
)

import logging
logging.getLogger("torchao").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*fast processor.*")

import hardware

# Suppress torchao C-level .so noise. If app.py / start.py already called this,
# it's a no-op (torchao is cached in sys.modules and _torchao_suppressed=True).
hardware.suppress_torchao_noise()
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig

# ── Hardware config (must run before model load) ───────────────────────────────
hardware.configure()
# Note: hardware.summary() is NOT called here — entry points (app.py / evaluate_dataset.py)
# call it once explicitly, avoiding duplicate output.

# ── Model paths ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VIT_PATH    = os.path.join(BASE_DIR, "models", "vit")
SIGLIP_PATH = os.path.join(BASE_DIR, "models", "siglip")


def _load_and_optimize(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply the right optimization strategy based on detected hardware profile.
      CUDA     → move to GPU + cast to float16/bfloat16 (Tensor Cores)
      CPU/OV   → INT8 dynamic quantization (torchao, uses AVX2/AVX_VNNI)
    These are mutually exclusive paths — never mix INT8 + GPU dtype cast.
    """
    model.eval()
    if hardware.PROFILE == "cuda":
        model = hardware.move_model(model)          # GPU + dtype cast
    else:
        quantize_(model, Int8DynamicActivationInt8WeightConfig())  # CPU INT8
    return model


# ── Load ViT ───────────────────────────────────────────────────────────────────
print(f"[models.py] Loading ViT  (profile={hardware.PROFILE}) ...", flush=True)
_vit_processor = AutoImageProcessor.from_pretrained(VIT_PATH, use_fast=False)
# Only pass torch_dtype for CUDA — on CPU load as float32, quantize separately
_vit_model = ViTForImageClassification.from_pretrained(
    VIT_PATH,
    torch_dtype=hardware.DTYPE if hardware.PROFILE == "cuda" else torch.float32,
)
_vit_model = _load_and_optimize(_vit_model)
print("[models.py] ViT ready.", flush=True)

# ── Load SigLIP ────────────────────────────────────────────────────────────────
print(f"[models.py] Loading SigLIP (profile={hardware.PROFILE}) ...", flush=True)
_siglip_processor = AutoImageProcessor.from_pretrained(SIGLIP_PATH, use_fast=False)
_siglip_model = SiglipForImageClassification.from_pretrained(
    SIGLIP_PATH,
    torch_dtype=hardware.DTYPE if hardware.PROFILE == "cuda" else torch.float32,
)
_siglip_model = _load_and_optimize(_siglip_model)
print("[models.py] SigLIP ready.", flush=True)

# ── Inference lock (held during predict; reload waits for it) ──────────────────
_inference_lock = threading.Lock()


# ── Public inference functions ─────────────────────────────────────────────────

def predict_vit(image: Image.Image) -> tuple[float, float]:
    """
    Run ViT inference.
    Returns (fake_prob, real_prob) — floats, sum to 1.0
    """
    img    = image.convert("RGB")
    inputs = _vit_processor(images=img, return_tensors="pt")

    # Move inputs to correct device (GPU if CUDA, else CPU)
    inputs = {k: v.to(hardware.DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = _vit_model(**inputs).logits          # [1, 2]
        probs  = torch.nn.functional.softmax(logits.float(), dim=1).squeeze().tolist()

    # ViT: index 0 = Real, index 1 = Fake  → normalize to (fake, real)
    real_prob = round(probs[0], 4)
    fake_prob = round(probs[1], 4)
    return fake_prob, real_prob


def predict_siglip(image: Image.Image) -> tuple[float, float]:
    """
    Run SigLIP inference.
    Returns (fake_prob, real_prob) — floats, sum to 1.0
    """
    img    = image.convert("RGB")
    inputs = _siglip_processor(images=img, return_tensors="pt")

    # Move inputs to correct device
    inputs = {k: v.to(hardware.DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = _siglip_model(**inputs).logits       # [1, 2]
        probs  = torch.nn.functional.softmax(logits.float(), dim=1).squeeze().tolist()

    # SigLIP: index 0 = Fake, index 1 = Real  → already normalized
    fake_prob = round(probs[0], 4)
    real_prob = round(probs[1], 4)
    return fake_prob, real_prob
