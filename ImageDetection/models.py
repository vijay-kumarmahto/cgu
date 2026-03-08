"""
models.py
=========
Loads both ViT and SigLIP models once at startup.
Applies hardware-aware dtype selection:
  - CUDA GPU  → float16/bfloat16, model on GPU
  - CPU       → float32 (quantization removed — torchao INT8 does not support autograd)

At startup: cleans up any orphaned .tmp files left from a crashed atomic save.

Label contracts (normalized):
  Both predict() functions always return (fake_prob, real_prob)
  where fake_prob + real_prob ≈ 1.0

ViT  raw:   class 0 = Real, class 1 = Fake  → normalize to (fake=probs[1], real=probs[0])
SigLIP raw: class 0 = Fake, class 1 = Real  → natural   (fake=probs[0], real=probs[1])

Public API
----------
  predict_vit(image)    → (fake_prob, real_prob)
  predict_siglip(image) → (fake_prob, real_prob)
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

warnings.filterwarnings("ignore", message=".*fast processor.*")

# ── Simple hardware detection ──────────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ── Model paths ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VIT_PATH    = os.path.join(BASE_DIR, "models", "vit")
SIGLIP_PATH = os.path.join(BASE_DIR, "models", "siglip")

# ── Startup cleanup: remove orphaned .tmp files from aborted atomic saves ──────
for _tmp in [
    os.path.join(VIT_PATH,    "model.safetensors.tmp"),
    os.path.join(SIGLIP_PATH, "model.safetensors.tmp"),
]:
    if os.path.isfile(_tmp):
        try:
            os.remove(_tmp)
            print(f"[models.py] Cleaned up orphaned temp file: {_tmp}", flush=True)
        except OSError as _e:
            print(f"[models.py] Warning: could not remove {_tmp}: {_e}", flush=True)


def _load_and_optimize(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply the right optimization strategy based on detected hardware.
      CUDA → move to GPU + cast to float16 (Tensor Cores)
      CPU  → stay float32, just call eval()
    """
    model.eval()
    if torch.cuda.is_available():
        model = model.to(DEVICE).to(DTYPE)
    return model


# ── Load ViT ───────────────────────────────────────────────────────────────────
print(f"[models.py] Loading ViT (device={DEVICE}) ...", flush=True)
_vit_processor = AutoImageProcessor.from_pretrained(VIT_PATH, use_fast=False)
_vit_model = ViTForImageClassification.from_pretrained(
    VIT_PATH,
    torch_dtype=DTYPE,
)
_vit_model = _load_and_optimize(_vit_model)
print("[models.py] ViT ready.", flush=True)

# ── Load SigLIP ────────────────────────────────────────────────────────────────
print(f"[models.py] Loading SigLIP (device={DEVICE}) ...", flush=True)
_siglip_processor = AutoImageProcessor.from_pretrained(SIGLIP_PATH, use_fast=False)
_siglip_model = SiglipForImageClassification.from_pretrained(
    SIGLIP_PATH,
    torch_dtype=DTYPE,
)
_siglip_model = _load_and_optimize(_siglip_model)
print("[models.py] SigLIP ready.", flush=True)

# ── Shared lock for thread safety ─────────────────────────────────────────────
# Both inference and training must acquire this lock to prevent concurrent access
_model_lock = threading.Lock()

# ── Register with trainer so fine-tune can access live model references ────────
import trainer
trainer.set_models(_vit_model, _vit_processor, _siglip_model, _siglip_processor, _model_lock)


# ── Public inference functions ─────────────────────────────────────────────────

def predict_vit(image: Image.Image) -> tuple[float, float]:
    """
    Run ViT inference.
    Returns (fake_prob, real_prob) — floats, sum to 1.0
    """
    img    = image.convert("RGB")
    inputs = _vit_processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with _model_lock:
        with torch.inference_mode():
            logits = _vit_model(**inputs).logits
            probs  = torch.nn.functional.softmax(logits.float(), dim=1)
            
            # Handle batch dimension
            if probs.dim() > 1:
                probs = probs.squeeze(0)
            
            probs = probs.tolist()
            
            # Validate shape
            if not isinstance(probs, list) or len(probs) != 2:
                raise RuntimeError(f"Unexpected probs shape: {probs}")

    # ViT: index 0 = Real, index 1 = Fake
    return round(probs[1], 4), round(probs[0], 4)


def predict_siglip(image: Image.Image) -> tuple[float, float]:
    """
    Run SigLIP inference.
    Returns (fake_prob, real_prob) — floats, sum to 1.0
    """
    img    = image.convert("RGB")
    inputs = _siglip_processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with _model_lock:
        with torch.inference_mode():
            logits = _siglip_model(**inputs).logits
            probs  = torch.nn.functional.softmax(logits.float(), dim=1)
            
            # Handle batch dimension
            if probs.dim() > 1:
                probs = probs.squeeze(0)
            
            probs = probs.tolist()
            
            # Validate shape
            if not isinstance(probs, list) or len(probs) != 2:
                raise RuntimeError(f"Unexpected probs shape: {probs}")

    # SigLIP: index 0 = Fake, index 1 = Real
    return round(probs[0], 4), round(probs[1], 4)
