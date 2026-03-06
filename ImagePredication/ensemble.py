"""
ensemble.py
===========
Cascade + Soft-Vote ensemble logic.

Strategy
--------
1. Run ViT (fast, ~0.5-1s after quantization).
2. If ViT confidence >= CONFIDENCE_THRESHOLD → return ViT result immediately.
3. If ViT is uncertain → run SigLIP too, soft-average both sets of probs.
4. Return final verdict with full breakdown.

Result dict schema
------------------
{
    "verdict":        "Fake" | "Real",
    "fake_prob":      float,   # 0.0 – 1.0
    "real_prob":      float,   # 0.0 – 1.0
    "path":           "vit_only" | "full_ensemble",
    "vit_fake":       float,
    "vit_real":       float,
    "siglip_fake":    float | None,
    "siglip_real":    float | None,
}
"""

from PIL import Image
from models import predict_vit, predict_siglip
import os

# Tune via environment variable: DEEPFAKE_THRESHOLD=0.85 python app.py
# Default 0.90 — ViT result trusted alone if its top prob >= this value.
CONFIDENCE_THRESHOLD = float(os.environ.get("DEEPFAKE_THRESHOLD", "0.90"))


def run(image: Image.Image) -> dict:
    """
    Main entry point. Pass a PIL Image, get back a result dict.
    """
    image = image.convert("RGB")

    # ── Stage 1: ViT ──────────────────────────────────────────────────────────
    vit_fake, vit_real = predict_vit(image)
    vit_confidence     = max(vit_fake, vit_real)

    if vit_confidence >= CONFIDENCE_THRESHOLD:
        # ViT is confident — skip SigLIP entirely (save 3–8s on CPU)
        verdict = "Fake" if vit_fake > vit_real else "Real"
        return {
            "verdict":     verdict,
            "fake_prob":   vit_fake,
            "real_prob":   vit_real,
            "path":        "vit_only",
            "vit_fake":    vit_fake,
            "vit_real":    vit_real,
            "siglip_fake": None,
            "siglip_real": None,
        }

    # ── Stage 2: ViT uncertain → run SigLIP + soft-average ───────────────────
    siglip_fake, siglip_real = predict_siglip(image)

    # Soft vote: equal weight average of normalized probabilities
    ensemble_fake = round((vit_fake + siglip_fake) / 2, 4)
    ensemble_real = round((vit_real + siglip_real) / 2, 4)

    verdict = "Fake" if ensemble_fake > ensemble_real else "Real"

    return {
        "verdict":     verdict,
        "fake_prob":   ensemble_fake,
        "real_prob":   ensemble_real,
        "path":        "full_ensemble",
        "vit_fake":    vit_fake,
        "vit_real":    vit_real,
        "siglip_fake": siglip_fake,
        "siglip_real": siglip_real,
    }


def run_vit_only(image: Image.Image) -> dict:
    """Standalone ViT prediction. Used in evaluate_dataset.py."""
    image            = image.convert("RGB")
    fake_prob, real_prob = predict_vit(image)
    verdict          = "Fake" if fake_prob > real_prob else "Real"
    return {"verdict": verdict, "fake_prob": fake_prob, "real_prob": real_prob}


def run_siglip_only(image: Image.Image) -> dict:
    """Standalone SigLIP prediction. Used in evaluate_dataset.py."""
    image            = image.convert("RGB")
    fake_prob, real_prob = predict_siglip(image)
    verdict          = "Fake" if fake_prob > real_prob else "Real"
    return {"verdict": verdict, "fake_prob": fake_prob, "real_prob": real_prob}
