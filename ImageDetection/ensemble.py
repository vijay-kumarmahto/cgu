"""
ensemble.py
===========
Soft-Vote ensemble logic.

Strategy
--------
Always runs both ViT and SigLIP models and averages their probabilities.
This maximizes accuracy by leveraging both models' strengths and catching
confident errors from either model.

Result dict schema
------------------
{
    "verdict":        "Fake" | "Real",
    "fake_prob":      float,   # 0.0 – 1.0
    "real_prob":      float,   # 0.0 – 1.0
    "path":           "full_ensemble",
    "vit_fake":       float,
    "vit_real":       float,
    "siglip_fake":    float,
    "siglip_real":    float,
}
"""

from PIL import Image
from models import predict_vit, predict_siglip


def run(image: Image.Image) -> dict:
    """
    Main entry point. Pass a PIL Image, get back a result dict.
    Always runs both ViT and SigLIP for maximum accuracy.
    """
    image = image.convert("RGB")

    # Run both models
    vit_fake, vit_real = predict_vit(image)
    siglip_fake, siglip_real = predict_siglip(image)

    # Soft vote: equal weight average of normalized probabilities
    ensemble_fake = (vit_fake + siglip_fake) / 2
    ensemble_real = (vit_real + siglip_real) / 2
    
    # Normalize to ensure sum = 1.0 after any floating point errors
    total = ensemble_fake + ensemble_real
    ensemble_fake = round(ensemble_fake / total, 4)
    ensemble_real = round(ensemble_real / total, 4)

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
