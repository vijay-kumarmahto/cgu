# Ensemble Deepfake Detector

Combines **ViT** and **SigLIP** into a single cascade + soft-vote ensemble for deepfake image detection. No training required. Optimized for CPU-only machines.

---

## Strategy

1. **ViT runs first** (fast, ~0.5–1s after quantization)
2. If ViT confidence ≥ 90% → return result immediately (**fast path**)
3. If ViT is uncertain → run SigLIP, soft-average both outputs (**full ensemble**)

This gives best-case speed on easy images and best-case accuracy on hard ones.

---

## Models

| Model | Architecture | Input | Label 0 | Label 1 |
|-------|-------------|-------|---------|---------|
| ViT   | `ViTForImageClassification` | 224×224 | Real | Fake |
| SigLIP | `SiglipForImageClassification` | 224×224 | Fake | Real |

Both use dynamic INT8 quantization at load time for ~2x faster CPU inference.

---

## Project Structure

```
New-project/
├── app.py                  # Gradio web UI
├── ensemble.py             # Cascade + soft-vote logic
├── models.py               # Load & quantize both models
├── evaluate_dataset.py     # Benchmark script
├── requirements.txt
└── models/
    ├── vit/                # ViT model files
    │   ├── config.json
    │   ├── model.safetensors
    │   └── preprocessor_config.json
    └── siglip/             # SigLIP model files
        ├── config.json
        ├── model.safetensors
        └── preprocessor_config.json
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run the App

```bash
cd New-project
python app.py
```

Open `http://localhost:7860` in your browser.

---

## Run Evaluation

```bash
cd New-project
python evaluate_dataset.py
```

Dataset expected at:
```
~/Downloads/DEEPFAKE_images/Dataset/Test/
    Fake/
    Real/
```

Results saved to `results/` as CSV files with per-image breakdown.

---

## Tuning

- **Confidence threshold**: Edit `CONFIDENCE_THRESHOLD` in `ensemble.py` (default `0.90`)
  - Lower = more images go through full ensemble (slower, more accurate)
  - Higher = more images use fast path (faster, slightly less accurate on borderline cases)
- **Image limit for eval**: Edit `LIMIT` in `evaluate_dataset.py` (default `100`, set `None` for all)
