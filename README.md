# Ensemble Deepfake Detector

Multi-modal deepfake detection system supporting **images**, **videos**, and **audio**. Features intelligent cascade ensemble, continuous learning from user feedback, and drift protection.

---

## Features

### 📷 Image Detection
- **Cascade Ensemble**: ViT + SigLIP with smart routing
- **Fast Path**: ViT-only for confident predictions (~0.5-1s)
- **Full Ensemble**: Soft-vote averaging for uncertain cases
- **Continuous Learning**: User feedback triggers automatic fine-tuning
- **Drift Protection**: Auto-rollback if accuracy degrades

### 🎥 Video Detection
- **GenConViT Ensemble**: ED + VAE models
- **Frame Sampling**: Analyzes 10 uniformly sampled frames
- **Hybrid Architecture**: ConvNeXt + Swin Transformer

### 🎤 Audio Detection
- **Wav2Vec2**: Fine-tuned for audio deepfake detection
- **Multi-format**: WAV, MP3, FLAC, OGG, M4A
- **Real-time Analysis**: Fast inference with confidence scores

---

## Image Detection Strategy

1. **ViT runs first** (fast, ~0.5–1s after quantization)
2. If ViT confidence ≥ 90% → return result immediately (**fast path**)
3. If ViT is uncertain → run SigLIP, soft-average both outputs (**full ensemble**)

This gives best-case speed on easy images and best-case accuracy on hard ones.

---

## Models

### Image Models
| Model | Architecture | Input | Label 0 | Label 1 |
|-------|-------------|-------|---------|---------|
| ViT   | `ViTForImageClassification` | 224×224 | Real | Fake |
| SigLIP | `SiglipForImageClassification` | 224×224 | Fake | Real |

### Video Models
| Model | Architecture | Input | Output |
|-------|-------------|-------|--------|
| GenConViT-ED | ConvNeXt + Swin | 224×224 frames | Fake probability |
| GenConViT-VAE | ConvNeXt + Swin | 224×224 frames | Fake probability |

### Audio Model
| Model | Architecture | Input | Label 0 | Label 1 |
|-------|-------------|-------|---------|---------|
| Wav2Vec2 | `Wav2Vec2ForSequenceClassification` | 16kHz audio | Fake | Real |

---

## Project Structure

```
DeepFake_Predication/
├── start.py                # Entry point with pre-flight checks
├── app.py                  # Gradio web UI (3 tabs)
├── setup.sh                # Automated setup script
├── run.sh                  # Quick launch script
├── requirements.txt        # Dependencies
│
├── ImageDetection/
│   ├── models.py           # ViT + SigLIP loaders
│   ├── ensemble.py         # Cascade logic
│   ├── trainer.py          # Fine-tuning engine
│   ├── feedback_store.py   # Replay buffer + CSV logging
│   ├── evaluate_dataset.py
│   └── models/
│       ├── vit/            # ViT weights
│       └── siglip/         # SigLIP weights
│
├── VideoDetection/
│   ├── models.py           # GenConViT models
│   ├── inference.py        # Frame extraction + prediction
│   ├── evaluate_dataset.py
│   └── weights/
│       ├── genconvit_ed_inference.pth
│       └── genconvit_vae_inference.pth
│
└── AudioDetection/
    ├── models.py           # Wav2Vec2 loader
    ├── inference.py        # Audio analysis
    ├── evaluate_dataset.py
    └── models/
        ├── config.json
        ├── model.safetensors
        └── preprocessor_config.json
```

---

## Setup

```bash
./setup.sh  # Creates venv, installs deps, downloads all models
```

This will:
1. Create virtual environment
2. Install all dependencies (including librosa for audio)
3. Download image models (ViT, SigLIP)
4. Download video models (GenConViT-ED, GenConViT-VAE)
5. Download audio model (Wav2Vec2)

---

## Run the App

```bash
./run.sh
# OR
source .venv/bin/activate
python start.py
```

Open `http://localhost:7860` in your browser.

**Three tabs available:**
- 📷 **Image**: Upload images for deepfake detection
- 🎥 **Video**: Upload videos for frame-by-frame analysis
- 🎤 **Audio**: Upload audio files for voice deepfake detection

---

## Run Evaluation

### Image Evaluation
```bash
cd ImageDetection
python evaluate_dataset.py
```
Dataset expected at: `~/Downloads/DEEPFAKE_images/Dataset/Test/`

### Video Evaluation
```bash
cd VideoDetection
python evaluate_dataset.py
```
Dataset expected at: `~/Downloads/DEEPFAKE_videos/Dataset/Test/`

### Audio Evaluation
```bash
cd AudioDetection
python evaluate_dataset.py
```
Dataset expected at: `~/Downloads/DEEPFAKE_audio/Dataset/Test/`

All results saved to respective `results/` directories as CSV files.

---

## Tuning

### Image Detection
- **Confidence threshold**: `DEEPFAKE_THRESHOLD=0.85 python start.py` (default `0.90`)
  - Lower = more images go through full ensemble (slower, more accurate)
  - Higher = more images use fast path (faster, slightly less accurate)
- **Image limit for eval**: Edit `LIMIT` in `ImageDetection/evaluate_dataset.py`

### Training Parameters
- **Replay buffer size**: `MAX_PER_CLASS` in `ImageDetection/feedback_store.py` (default 50 per class)
- **Drift tolerance**: `DRIFT_TOLERANCE` in `ImageDetection/trainer.py` (default 0.07)
- **Drift check frequency**: `DRIFT_CHECK_EVERY` in `ImageDetection/trainer.py` (default 10)

---

## Status

| Feature | Status | Notes |
|---------|--------|-------|
| Image Detection | ✅ Complete | Full ensemble + continuous learning |
| Image Fine-tuning | ✅ Complete | Automatic on user feedback |
| Video Detection | ✅ Complete | GenConViT ensemble |
| Video Fine-tuning | ⚠️ Planned | UI ready, training not implemented |
| Audio Detection | ✅ Complete | Wav2Vec2 inference |
| Audio Fine-tuning | ⚠️ Planned | UI ready, training not implemented |
