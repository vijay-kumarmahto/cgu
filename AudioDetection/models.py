"""
models.py
=========
Loads Wav2Vec2 model for audio deepfake detection.
Hardware-aware dtype selection (same as image models).

Label contract:
  class 0 = Fake, class 1 = Real
  predict_audio() returns (fake_prob, real_prob)
"""

import os
import warnings
import torch
import threading
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import librosa

warnings.filterwarnings("ignore")

# ── Hardware detection ─────────────────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ── Model paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")

# ── Thread safety ──────────────────────────────────────────────────────────────
_model_lock = threading.Lock()

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"[audio/models.py] Loading Wav2Vec2 (device={DEVICE}) ...", flush=True)
_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
)
_model.eval()
if torch.cuda.is_available():
    _model = _model.to(DEVICE).to(DTYPE)
print("[audio/models.py] Wav2Vec2 ready.", flush=True)


def predict_audio(audio_path: str) -> tuple[float, float]:
    """
    Run Wav2Vec2 inference on audio file.
    Returns (fake_prob, real_prob) — floats, sum to 1.0
    Thread-safe with model lock.
    """
    with _model_lock:
        try:
            # Load audio at 16kHz (model's expected sample rate)
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Validate audio length (min 0.5s, max 30s)
            if len(audio) < 8000:  # 0.5s at 16kHz
                raise ValueError("Audio too short (< 0.5s)")
            if len(audio) > 480000:  # 30s at 16kHz
                audio = audio[:480000]  # Truncate
            
            # Extract features
            inputs = _feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.inference_mode():
                logits = _model(**inputs).logits
                probs = torch.nn.functional.softmax(logits.float(), dim=1)[0].tolist()
            
            # Model: index 0 = Fake, index 1 = Real
            return round(probs[0], 4), round(probs[1], 4)
        
        except Exception as e:
            raise RuntimeError(f"Audio inference failed: {e}")
        
        finally:
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def get_model():
    """Return model for training."""
    return _model


def get_feature_extractor():
    """Return feature extractor for training."""
    return _feature_extractor


def get_lock():
    """Return model lock for training."""
    return _model_lock
