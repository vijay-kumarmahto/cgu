"""
inference.py
============
Audio deepfake detection inference.
"""

import os
import sys
import importlib.util

# Import audio models using absolute path to avoid conflicts
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_models_path = os.path.join(_THIS_DIR, 'models.py')
_spec = importlib.util.spec_from_file_location('audio_models', _models_path)
_audio_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audio_models)
predict_audio = _audio_models.predict_audio

# Inject models into trainer
_trainer_path = os.path.join(_THIS_DIR, 'trainer.py')
_spec = importlib.util.spec_from_file_location('audio_trainer', _trainer_path)
_audio_trainer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audio_trainer)
_audio_trainer.set_models(
    _audio_models.get_model(),
    _audio_models.get_feature_extractor(),
    _audio_models.get_lock(),
    _audio_models.DEVICE
)

# Export for app.py to use
__all__ = ['analyze_audio', '_audio_trainer']


def analyze_audio(audio_path: str) -> dict:
    """
    Analyze audio file for deepfake detection.
    
    Returns
    -------
    {
        "verdict": "Fake" | "Real",
        "fake_prob": float,
        "real_prob": float,
        "confidence": float,
    }
    """
    fake_prob, real_prob = predict_audio(audio_path)
    verdict = "Fake" if fake_prob > real_prob else "Real"
    confidence = max(fake_prob, real_prob)
    
    return {
        "verdict": verdict,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "confidence": confidence,
    }
