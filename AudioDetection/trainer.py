"""
trainer.py
==========
Fine-tunes Wav2Vec2 classifier head on user corrections.
"""

import os
import shutil
import torch
import torch.nn.functional as F
import librosa
from safetensors.torch import save_file, load_file
import feedback_store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.safetensors")

# Training hyperparameters
EARLY_STOP_LOSS = 0.1
ANCHOR_WEIGHT = 0.01

# Injected from models.py
_model = None
_feature_extractor = None
_model_lock = None
_original_head = None
DEVICE = None


def set_models(model, feature_extractor, model_lock, device):
    """Inject model, feature extractor, and lock from models.py."""
    global _model, _feature_extractor, _model_lock, _original_head, DEVICE
    _model = model
    _feature_extractor = feature_extractor
    _model_lock = model_lock
    DEVICE = device
    
    # Save original classifier head weights
    _original_head = {
        k: v.clone().detach().to(DEVICE)
        for k, v in _model.classifier.state_dict().items()
    }


def _get_training_params(buffer_size: int) -> dict:
    """Adaptive training params based on buffer size."""
    if buffer_size <= 1:
        return {"lr": 0, "max_steps": 0, "min_steps": 0, "note": "insufficient data"}
    elif buffer_size <= 5:
        return {"lr": 5e-5, "max_steps": 3, "min_steps": 2, "note": ""}
    elif buffer_size <= 20:
        return {"lr": 3e-5, "max_steps": 5, "min_steps": 3, "note": ""}
    else:
        return {"lr": 1e-5, "max_steps": 8, "min_steps": 4, "note": ""}


def _set_head_only(model):
    """Freeze all layers except classifier head."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def _l2_anchor_loss(model, original_head: dict) -> torch.Tensor:
    """L2 regularization to keep weights close to original."""
    loss = torch.tensor(0.0, device=DEVICE)
    for name, param in model.classifier.named_parameters():
        if name in original_head:
            loss = loss + torch.sum((param - original_head[name]) ** 2)
    return ANCHOR_WEIGHT * loss


def _save_model_atomic(model, model_path: str):
    """Atomic save with backup."""
    backup_path = model_path.replace("model.safetensors", "model_backup.safetensors")
    
    # Backup existing
    if os.path.isfile(model_path):
        shutil.copy2(model_path, backup_path)
    
    # Save to temp
    tmp_path = model_path + ".tmp"
    save_file(model.state_dict(), tmp_path)
    
    # Atomic replace
    os.replace(tmp_path, model_path)


def _fine_tune(audio_path: str, label: str, lr: float, max_steps: int, min_steps: int) -> int:
    """Fine-tune on single audio file. Returns steps taken."""
    if max_steps == 0:
        return 0
    
    _set_head_only(_model)
    
    try:
        _model.train()
        
        trainable = [p for p in _model.parameters() if p.requires_grad]
        if not trainable:
            return 0
        
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
        
        # Load audio
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        if len(audio) > 480000:
            audio = audio[:480000]
        
        inputs = _feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        target = torch.tensor([0 if label == "Fake" else 1], dtype=torch.long).to(DEVICE)
        
        steps_done = 0
        loss_val = 0.0
        
        for step in range(max_steps):
            logits = _model(**inputs).logits
            ce_loss = F.cross_entropy(logits.float(), target)
            total_loss = ce_loss + _l2_anchor_loss(_model, _original_head)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            steps_done += 1
            loss_val = total_loss.item()
            
            if step >= min_steps - 1 and loss_val < EARLY_STOP_LOSS:
                break
        
        return steps_done
    
    finally:
        _model.eval()


def train_on_correction(audio_path: str, correct_label: str) -> dict:
    """
    Fine-tune on user correction.
    Returns {"steps": int, "correction_num": int}
    """
    if _model is None or _model_lock is None:
        return {"steps": 0, "correction_num": 0}
    
    n = feedback_store.correction_count()
    
    with _model_lock:
        buf_size = feedback_store.buffer_size()
        train_params = _get_training_params(buf_size)
        
        print(
            f"[audio/trainer] ── Correction #{n} ──  label={correct_label}  "
            f"buffer={buf_size}  lr={train_params['lr']:.0e}  "
            f"steps={train_params['min_steps']}–{train_params['max_steps']}",
            flush=True
        )
        
        steps = _fine_tune(
            audio_path=audio_path,
            label=correct_label,
            lr=train_params["lr"],
            max_steps=train_params["max_steps"],
            min_steps=train_params["min_steps"]
        )
        
        # Save model
        try:
            _save_model_atomic(_model, MODEL_PATH)
        except Exception as exc:
            print(f"[audio/trainer] ⚠️  Save error: {exc}", flush=True)
    
    print(f"[audio/trainer] ── Done ──  {steps} step(s)", flush=True)
    
    return {"steps": steps, "correction_num": n}
