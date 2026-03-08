"""
trainer.py
==========
Fine-tunes both ViT and SigLIP classifier heads on user feedback.

Strategy (prevents overfitting AND underfitting)
-------------------------------------------------
  1. Head-only training  — backbone frozen, only classifier layer updated
  2. Adaptive LR/steps   — scales with replay buffer size (more data = more training)
  3. Early stop          — stop if loss plateaus below threshold
  4. Mini-batch          — new image + samples from replay buffer (mixed)
  5. L2 anchor           — penalty toward original weights (prevents drift)
  6. Both models updated — ViT and SigLIP always trained together

Persistence
-----------
  After each fine-tune, the updated state_dict is saved directly back to
  models/vit/model.safetensors and models/siglip/model.safetensors.
  Write is atomic: write to .tmp → os.replace() so a crash never corrupts.
  A backup (model_backup.safetensors) is kept before each overwrite so
  drift rollback can restore the prior good weights.

Note: INT8 quantization removed — torchao quantized tensors do not support
autograd. Models stay float32 throughout (inference + training).

Label conventions (same as models.py normalized output)
---------------------------------------------------------
  "Fake" → class index 1 for ViT    (ViT raw:    0=Real, 1=Fake)
  "Fake" → class index 0 for SigLIP (SigLIP raw: 0=Fake, 1=Real)

Drift detection
---------------
  Every DRIFT_CHECK_EVERY corrections, run_drift_check() is called
  automatically from train_on_correction(). It tests both models on a
  fixed small validation set (feedback/val_set/ — 20 images, 10F+10R).
  If accuracy drops > DRIFT_TOLERANCE from baseline → auto-rollback from backup.

Public API
----------
  train_on_correction(image, correct_label)
      → builds batch, fine-tunes both models, saves atomically, checks drift
      → returns {"vit_steps": int, "siglip_steps": int,
                  "drift_check": bool, "rollback": bool}

  set_models(vit_model, vit_processor, siglip_model, siglip_processor)
      → called by models.py after loading — injects live model references

  restore_from_backup()
      → loads model_backup.safetensors back into both live models
      → used by drift rollback; returns True on success
"""

import os
import shutil
import random
import threading
import sys
import hashlib
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file, load_file

# ── Path setup ─────────────────────────────────────────────────────────────────
# trainer.py is in ImageDetection/
_HERE       = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_HERE)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Simple hardware detection ──────────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import feedback_store

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
VIT_MODEL_PATH    = os.path.join(BASE_DIR, "models", "vit",    "model.safetensors")
SIGLIP_MODEL_PATH = os.path.join(BASE_DIR, "models", "siglip", "model.safetensors")
VAL_DIR           = os.path.join(BASE_DIR, "feedback", "val_set")

# ── Training config ────────────────────────────────────────────────────────────
EARLY_STOP_LOSS   = 0.05
REPLAY_BATCH_SIZE = 6
L2_LAMBDA         = 0.001
DRIFT_CHECK_EVERY = 10
DRIFT_TOLERANCE   = 0.07

# ── Model references (injected by models.py via set_models()) ──────────────────
_vit_model        = None
_vit_processor    = None
_siglip_model     = None
_siglip_processor = None
_model_lock       = None

# ── Original weights snapshot ─────────────────────────────────────────────────
_vit_original_head    = None
_siglip_original_head = None

# ── Baseline accuracy ─────────────────────────────────────────────────────────
_baseline_vit_acc    = None
_baseline_siglip_acc = None


# ── Injection ─────────────────────────────────────────────────────────────────
def set_models(vit_model, vit_processor, siglip_model, siglip_processor, model_lock) -> None:
    global _vit_model, _vit_processor, _siglip_model, _siglip_processor, _model_lock
    global _vit_original_head, _siglip_original_head

    _vit_model        = vit_model
    _vit_processor    = vit_processor
    _siglip_model     = siglip_model
    _siglip_processor = siglip_processor
    _model_lock       = model_lock

    def _snapshot_head(model):
        return {
            name: param.data.float().clone().detach()
            for name, param in model.named_parameters()
            if "classifier" in name
        }

    _vit_original_head    = _snapshot_head(_vit_model)
    _siglip_original_head = _snapshot_head(_siglip_model)

    print("[trainer] Models registered. Original head weights snapshotted for L2 anchor.", flush=True)


# ── Label → class index ────────────────────────────────────────────────────────
def _vit_label_index(label: str) -> int:
    return 1 if label == "Fake" else 0


def _siglip_label_index(label: str) -> int:
    return 0 if label == "Fake" else 1


# ── Freeze backbone ───────────────────────────────────────────────────────────
def _set_head_only(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name


# ── L2 anchor loss ─────────────────────────────────────────────────────────────
def _l2_anchor_loss(model: torch.nn.Module, original_head: dict) -> torch.Tensor:
    loss = torch.tensor(0.0, device=DEVICE)
    for name, param in model.named_parameters():
        if name in original_head:
            orig = original_head[name].to(param.device)
            loss = loss + ((param.float() - orig) ** 2).sum()
    return L2_LAMBDA * loss


# ── Adaptive training params ───────────────────────────────────────────────────
def _get_training_params(buffer_size: int) -> dict:
    """
    Scale training intensity with replay buffer size.
      0–1     → skip training (not enough data)
      2–3     → small sample; cautious
      4–9     → reasonable sample; normal
      10+     → full sample; standard
    """
    if buffer_size <= 1:
        return {"max_steps": 0, "min_steps": 0, "lr": 0,
                "note": "insufficient data — skipping training"}
    elif buffer_size <= 3:
        return {"max_steps": 3, "min_steps": 2, "lr": 7e-5,
                "note": f"small buffer ({buffer_size}) — cautious update"}
    elif buffer_size <= 9:
        return {"max_steps": 4, "min_steps": 3, "lr": 1e-4, "note": None}
    else:
        return {"max_steps": 5, "min_steps": 3, "lr": 1e-4, "note": None}


# ── Build batch ────────────────────────────────────────────────────────────────
def _build_batch(new_image: Image.Image, correct_label: str) -> list[dict]:
    # Get replay samples (may include the new image if buffer already has it)
    replay = feedback_store.get_replay_batch(REPLAY_BATCH_SIZE)
    
    # Check if new image is already in replay batch (by comparing image data)
    new_hash = hashlib.md5(new_image.tobytes()).hexdigest()
    replay_hashes = {hashlib.md5(item["image"].tobytes()).hexdigest() for item in replay}
    
    # Only add new image explicitly if it's not in replay batch
    if new_hash not in replay_hashes:
        batch = [{"image": new_image, "label": correct_label}]
        batch.extend(replay)
    else:
        batch = replay
    
    random.shuffle(batch)
    return batch


# ── Atomic model save with backup ─────────────────────────────────────────────
def _save_model_atomic(model: torch.nn.Module, model_path: str, model_name: str) -> None:
    """
    1. Backup current model_path → model_backup.safetensors
    2. Write new weights → model_path.tmp
    3. os.replace(tmp → model_path)  ← atomic on Linux
    On failure: cleanup .tmp, re-raise.
    """
    backup_path = model_path.replace("model.safetensors", "model_backup.safetensors")
    tmp_path    = model_path + ".tmp"

    try:
        if os.path.isfile(model_path):
            shutil.copy2(model_path, backup_path)

        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        save_file(state, tmp_path)
        os.replace(tmp_path, model_path)

        print(f"[trainer] {model_name}: saved → {os.path.basename(model_path)}", flush=True)

    except Exception as exc:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise RuntimeError(f"[trainer] {model_name}: save failed — {exc}") from exc


# ── Restore from backup ────────────────────────────────────────────────────────
def restore_from_backup() -> bool:
    """
    Load model_backup.safetensors into both live models.
    Called automatically on drift detection.
    Returns True if both restored successfully.
    """
    if _vit_model is None or _siglip_model is None:
        print("[trainer] Models not set — cannot restore.", flush=True)
        return False

    success = True
    pairs = [
        (_vit_model,    VIT_MODEL_PATH,    "vit"),
        (_siglip_model, SIGLIP_MODEL_PATH, "siglip"),
    ]

    for model, model_path, name in pairs:
        backup_path = model_path.replace("model.safetensors", "model_backup.safetensors")
        if not os.path.isfile(backup_path):
            print(f"[trainer] {name}: no backup at {backup_path}", flush=True)
            success = False
            continue
        try:
            state = load_file(backup_path, device=str(DEVICE))
            model.load_state_dict(state)
            model.eval()
            print(f"[trainer] {name}: restored from backup", flush=True)
        except Exception as exc:
            print(f"[trainer] {name}: restore failed — {exc}", flush=True)
            success = False

    return success


# ── Single model fine-tune ─────────────────────────────────────────────────────
def _fine_tune(
    model: torch.nn.Module,
    processor,
    original_head: dict,
    batch: list[dict],
    label_fn,
    model_name: str,
    lr: float,
    max_steps: int,
    min_steps: int,
) -> int:
    """Fine-tune classifier head. Returns steps taken."""
    if max_steps == 0:
        return 0
    
    _set_head_only(model)
    
    try:
        model.train()

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            print(f"[trainer] {model_name}: no trainable params — skipping.", flush=True)
            return 0

        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

        steps_done = 0
        loss_val   = 0.0

        for step in range(max_steps):
            total_loss = torch.tensor(0.0, device=DEVICE)

            for item in batch:
                img    = item["image"].convert("RGB")
                label  = item["label"]
                inputs = processor(images=img, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                target = torch.tensor([label_fn(label)], dtype=torch.long).to(DEVICE)

                logits     = model(**inputs).logits
                ce_loss    = F.cross_entropy(logits.float(), target)
                total_loss = total_loss + ce_loss

            total_loss = total_loss + _l2_anchor_loss(model, original_head)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            steps_done += 1
            loss_val    = total_loss.item()

            print(
                f"[trainer] {model_name}: step {step+1}/{max_steps}  loss={loss_val:.4f}",
                flush=True,
            )

            if step >= min_steps - 1 and loss_val < EARLY_STOP_LOSS:
                print(
                    f"[trainer] {model_name}: early stop at step {step+1} "
                    f"(loss={loss_val:.4f} < {EARLY_STOP_LOSS})",
                    flush=True,
                )
                break

        print(
            f"[trainer] {model_name}: done — {steps_done} step(s), final loss={loss_val:.4f}",
            flush=True,
        )
        return steps_done
    
    finally:
        model.eval()


# ── Drift detection ────────────────────────────────────────────────────────────
def _eval_val_set() -> tuple:
    """
    Returns (vit_accuracy, siglip_accuracy) on val_set.
    Returns (-1.0, -1.0) if val_set absent or empty.
    """
    fake_dir = os.path.join(VAL_DIR, "Fake")
    real_dir = os.path.join(VAL_DIR, "Real")
    exts     = (".jpg", ".jpeg", ".png", ".webp")

    items = []
    for d, lbl in [(fake_dir, "Fake"), (real_dir, "Real")]:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(exts):
                items.append((os.path.join(d, f), lbl))

    if not items:
        return -1.0, -1.0

    from models import predict_vit, predict_siglip

    vit_correct = siglip_correct = 0
    for img_path, gt in items:
        try:
            img         = Image.open(img_path).convert("RGB")
            vf, vr      = predict_vit(img)
            sf, sr      = predict_siglip(img)
            vit_pred    = "Fake" if vf > vr else "Real"
            siglip_pred = "Fake" if sf > sr else "Real"
            if vit_pred    == gt: vit_correct    += 1
            if siglip_pred == gt: siglip_correct += 1
        except Exception:
            continue

    n = len(items)
    return round(vit_correct / n, 4), round(siglip_correct / n, 4)


def run_drift_check() -> dict:
    """
    Compare current accuracy to baseline.
    If either model dropped > DRIFT_TOLERANCE → rollback from backup.
    """
    global _baseline_vit_acc, _baseline_siglip_acc

    vit_acc, siglip_acc = _eval_val_set()

    if vit_acc < 0:
        return {"triggered": False, "vit_acc": -1.0, "siglip_acc": -1.0}

    if _baseline_vit_acc is None:
        _baseline_vit_acc    = vit_acc
        _baseline_siglip_acc = siglip_acc
        print(
            f"[trainer] Drift baseline — ViT={vit_acc:.2%}  SigLIP={siglip_acc:.2%}",
            flush=True,
        )
        return {"triggered": False, "vit_acc": vit_acc, "siglip_acc": siglip_acc}

    vit_drop    = _baseline_vit_acc    - vit_acc
    siglip_drop = _baseline_siglip_acc - siglip_acc

    print(
        f"[trainer] Drift check — "
        f"ViT: {vit_acc:.2%} (Δ={vit_drop:+.2%}) | "
        f"SigLIP: {siglip_acc:.2%} (Δ={siglip_drop:+.2%})",
        flush=True,
    )

    if vit_drop > DRIFT_TOLERANCE or siglip_drop > DRIFT_TOLERANCE:
        print("[trainer] ⚠️  Drift detected — rolling back from backup.", flush=True)
        restore_from_backup()
        return {"triggered": True, "vit_acc": vit_acc, "siglip_acc": siglip_acc}

    return {"triggered": False, "vit_acc": vit_acc, "siglip_acc": siglip_acc}


# ── Main public function ───────────────────────────────────────────────────────
def train_on_correction(image: Image.Image, correct_label: str) -> dict:
    """
    Full fine-tune pipeline triggered by one user correction.

    Parameters
    ----------
    image         : PIL image that was predicted wrongly
    correct_label : the true label — already flipped by feedback_store

    Returns
    -------
    {"vit_steps": int, "siglip_steps": int, "drift_check": bool, "rollback": bool}
    """
    if _vit_model is None or _siglip_model is None or _model_lock is None:
        print("[trainer] Models not injected — call set_models() first.", flush=True)
        return {"vit_steps": 0, "siglip_steps": 0, "drift_check": False, "rollback": False}

    with _model_lock:
        # Buffer size AFTER this correction was already added by record_correction
        buf_size     = feedback_store.buffer_size()
        train_params = _get_training_params(buf_size)
        n            = feedback_store.correction_count()
        note_str     = f"  [{train_params['note']}]" if train_params["note"] else ""

        print(
            f"[trainer] ── Correction #{n} ──  label={correct_label}  "
            f"buffer={buf_size}  lr={train_params['lr']:.0e}  "
            f"steps={train_params['min_steps']}–{train_params['max_steps']}"
            f"{note_str}",
            flush=True,
        )

        batch = _build_batch(image, correct_label)
        print(f"[trainer] Batch: {len(batch)} image(s)", flush=True)

        vit_steps = _fine_tune(
            model         = _vit_model,
            processor     = _vit_processor,
            original_head = _vit_original_head,
            batch         = batch,
            label_fn      = _vit_label_index,
            model_name    = "vit",
            lr            = train_params["lr"],
            max_steps     = train_params["max_steps"],
            min_steps     = train_params["min_steps"],
        )

        siglip_steps = _fine_tune(
            model         = _siglip_model,
            processor     = _siglip_processor,
            original_head = _siglip_original_head,
            batch         = batch,
            label_fn      = _siglip_label_index,
            model_name    = "siglip",
            lr            = train_params["lr"],
            max_steps     = train_params["max_steps"],
            min_steps     = train_params["min_steps"],
        )

        # Atomic save — backup existing, write .tmp, os.replace
        try:
            _save_model_atomic(_vit_model,    VIT_MODEL_PATH,    "vit")
            _save_model_atomic(_siglip_model, SIGLIP_MODEL_PATH, "siglip")
        except RuntimeError as exc:
            print(f"[trainer] ⚠️  Save error: {exc}", flush=True)

    # Drift check every N corrections (OUTSIDE lock to avoid deadlock)
    drift_result = {"triggered": False, "vit_acc": -1.0, "siglip_acc": -1.0}
    ran_drift    = False

    if n % DRIFT_CHECK_EVERY == 0:
        ran_drift    = True
        drift_result = run_drift_check()

    print(
        f"[trainer] ── Done ──  ViT: {vit_steps} step(s)  SigLIP: {siglip_steps} step(s)",
        flush=True,
    )

    return {
        "vit_steps":    vit_steps,
        "siglip_steps": siglip_steps,
        "drift_check":  ran_drift,
        "rollback":     drift_result["triggered"],
        "correction_num": n,
    }
