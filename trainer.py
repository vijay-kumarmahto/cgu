"""
trainer.py
==========
Fine-tunes both ViT and SigLIP classifier heads on user feedback.

Strategy (prevents overfitting AND underfitting)
-------------------------------------------------
  1. Head-only training  — backbone frozen, only classifier layer updated
  2. Small LR            — 1e-5 (nudge, not a jump)
  3. 3–5 steps           — stop early if loss plateaus
  4. Mini-batch          — new image + samples from replay buffer (mixed)
  5. L2 anchor           — penalty toward original weights (prevents drift)
  6. Both models updated — ViT and SigLIP always trained together

Note: INT8 quantization removed — torchao quantized tensors do not support
autograd. Models stay float32 throughout (inference + training).

Label conventions (same as models.py normalized output)
---------------------------------------------------------
  "Fake" → class index 1 for ViT   (ViT raw: 0=Real, 1=Fake)
  "Fake" → class index 0 for SigLIP (SigLIP raw: 0=Fake, 1=Real)

Drift detection
---------------
  Every DRIFT_CHECK_EVERY corrections, run_drift_check() is called
  automatically from train_on_correction(). It tests both models on a
  fixed small validation set (feedback/val_set/ — 20 images, 10F+10R).
  If accuracy drops > DRIFT_TOLERANCE from baseline → auto-rollback.

Public API
----------
  train_on_correction(image, correct_label)
      → builds batch, fine-tunes both models, saves, checks drift
      → returns {"vit_steps": int, "siglip_steps": int, "drift_check": bool}

  set_models(vit_model, vit_processor, siglip_model, siglip_processor)
      → called by models.py after loading — injects live model references

  restore_last_checkpoint()
      → loads last saved weights for both models from feedback/checkpoints/
"""

import os
import random
import threading
import torch
import torch.nn.functional as F
from PIL import Image

import hardware
import feedback_store

# ── Config ─────────────────────────────────────────────────────────────────────
LR                 = 1e-4    # learning rate — increased from 1e-5 for stronger correction signal
MAX_STEPS          = 5       # max gradient steps per correction
MIN_STEPS          = 3       # always do at least this many steps
EARLY_STOP_LOSS    = 0.05    # stop early if loss drops below this (well learned)
REPLAY_BATCH_SIZE  = 6       # how many replay buffer samples to mix in
L2_LAMBDA          = 0.001   # L2 anchor strength — reduced from 0.01 so corrections stick
DRIFT_CHECK_EVERY  = 10      # run drift check every N corrections
DRIFT_TOLERANCE    = 0.07    # if accuracy drops more than 7% → rollback
MAX_CHECKPOINTS    = 3       # keep only this many checkpoint files per model

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "feedback", "checkpoints")
VAL_DIR         = os.path.join(BASE_DIR, "feedback", "val_set")   # optional
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Model references (injected by models.py via set_models()) ──────────────────
_vit_model       = None
_vit_processor   = None
_siglip_model    = None
_siglip_processor = None

# ── Original weights snapshot (taken once at set_models() time) ───────────────
# Used for L2 anchor regularization — keeps fine-tuned weights close to original
_vit_original_head    = None   # clone of original ViT classifier weights
_siglip_original_head = None   # clone of original SigLIP classifier weights

# ── Baseline accuracy (set after first drift check) ───────────────────────────
_baseline_vit_acc    = None
_baseline_siglip_acc = None

# ── Thread lock — prevents two fine-tune calls overlapping ────────────────────
_train_lock = threading.Lock()


# ── Injection (called by models.py) ───────────────────────────────────────────
def set_models(vit_model, vit_processor, siglip_model, siglip_processor) -> None:
    """
    Inject live model + processor references.
    Called once by models.py after loading.
    Snapshots original classifier head weights (plain float32) for L2 anchoring.
    """
    global _vit_model, _vit_processor, _siglip_model, _siglip_processor
    global _vit_original_head, _siglip_original_head

    _vit_model        = vit_model
    _vit_processor    = vit_processor
    _siglip_model     = siglip_model
    _siglip_processor = siglip_processor

    # Models are plain float32 — snapshot classifier heads directly
    def _snapshot_head(model: torch.nn.Module) -> dict:
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
    """ViT raw labels: 0=Real, 1=Fake"""
    return 1 if label == "Fake" else 0


def _siglip_label_index(label: str) -> int:
    """SigLIP raw labels: 0=Fake, 1=Real"""
    return 0 if label == "Fake" else 1


# ── Freeze backbone, unfreeze only classifier head ────────────────────────────
def _set_head_only(model: torch.nn.Module) -> None:
    """Freeze all parameters except the final classifier layer."""
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name


# ── L2 anchor loss ─────────────────────────────────────────────────────────────
def _l2_anchor_loss(model: torch.nn.Module, original_head: dict) -> torch.Tensor:
    """
    Penalizes deviation of current classifier weights from original weights.
    Loss = L2_LAMBDA * sum(||current - original||^2) for each classifier param.
    Both sides are plain float32 — no quantization involved.
    """
    loss = torch.tensor(0.0)
    for name, param in model.named_parameters():
        if name in original_head:
            orig = original_head[name].to(param.device)
            loss = loss + ((param.float() - orig) ** 2).sum()
    return L2_LAMBDA * loss


# ── Build training batch ───────────────────────────────────────────────────────
def _build_batch(new_image: Image.Image, correct_label: str) -> list[dict]:
    """
    Combine the new correction with samples from the replay buffer.
    Returns list of {"image": PIL.Image, "label": str} dicts.
    """
    batch = [{"image": new_image, "label": correct_label}]
    replay = feedback_store.get_replay_batch(REPLAY_BATCH_SIZE)
    batch.extend(replay)
    random.shuffle(batch)   # mix order so model doesn't learn position bias
    return batch


# ── Single model fine-tune ─────────────────────────────────────────────────────
def _fine_tune(
    model: torch.nn.Module,
    processor,
    original_head: dict,
    batch: list[dict],
    label_fn,            # callable: label str → class index int
    model_name: str,
) -> int:
    """
    Fine-tune a single model's classifier head on the given batch.

    Returns the number of gradient steps actually taken.
    """
    _set_head_only(model)
    model.train()

    # Only optimize classifier parameters (all others are frozen)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        print(f"[trainer] {model_name}: no trainable params found — skipping.", flush=True)
        return 0

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)

    steps_done = 0
    for step in range(MAX_STEPS):
        total_loss = torch.tensor(0.0)

        for item in batch:
            img   = item["image"].convert("RGB")
            label = item["label"]

            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(hardware.DEVICE) for k, v in inputs.items()}
            target = torch.tensor([label_fn(label)], dtype=torch.long).to(hardware.DEVICE)

            logits = model(**inputs).logits            # [1, num_classes]
            ce_loss = F.cross_entropy(logits.float(), target)
            total_loss = total_loss + ce_loss

        # Add L2 anchor regularization
        total_loss = total_loss + _l2_anchor_loss(model, original_head)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        steps_done += 1
        loss_val = total_loss.item()

        # Early stop: loss is already very low — no point continuing
        if step >= MIN_STEPS - 1 and loss_val < EARLY_STOP_LOSS:
            print(
                f"[trainer] {model_name}: early stop at step {step+1} "
                f"(loss={loss_val:.4f} < {EARLY_STOP_LOSS})",
                flush=True,
            )
            break

    model.eval()
    print(
        f"[trainer] {model_name}: {steps_done} steps done, "
        f"final loss={loss_val:.4f}",
        flush=True,
    )
    return steps_done


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def _save_checkpoint(model: torch.nn.Module, model_name: str) -> None:
    """
    Save current model weights as a numbered checkpoint.
    Deletes oldest checkpoints beyond MAX_CHECKPOINTS.
    """
    # Find next checkpoint index
    existing = sorted([
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith(f"{model_name}_ckpt_") and f.endswith(".pt")
    ])
    next_idx = len(existing) + 1
    fname    = f"{model_name}_ckpt_{next_idx:04d}.pt"
    fpath    = os.path.join(CHECKPOINT_DIR, fname)

    # Save only state_dict (weights), not full model
    torch.save(model.state_dict(), fpath)
    print(f"[trainer] Saved checkpoint: {fname}", flush=True)

    # Prune old checkpoints beyond MAX_CHECKPOINTS
    all_ckpts = sorted([
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith(f"{model_name}_ckpt_") and f.endswith(".pt")
    ])
    while len(all_ckpts) > MAX_CHECKPOINTS:
        old = os.path.join(CHECKPOINT_DIR, all_ckpts.pop(0))
        os.remove(old)
        print(f"[trainer] Removed old checkpoint: {os.path.basename(old)}", flush=True)


def restore_last_checkpoint() -> bool:
    """
    Restore both models to their most recent saved checkpoints.
    Returns True if both restored successfully, False otherwise.
    """
    if _vit_model is None or _siglip_model is None:
        print("[trainer] Models not set — cannot restore.", flush=True)
        return False

    success = True
    for model, name in [(_vit_model, "vit"), (_siglip_model, "siglip")]:
        ckpts = sorted([
            f for f in os.listdir(CHECKPOINT_DIR)
            if f.startswith(f"{name}_ckpt_") and f.endswith(".pt")
        ])
        if not ckpts:
            print(f"[trainer] No checkpoint found for {name}.", flush=True)
            success = False
            continue
        path = os.path.join(CHECKPOINT_DIR, ckpts[-1])
        model.load_state_dict(torch.load(path, map_location=hardware.DEVICE))
        model.eval()
        print(f"[trainer] Restored {name} from: {ckpts[-1]}", flush=True)

    return success


# ── Drift detection ────────────────────────────────────────────────────────────
def _eval_val_set() -> tuple[float, float]:
    """
    Run both models on feedback/val_set/Fake/ and feedback/val_set/Real/.
    Returns (vit_accuracy, siglip_accuracy) as 0.0–1.0 floats.
    Returns (-1.0, -1.0) if val_set doesn't exist or has no images.
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
            img = Image.open(img_path).convert("RGB")
            vf, vr     = predict_vit(img)
            sf, sr     = predict_siglip(img)
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
    If either model dropped > DRIFT_TOLERANCE → auto-rollback both.

    Returns {"triggered": bool, "vit_acc": float, "siglip_acc": float}
    """
    global _baseline_vit_acc, _baseline_siglip_acc

    vit_acc, siglip_acc = _eval_val_set()

    if vit_acc < 0:
        # No val set — skip silently
        return {"triggered": False, "vit_acc": -1.0, "siglip_acc": -1.0}

    # Set baseline on first check
    if _baseline_vit_acc is None:
        _baseline_vit_acc    = vit_acc
        _baseline_siglip_acc = siglip_acc
        print(
            f"[trainer] Drift baseline set — "
            f"ViT={vit_acc:.2%}, SigLIP={siglip_acc:.2%}",
            flush=True,
        )
        return {"triggered": False, "vit_acc": vit_acc, "siglip_acc": siglip_acc}

    vit_drop    = _baseline_vit_acc    - vit_acc
    siglip_drop = _baseline_siglip_acc - siglip_acc

    print(
        f"[trainer] Drift check — "
        f"ViT: {vit_acc:.2%} (drop={vit_drop:.2%}) | "
        f"SigLIP: {siglip_acc:.2%} (drop={siglip_drop:.2%})",
        flush=True,
    )

    if vit_drop > DRIFT_TOLERANCE or siglip_drop > DRIFT_TOLERANCE:
        print("[trainer] ⚠️  Drift detected — rolling back both models.", flush=True)
        restore_last_checkpoint()
        return {"triggered": True, "vit_acc": vit_acc, "siglip_acc": siglip_acc}

    return {"triggered": False, "vit_acc": vit_acc, "siglip_acc": siglip_acc}


# ── Main public function ───────────────────────────────────────────────────────
def train_on_correction(image: Image.Image, correct_label: str) -> dict:
    """
    Full fine-tune pipeline triggered by one user correction.

    Parameters
    ----------
    image         : PIL image that was predicted wrongly
    correct_label : the true label ("Fake" or "Real") — already flipped by feedback_store

    Returns
    -------
    {
        "vit_steps":    int,
        "siglip_steps": int,
        "drift_check":  bool,   # True if drift check was run this round
        "rollback":     bool,   # True if rollback was triggered
    }
    """
    if _vit_model is None or _siglip_model is None:
        print("[trainer] Models not injected — call set_models() first.", flush=True)
        return {"vit_steps": 0, "siglip_steps": 0, "drift_check": False, "rollback": False}

    with _train_lock:
        # ── Build shared batch ─────────────────────────────────────────────────
        batch = _build_batch(image, correct_label)
        n     = feedback_store.correction_count()

        print(
            f"[trainer] Starting fine-tune — correction #{n}, "
            f"label={correct_label}, batch_size={len(batch)}",
            flush=True,
        )

        # ── Fine-tune ViT ──────────────────────────────────────────────────────
        vit_steps = _fine_tune(
            model         = _vit_model,
            processor     = _vit_processor,
            original_head = _vit_original_head,
            batch         = batch,
            label_fn      = _vit_label_index,
            model_name    = "vit",
        )

        # ── Fine-tune SigLIP ───────────────────────────────────────────────────
        siglip_steps = _fine_tune(
            model         = _siglip_model,
            processor     = _siglip_processor,
            original_head = _siglip_original_head,
            batch         = batch,
            label_fn      = _siglip_label_index,
            model_name    = "siglip",
        )

        # ── Save checkpoints ───────────────────────────────────────────────────
        _save_checkpoint(_vit_model,    "vit")
        _save_checkpoint(_siglip_model, "siglip")

        # ── Drift check every N corrections ───────────────────────────────────
        drift_result = {"triggered": False, "vit_acc": -1.0, "siglip_acc": -1.0}
        ran_drift    = False

        if n % DRIFT_CHECK_EVERY == 0:
            ran_drift    = True
            drift_result = run_drift_check()

        print(
            f"[trainer] Done — ViT:{vit_steps} steps, SigLIP:{siglip_steps} steps",
            flush=True,
        )

        return {
            "vit_steps":    vit_steps,
            "siglip_steps": siglip_steps,
            "drift_check":  ran_drift,
            "rollback":     drift_result["triggered"],
        }
