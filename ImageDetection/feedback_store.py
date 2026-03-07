"""
feedback_store.py
=================
Manages all user feedback data:
  - Replay buffer : last N corrected images kept in memory for training
  - Image store   : saves corrected images to disk (feedback/images/)
  - CSV log       : appends every correction to feedback/feedback_log.csv

Balanced buffer
---------------
  At most MAX_PER_CLASS images per class (Fake / Real) are kept in memory.
  When a class hits its cap, the oldest item of that class is evicted first
  before the new one is added — so the buffer never skews toward one label.
  Total capacity: MAX_PER_CLASS * 2 = 100 images.

How correct label is determined (NO user input needed)
-------------------------------------------------------
  Model predicted "Fake" + user clicked Wrong  →  correct label = "Real"
  Model predicted "Real" + user clicked Wrong  →  correct label = "Fake"

Public API
----------
  record_correction(image, predicted_label)
      → saves image, logs to CSV, adds to buffer, returns correct_label

  get_replay_batch(n)
      → returns up to n random (image, label) pairs from buffer

  buffer_size()
      → total images currently in the replay buffer

  correction_count()
      → total corrections since startup (session-only)

  total_logged()
      → total corrections ever logged (from CSV, survives restarts)

  load_buffer_from_disk()
      → pre-populates buffer from saved images at startup
"""

import os
import csv
import random
import hashlib
import datetime
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")
IMAGE_DIR    = os.path.join(FEEDBACK_DIR, "images")
LOG_PATH     = os.path.join(FEEDBACK_DIR, "feedback_log.csv")

os.makedirs(IMAGE_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_PER_CLASS = 50          # max images per class in replay buffer (total = 100)
LOG_MAX_ROWS  = 1000        # rotate CSV log beyond this many rows
LOG_ARCHIVE   = os.path.join(FEEDBACK_DIR, "feedback_log_archive.csv")

# ── CSV fields ─────────────────────────────────────────────────────────────────
_CSV_FIELDS = [
    "timestamp",
    "image_hash",
    "predicted_label",
    "correct_label",
    "image_path",
]

# ── Internal state ─────────────────────────────────────────────────────────────
# Each entry: {"image": PIL.Image, "label": "Fake" | "Real"}
# Two separate lists maintain per-class cap; combined they form the replay buffer.
_fake_buf: list = []    # items with label == "Fake"
_real_buf: list = []    # items with label == "Real"

# Session correction count (resets on restart)
_session_count: int = 0


# ── CSV helpers ────────────────────────────────────────────────────────────────
def _ensure_csv() -> None:
    if not os.path.isfile(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=_CSV_FIELDS).writeheader()


def _append_csv(row: dict) -> None:
    _ensure_csv()
    with open(LOG_PATH, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=_CSV_FIELDS).writerow(row)
    _maybe_rotate_log()


def _maybe_rotate_log() -> None:
    """
    If the log exceeds LOG_MAX_ROWS data rows, archive the oldest half and
    keep the newest half in the live log.  This prevents unbounded file growth.
    """
    _ensure_csv()
    try:
        with open(LOG_PATH, "r", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return

    if len(rows) <= LOG_MAX_ROWS:
        return

    keep_n   = LOG_MAX_ROWS // 2
    overflow = rows[:-keep_n]      # older rows to archive
    keep     = rows[-keep_n:]      # newer rows to keep

    # Append overflow to archive file
    write_header = not os.path.isfile(LOG_ARCHIVE)
    try:
        with open(LOG_ARCHIVE, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            if write_header:
                w.writeheader()
            w.writerows(overflow)
    except Exception:
        return   # don't clobber live log if archive write fails

    # Rewrite live log with only the kept rows
    try:
        with open(LOG_PATH, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            w.writeheader()
            w.writerows(keep)
        print(
            f"[feedback_store] Log rotated — archived {len(overflow)} rows, "
            f"kept {len(keep)} rows.",
            flush=True,
        )
    except Exception:
        pass


def total_logged() -> int:
    """Count total rows in the live CSV log (does not include archived rows)."""
    _ensure_csv()
    try:
        with open(LOG_PATH, "r", newline="") as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except Exception:
        return 0


# ── Image hash ─────────────────────────────────────────────────────────────────
def _image_hash(image: Image.Image) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()[:16]


# ── Balanced buffer helpers ───────────────────────────────────────────────────
def _buf_for(label: str) -> list:
    return _fake_buf if label == "Fake" else _real_buf


def _add_to_buffer(image: Image.Image, label: str) -> None:
    """
    Add item to the per-class list. If the class is at cap, evict the
    oldest item of that class first (FIFO within class).
    """
    buf = _buf_for(label)
    if len(buf) >= MAX_PER_CLASS:
        buf.pop(0)          # evict oldest of same class
    buf.append({"image": image.copy(), "label": label})


def buffer_size() -> int:
    """Total images currently in the replay buffer (both classes)."""
    return len(_fake_buf) + len(_real_buf)


# ── Public API ─────────────────────────────────────────────────────────────────
def record_correction(image: Image.Image, predicted_label: str) -> str:
    """
    Record a user correction.

    Parameters
    ----------
    image           : PIL image that was predicted wrongly
    predicted_label : what the model said — "Fake" or "Real"

    Returns
    -------
    correct_label : the flipped label
    """
    global _session_count

    correct_label = "Real" if predicted_label == "Fake" else "Fake"

    # ── Save image to disk ─────────────────────────────────────────────────────
    image    = image.convert("RGB")
    img_hash = _image_hash(image)
    fname    = f"{img_hash}_{correct_label}.jpg"
    img_path = os.path.join(IMAGE_DIR, fname)

    if not os.path.isfile(img_path):
        image.save(img_path, "JPEG", quality=95)

    # ── Append to CSV ──────────────────────────────────────────────────────────
    _append_csv({
        "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
        "image_hash":      img_hash,
        "predicted_label": predicted_label,
        "correct_label":   correct_label,
        "image_path":      img_path,
    })

    # ── Add to balanced buffer ────────────────────────────────────────────────
    _add_to_buffer(image, correct_label)

    _session_count += 1

    print(
        f"[feedback_store] Correction #{_session_count} — "
        f"predicted={predicted_label}  correct={correct_label}  "
        f"buffer={buffer_size()} (fake={len(_fake_buf)} real={len(_real_buf)})",
        flush=True,
    )

    return correct_label


def get_replay_batch(n: int) -> list[dict]:
    """
    Return up to n random samples from the replay buffer (both classes combined).
    Each item: {"image": PIL.Image, "label": "Fake" | "Real"}
    """
    combined = _fake_buf + _real_buf
    if not combined:
        return []
    k = min(n, len(combined))
    return random.sample(combined, k)


def correction_count() -> int:
    """Number of corrections recorded this session."""
    return _session_count


def load_buffer_from_disk() -> None:
    """
    Pre-populate the replay buffer from saved images on disk at startup.
    Reads the CSV log for labels, loads images from IMAGE_DIR.
    Respects balanced cap (MAX_PER_CLASS per class).
    Call once at startup.
    """
    global _fake_buf, _real_buf
    _fake_buf = []
    _real_buf = []

    _ensure_csv()
    rows = []
    try:
        with open(LOG_PATH, "r", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return

    # Process newest-first so we fill the cap with the most recent images
    loaded_fake = loaded_real = 0
    for row in reversed(rows):
        img_path = row.get("image_path", "")
        label    = row.get("correct_label", "")
        if not img_path or label not in ("Fake", "Real"):
            continue
        if not os.path.isfile(img_path):
            continue

        # Stop loading a class once it hits the cap
        if label == "Fake"  and loaded_fake >= MAX_PER_CLASS:
            continue
        if label == "Real" and loaded_real >= MAX_PER_CLASS:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        if label == "Fake":
            _fake_buf.insert(0, {"image": img, "label": label})   # insert at front (oldest)
            loaded_fake += 1
        else:
            _real_buf.insert(0, {"image": img, "label": label})
            loaded_real += 1

        if loaded_fake >= MAX_PER_CLASS and loaded_real >= MAX_PER_CLASS:
            break

    total = loaded_fake + loaded_real
    if total:
        print(
            f"[feedback_store] Loaded {total} images into replay buffer "
            f"(fake={loaded_fake}  real={loaded_real})",
            flush=True,
        )
