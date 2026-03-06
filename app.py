"""
app.py
======
Gradio web UI for the ensemble deepfake detector.

Shows:
  - ViT individual result (always)
  - SigLIP individual result (only when ensemble path was taken)
  - Final ensemble verdict with confidence bar
  - Which inference path was used
  - ✅ / ❌ feedback buttons — wrong click triggers background fine-tuning
    of both ViT and SigLIP (no extra label question needed — auto-flipped)
  - 🔄 "Check Training Status" button — polls result after background thread done
"""

import os
import queue
import threading
import sys

# ── Path setup ─────────────────────────────────────────────────────────────────
# app.py lives in DeepFake_Predication/
# hardware.py is in the same folder → already on path when run from here
# ensemble, models, trainer, feedback_store live in ImagePredication/
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_IMAGE_DIR = os.path.join(_BASE_DIR, "ImagePredication")
if _IMAGE_DIR not in sys.path:
    sys.path.insert(0, _IMAGE_DIR)
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

import hardware

hardware.configure()
hardware.summary()

import gradio as gr
from PIL import Image
import ensemble
import feedback_store
import trainer

# Pre-populate replay buffer from previously saved corrections on disk
feedback_store.load_buffer_from_disk()

# ── Session state ──────────────────────────────────────────────────────────────
_last = {
    "image":   None,   # PIL Image
    "verdict": None,   # "Fake" or "Real"
}

# Queue used to pass training result from background thread → UI poll
# maxsize=1 — only the latest result matters
_train_result_queue: queue.Queue = queue.Queue(maxsize=1)


# ── Inference handler ──────────────────────────────────────────────────────────
def detect(image):
    if image is None:
        return {}, {}, {}, "⚠️ No image uploaded.", "Upload an image first."

    try:
        pil_image = Image.fromarray(image).convert("RGB")
        result    = ensemble.run(pil_image)
    except Exception as exc:
        err = f"❌ Inference error: {exc}"
        return {"Error": 1.0}, {"Error": 1.0}, {"Error": 1.0}, err, ""

    # Save last result for feedback buttons
    _last["image"]   = pil_image
    _last["verdict"] = result["verdict"]

    # ── ViT panel (always shown) ───────────────────────────────────────────────
    vit_output = {
        "Fake": round(result["vit_fake"], 4),
        "Real": round(result["vit_real"], 4),
    }

    # ── SigLIP panel (shown only when ensemble was triggered) ─────────────────
    if result["path"] == "full_ensemble":
        siglip_output = {
            "Fake": round(result["siglip_fake"], 4),
            "Real": round(result["siglip_real"], 4),
        }
    else:
        siglip_output = {"(not used — ViT was confident)": 1.0}

    # ── Final ensemble result ──────────────────────────────────────────────────
    final_output = {
        "Fake": round(result["fake_prob"], 4),
        "Real": round(result["real_prob"], 4),
    }

    # ── Path info ──────────────────────────────────────────────────────────────
    if result["path"] == "vit_only":
        path_info = (
            f"⚡ **Fast path** — ViT was confident ({max(result['vit_fake'], result['vit_real'])*100:.1f}%). "
            f"SigLIP skipped."
        )
    else:
        path_info = (
            f"🔍 **Full ensemble** — ViT was uncertain "
            f"({max(result['vit_fake'], result['vit_real'])*100:.1f}% < {ensemble.CONFIDENCE_THRESHOLD*100:.0f}%). "
            f"Both models used, soft-averaged."
        )

    feedback_status = (
        f"Prediction: **{result['verdict']}** — Was this correct? "
        f"Click ✅ or ❌ below."
    )

    return vit_output, siglip_output, final_output, path_info, feedback_status


# ── Feedback: correct ─────────────────────────────────────────────────────────
def feedback_correct():
    if _last["verdict"] is None:
        return "⚠️ No prediction yet — upload and detect first."
    count = feedback_store.correction_count()
    return (
        f"✅ **Prediction was correct — no training needed.**  "
        f"Session corrections so far: {count}"
    )


# ── Feedback: wrong — triggers background fine-tuning ─────────────────────────
def feedback_wrong():
    if _last["image"] is None or _last["verdict"] is None:
        return "⚠️ No prediction yet — upload and detect first."

    image           = _last["image"]
    predicted_label = _last["verdict"]

    # Record correction → auto-flips label, saves image, updates buffer
    correct_label = feedback_store.record_correction(image, predicted_label)

    # Drain any stale result from previous training so queue is clean
    try:
        _train_result_queue.get_nowait()
    except queue.Empty:
        pass

    # Launch fine-tuning in a background thread so UI stays responsive
    def _train():
        res = trainer.train_on_correction(image, correct_label)
        try:
            _train_result_queue.put_nowait(res)
        except queue.Full:
            pass   # already a newer result waiting

    threading.Thread(target=_train, daemon=True).start()

    total = feedback_store.total_logged()
    return (
        f"❌ **Wrong prediction recorded.**\n\n"
        f"- Model said: **{predicted_label}** → Correct label: **{correct_label}**\n"
        f"- 🔄 Both ViT & SigLIP fine-tuning in background...\n"
        f"- Total corrections logged: **{total}**\n\n"
        f"_Click **🔄 Check Training Status** below when ready._"
    )


# ── Training status poll ───────────────────────────────────────────────────────
def check_training_status():
    """
    Called by the 'Check Training Status' button.
    Reads the result that the background training thread posted to the queue.
    Non-blocking — returns immediately whether training is done or still running.
    """
    try:
        res = _train_result_queue.get_nowait()
    except queue.Empty:
        return "⏳ **Training still in progress** — check again in a few seconds."

    vit_steps    = res.get("vit_steps", 0)
    siglip_steps = res.get("siglip_steps", 0)
    rollback     = res.get("rollback", False)
    drift_ran    = res.get("drift_check", False)

    if rollback:
        return (
            f"⚠️ **Drift detected — models rolled back to previous backup.**\n\n"
            f"- ViT steps taken : {vit_steps}\n"
            f"- SigLIP steps taken: {siglip_steps}\n"
            f"- Recent corrections caused accuracy drop > threshold.\n"
            f"- Both models restored from backup. Keep providing feedback to improve them."
        )

    drift_note = (
        f"\n- Drift check ran: accuracy OK ✅"
        if drift_ran else
        f"\n- Drift check: skipped (runs every {trainer.DRIFT_CHECK_EVERY} corrections)"
    )

    return (
        f"✅ **Both models updated and saved.**\n\n"
        f"- ViT  : {vit_steps} gradient step(s)\n"
        f"- SigLIP: {siglip_steps} gradient step(s)"
        f"{drift_note}"
    )


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Ensemble Deepfake Detector") as demo:
    gr.Markdown(
        """
        # 🔍 Ensemble Deepfake Detector
        Combines **ViT** + **SigLIP** using a cascade + soft-vote strategy.
        - If ViT confidence ≥ 90% → result returned instantly (fast path).
        - If ViT is uncertain → SigLIP runs and both are averaged (full ensemble).
        """
    )

    with gr.Row():
        image_input = gr.Image(type="numpy", label="Upload Image")

    detect_btn = gr.Button("🔍 Detect", variant="primary")

    with gr.Row():
        vit_out    = gr.Label(num_top_classes=2, label="ViT Result")
        siglip_out = gr.Label(num_top_classes=2, label="SigLIP Result")
        final_out  = gr.Label(num_top_classes=2, label="✅ Final Verdict")

    path_out     = gr.Markdown(label="Inference Path")
    feedback_lbl = gr.Markdown(value="Upload an image and click Detect.")

    gr.Markdown("### Was the prediction correct?")
    with gr.Row():
        correct_btn = gr.Button("✅  Correct",              variant="secondary")
        wrong_btn   = gr.Button("❌  Wrong",                variant="stop")
        status_btn  = gr.Button("🔄  Check Training Status", variant="secondary")

    feedback_status = gr.Markdown(value="")

    # ── Wire up ────────────────────────────────────────────────────────────────
    detect_btn.click(
        fn=detect,
        inputs=[image_input],
        outputs=[vit_out, siglip_out, final_out, path_out, feedback_lbl],
    )

    correct_btn.click(
        fn=feedback_correct,
        inputs=[],
        outputs=[feedback_status],
    )

    wrong_btn.click(
        fn=feedback_wrong,
        inputs=[],
        outputs=[feedback_status],
    )

    status_btn.click(
        fn=check_training_status,
        inputs=[],
        outputs=[feedback_status],
    )

if __name__ == "__main__":
    demo.launch()
