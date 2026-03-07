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
import time

# ── Path setup ─────────────────────────────────────────────────────────────────
# app.py lives in DeepFake_Detection/
# ImageDetection must be FIRST in path (has models.py for ensemble)
# VideoDetection added LAST (has different models.py for video)
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_IMAGE_DIR = os.path.join(_BASE_DIR, "ImageDetection")
_VIDEO_DIR = os.path.join(_BASE_DIR, "VideoDetection")
_AUDIO_DIR = os.path.join(_BASE_DIR, "AudioDetection")

# Order matters: ImageDetection first
if _IMAGE_DIR not in sys.path:
    sys.path.insert(0, _IMAGE_DIR)
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

import gradio as gr
from PIL import Image
import ensemble
import feedback_store
import trainer

# Load video models at startup
import importlib.util
_models_path = os.path.join(_VIDEO_DIR, 'models.py')
_spec = importlib.util.spec_from_file_location('video_models', _models_path)
_video_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_video_models)
_video_models.GenConViTDetector()  # Initialize singleton

video_inference_path = os.path.join(_VIDEO_DIR, "inference.py")
spec = importlib.util.spec_from_file_location("video_inference", video_inference_path)
video_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video_module)
_video_predict = video_module.predict_video

# Load audio inference
audio_inference_path = os.path.join(_AUDIO_DIR, "inference.py")
spec = importlib.util.spec_from_file_location("audio_inference", audio_inference_path)
audio_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_module)
_audio_analyze = audio_module.analyze_audio

# Pre-populate replay buffer from previously saved corrections on disk
feedback_store.load_buffer_from_disk()

# ── Session state ──────────────────────────────────────────────────────────────
_last = {
    "image":   None,   # PIL Image
    "verdict": None,   # "Fake" or "Real"
}

_last_video = {
    "path":    None,   # video file path
    "verdict": None,   # "FAKE" or "REAL"
    "score":   None,   # raw score
}

_last_audio = {
    "path":    None,   # audio file path
    "verdict": None,   # "Fake" or "Real"
    "fake_prob": None,
    "real_prob": None,
}

# Queue used to pass training result from background thread → UI poll
# maxsize=1 — only the latest result matters
_train_result_queue: queue.Queue = queue.Queue(maxsize=1)

# Queue for live training progress updates
_train_progress_queue: queue.Queue = queue.Queue()


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

    # Clear progress queue
    while not _train_progress_queue.empty():
        try:
            _train_progress_queue.get_nowait()
        except queue.Empty:
            break

    # Launch fine-tuning in a background thread so UI stays responsive
    def _train():
        correction_num = feedback_store.correction_count()
        _train_progress_queue.put(f"🔄 **Training correction #{correction_num}...**")
        
        res = trainer.train_on_correction(image, correct_label)
        
        try:
            _train_result_queue.put_nowait(res)
        except queue.Full:
            pass   # already a newer result waiting

    threading.Thread(target=_train, daemon=True).start()

    total = feedback_store.total_logged()
    correction_num = feedback_store.correction_count()
    return (
        f"❌ **Wrong prediction recorded.**\n\n"
        f"- Model said: **{predicted_label}** → Correct label: **{correct_label}**\n"
        f"- 🔄 Training correction **#{correction_num}** in background...\n"
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
        # Check for progress updates
        try:
            progress = _train_progress_queue.get_nowait()
            return progress + "\n\n⏳ **Training still in progress** — check again in a few seconds."
        except queue.Empty:
            return "⏳ **Training still in progress** — check again in a few seconds."

    vit_steps    = res.get("vit_steps", 0)
    siglip_steps = res.get("siglip_steps", 0)
    rollback     = res.get("rollback", False)
    drift_ran    = res.get("drift_check", False)
    correction_num = res.get("correction_num", "?")

    if rollback:
        return (
            f"⚠️ **Drift detected — models rolled back to previous backup.**\n\n"
            f"- Correction: **#{correction_num}**\n"
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
        f"✅ **Correction #{correction_num} training complete!**\n\n"
        f"- ViT  : {vit_steps} gradient step(s)\n"
        f"- SigLIP: {siglip_steps} gradient step(s)"
        f"{drift_note}"
    )


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Ensemble Deepfake Detector") as demo:
    gr.Markdown(
        """
        # 🔍 Deepfake Detector
        """
    )

    with gr.Tabs():
        # ── Image Tab ──────────────────────────────────────────────────────────
        with gr.Tab("📷 Image"):
            with gr.Row():
                image_input = gr.Image(type="numpy", label="Upload Image")

            image_detect_btn = gr.Button("Detect", variant="primary")

            with gr.Row():
                image_correct_btn = gr.Button("✅  Correct", variant="secondary")
                image_wrong_btn   = gr.Button("❌  Wrong", variant="stop")
                image_status_btn  = gr.Button("🔄  Check Training Status", variant="secondary")
    
            
            image_output = gr.Markdown(value="Upload an image and click Detect.")

            with gr.Row():
                vit_out    = gr.Label(num_top_classes=2, label="ViT Result")
                siglip_out = gr.Label(num_top_classes=2, label="SigLIP Result")
                final_out  = gr.Label(num_top_classes=2, label="✅ Final Verdict")

            image_feedback_status = gr.Markdown(value="")

            def detect_image(image):
                if image is None:
                    return "⚠️ No image uploaded.", {}, {}, {}
                
                try:
                    pil_image = Image.fromarray(image).convert("RGB")
                    result    = ensemble.run(pil_image)
                except Exception as exc:
                    return f"❌ Inference error: {exc}", {"Error": 1.0}, {"Error": 1.0}, {"Error": 1.0}

                # Save last result for feedback buttons
                _last["image"]   = pil_image
                _last["verdict"] = result["verdict"]

                vit_output = {
                    "Fake": round(result["vit_fake"], 4),
                    "Real": round(result["vit_real"], 4),
                }

                if result["path"] == "full_ensemble":
                    siglip_output = {
                        "Fake": round(result["siglip_fake"], 4),
                        "Real": round(result["siglip_real"], 4),
                    }
                else:
                    siglip_output = {"(not used — ViT was confident)": 1.0}

                final_output = {
                    "Fake": round(result["fake_prob"], 4),
                    "Real": round(result["real_prob"], 4),
                }

                return "", vit_output, siglip_output, final_output

            # ── Wire up ────────────────────────────────────────────────────────
            image_detect_btn.click(
                fn=detect_image,
                inputs=[image_input],
                outputs=[image_output, vit_out, siglip_out, final_out],
            )

            image_correct_btn.click(
                fn=feedback_correct,
                inputs=[],
                outputs=[image_feedback_status],
            )

            image_wrong_btn.click(
                fn=feedback_wrong,
                inputs=[],
                outputs=[image_feedback_status],
            )

            image_status_btn.click(
                fn=check_training_status,
                inputs=[],
                outputs=[image_feedback_status],
            )

        # ── Video Tab ──────────────────────────────────────────────────────────
        with gr.Tab("🎥 Video"):
            with gr.Row():
                video_input = gr.Video(label="Upload Video")
            
            video_detect_btn = gr.Button("Detect", variant="primary")

            with gr.Row():
                video_correct_btn = gr.Button("✅  Correct", variant="secondary")
                video_wrong_btn   = gr.Button("❌  Wrong", variant="stop")
                video_status_btn  = gr.Button("🔄  Check Training Status", variant="secondary")
    
            
            video_output = gr.Markdown(value="Upload a video and click Detect.")
            video_results = gr.Dataframe(
                headers=["Frame", "Timestamp", "Verdict", "Fake %", "Real %"],
                label="Frame Analysis Results",
                interactive=False
            )
            
            video_feedback_status = gr.Markdown(value="")

            def analyze_video(video_path):
                if video_path is None:
                    return "⚠️ No video uploaded.", []
                
                try:
                    label, confidence, score = _video_predict(video_path)
                    
                    _last_video["path"] = video_path
                    _last_video["verdict"] = label
                    _last_video["score"] = score
                    
                    result_md = f"""## 🎬 Video Analysis Result

**Verdict:** {label}  
**Confidence:** {confidence:.2f}%  
**Raw Score:** {score:.4f}

*Analysis based on 10 sampled frames using GenConViT ensemble.*
"""
                    return result_md, []
                    
                except Exception as e:
                    return f"❌ **Video analysis failed:** {e}", []
            
            video_detect_btn.click(
                fn=analyze_video,
                inputs=[video_input],
                outputs=[video_output, video_results],
            )

            def video_feedback_correct():
                if _last_video["verdict"] is None:
                    return "⚠️ No prediction yet — upload and detect first."
                return f"✅ **Prediction was correct — no training needed.**\n\nVerdict: {_last_video['verdict']}"
            
            def video_feedback_wrong():
                if _last_video["verdict"] is None:
                    return "⚠️ No prediction yet — upload and detect first."
                return f"❌ **Wrong prediction recorded.**\n\n⚠️ Video model fine-tuning not yet implemented.\n\nPredicted: {_last_video['verdict']}"
            
            video_correct_btn.click(
                fn=video_feedback_correct,
                inputs=[],
                outputs=[video_feedback_status],
            )

            video_wrong_btn.click(
                fn=video_feedback_wrong,
                inputs=[],
                outputs=[video_feedback_status],
            )

            video_status_btn.click(
                fn=lambda: "⚠️ Video model training not yet implemented.",
                inputs=[],
                outputs=[video_feedback_status],
            )

        # ── Audio Tab ──────────────────────────────────────────────────────────
        with gr.Tab("🎤 Audio"):
            with gr.Row():
                audio_input = gr.Audio(label="Upload Audio", type="filepath")
            
            audio_detect_btn = gr.Button("Detect", variant="primary")

            with gr.Row():
                audio_correct_btn = gr.Button("✅  Correct", variant="secondary")
                audio_wrong_btn   = gr.Button("❌  Wrong", variant="stop")
                audio_status_btn  = gr.Button("🔄  Check Training Status", variant="secondary")
            
            audio_output = gr.Markdown(value="Upload an audio file and click Detect.")
            audio_details = gr.JSON(label="Analysis Details")
            
            audio_feedback_status = gr.Markdown(value="")
            
            def analyze_audio(audio_path):
                if audio_path is None:
                    return "⚠️ No audio uploaded.", {}
                
                try:
                    result = _audio_analyze(audio_path)
                    
                    _last_audio["path"] = audio_path
                    _last_audio["verdict"] = result["verdict"]
                    _last_audio["fake_prob"] = result["fake_prob"]
                    _last_audio["real_prob"] = result["real_prob"]
                    
                    result_md = f"""## 🎤 Audio Analysis Result

**Verdict:** {result['verdict']}  
**Confidence:** {result['confidence']*100:.2f}%

**Probabilities:**
- Fake: {result['fake_prob']*100:.2f}%
- Real: {result['real_prob']*100:.2f}%

*Analysis using Wav2Vec2 audio deepfake detector.*
"""
                    details = {
                        "verdict": result["verdict"],
                        "fake_probability": f"{result['fake_prob']:.4f}",
                        "real_probability": f"{result['real_prob']:.4f}",
                        "confidence": f"{result['confidence']:.4f}"
                    }
                    return result_md, details
                    
                except Exception as e:
                    return f"❌ **Audio analysis failed:** {e}", {}
            
            audio_detect_btn.click(
                fn=analyze_audio,
                inputs=[audio_input],
                outputs=[audio_output, audio_details],
            )

            def audio_feedback_correct():
                if _last_audio["verdict"] is None:
                    return "⚠️ No prediction yet — upload and detect first."
                return f"✅ **Prediction was correct — no training needed.**\n\nVerdict: {_last_audio['verdict']}"
            
            def audio_feedback_wrong():
                if _last_audio["verdict"] is None:
                    return "⚠️ No prediction yet — upload and detect first."
                return f"❌ **Wrong prediction recorded.**\n\n⚠️ Audio model fine-tuning not yet implemented.\n\nPredicted: {_last_audio['verdict']}"
            
            audio_correct_btn.click(
                fn=audio_feedback_correct,
                inputs=[],
                outputs=[audio_feedback_status],
            )

            audio_wrong_btn.click(
                fn=audio_feedback_wrong,
                inputs=[],
                outputs=[audio_feedback_status],
            )

            audio_status_btn.click(
                fn=lambda: "⚠️ Audio model training not yet implemented.",
                inputs=[],
                outputs=[audio_feedback_status],
            )

if __name__ == "__main__":
    demo.launch()
