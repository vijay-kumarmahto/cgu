import torch
from torchvision import transforms
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import os
import sys

# Ensure we import GenConViTDetector from THIS directory's models.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Import models.py from VideoPredication using absolute path
import importlib.util
_models_path = os.path.join(_THIS_DIR, 'models.py')
_spec = importlib.util.spec_from_file_location('video_models', _models_path)
_video_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_video_models)
GenConViTDetector = _video_models.GenConViTDetector

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_frames(video_path, num_frames=10):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    if total_frames < num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        frame = vr[idx].asnumpy()
        pil_frame = Image.fromarray(frame)
        frames.append(transform(pil_frame))
    
    return torch.stack(frames)

def predict_video(video_path):
    detector = GenConViTDetector()
    frames_tensor = extract_frames(video_path).to(detector.device)
    score = detector.predict(frames_tensor)
    
    is_fake = score > 0.5
    confidence = score if is_fake else (1 - score)
    label = "FAKE" if is_fake else "REAL"
    
    return label, confidence * 100, score
