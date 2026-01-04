import torch
from PIL import Image
import os

from video_ai.video_model import model, processor, device

def analyze_video(frame_dir):
    frames = []

    for file in sorted(os.listdir(frame_dir))[:8]:
        img = Image.open(os.path.join(frame_dir, file)).convert("RGB")
        frames.append(img)

    inputs = processor(frames, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    fake_score = float(torch.max(probs))

    return {
        "video_fake_probability": round(fake_score, 3),
        "is_deepfake": fake_score > 0.65
    }
