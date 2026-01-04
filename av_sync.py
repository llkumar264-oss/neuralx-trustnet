import cv2
import librosa
import numpy as np

def analyze_av_sync(video_path):
    # Extract audio
    y, sr = librosa.load(video_path, sr=None)
    audio_energy = np.mean(librosa.feature.rms(y=y))

    # Extract video motion
    cap = cv2.VideoCapture(video_path)
    motions = []

    prev = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            motions.append(np.mean(np.abs(gray - prev)))
        prev = gray

    cap.release()

    video_motion = np.mean(motions) if motions else 0
    sync_score = abs(audio_energy - video_motion)

    return {
        "av_sync_score": round(sync_score, 3),
        "lip_sync_fake": sync_score > 0.5
    }
