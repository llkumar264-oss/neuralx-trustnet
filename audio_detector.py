import librosa
import numpy as np

def analyze_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    score = float(np.mean(mfcc) % 1)

    return {
        "audio_fake_probability": round(score, 3),
        "is_fake": score > 0.6
    }
