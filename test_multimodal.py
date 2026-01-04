from image_ai.image_detector import analyze_image
from video_ai.frame_extractor import extract_frames
from video_ai.video_detector import analyze_video
from audio_ai.audio_detector import analyze_audio
from fusion.fusion_logic import multi_modal_decision

# Sample files (optional – jo available ho wahi chalega)
image = analyze_image("sample.jpg") if False else None

frames = extract_frames("test_video.mp4")
video = analyze_video(frames)

audio = analyze_audio("sample.wav") if False else None

final = multi_modal_decision(
    image=image,
    video=video,
    audio=audio
)

print("\n=== MULTI-MODAL TRUST DECISION ===")
print(final)
