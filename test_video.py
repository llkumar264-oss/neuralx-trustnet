from video_ai.frame_extractor import extract_frames
from video_ai.video_detector import analyze_video

print("Video analysis start ho raha hai...")

# 1. Video ka naam
video_path = "test_video.mp4"

# 2. Video ko frames me todna
frames_dir = extract_frames(video_path)

# 3. Frames se fake/real nikalna
result = analyze_video(frames_dir)

# 4. Result dikhao
print("RESULT 👇")
print(result)
