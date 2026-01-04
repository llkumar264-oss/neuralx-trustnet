import cv2
import os

def extract_frames(video_path, out_dir="temp/frames", fps=1):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(video_fps // fps, 1)

    count, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            cv2.imwrite(f"{out_dir}/frame_{saved}.jpg", frame)
            saved += 1

        count += 1

    cap.release()
    return out_dir
