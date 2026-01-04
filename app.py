from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
import shutil, os, uuid

from image_ai.image_detector import analyze_image
from video_ai.frame_extractor import extract_frames
from video_ai.video_detector import analyze_video
from audio_ai.audio_detector import analyze_audio
from fusion.fusion_logic import multi_modal_decision
from sync.av_sync import analyze_av_sync

app = FastAPI(title="NeuralX TrustNet API")

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze_media(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    image = video = audio = sync = None

    if file.filename.endswith((".jpg", ".png")):
        image = analyze_image(file_path)

    elif file.filename.endswith((".mp4", ".avi")):
        frames = extract_frames(file_path)
        video = analyze_video(frames)
        sync = analyze_av_sync(file_path)

    elif file.filename.endswith((".wav", ".mp3")):
        audio = analyze_audio(file_path)

    final = multi_modal_decision(
        image=image,
        video=video,
        audio=audio,
        sync=sync
    )

    return JSONResponse({
        "image": image,
        "video": video,
        "audio": audio,
        "sync": sync,
        "final_decision": final
    })


@app.get("/")
def dashboard():
    with open("dashboard/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
