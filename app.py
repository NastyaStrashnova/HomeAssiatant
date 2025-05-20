from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from model_loader import whisper_model
from intent_utils import detect_intent
import soundfile as sf
import tempfile

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("test.html", "r") as file:
        return HTMLResponse(content=file.read())

@app.post("/transcribe/")
async def transcribe_and_detect(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_bytes = await file.read()
        tmp.write(audio_bytes)
        tmp.flush()

        segments, _ = whisper_model.transcribe(tmp.name)
        full_text = " ".join([seg.text for seg in segments])

        intent = detect_intent(full_text)
        apply_intent(intent)

    return {"transcription": full_text, "intent": intent, "states": device_states}
