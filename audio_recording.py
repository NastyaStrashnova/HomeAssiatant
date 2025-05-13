import sounddevice as sd
import numpy as np
import io
import wave
import requests
import time
from threading import Thread

# Parameters for recording
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 3  # seconds per request, adjust as needed
FASTAPI_URL = "http://127.0.0.1:8000/transcribe/"  # Update with your server URL

def record_audio():
    """Function to record audio in real time and send to FastAPI server."""
    while True:
        # Record audio for a defined duration
        audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
        sd.wait()

        # Save the audio data to a memory buffer (in-memory file)
        byte_io = io.BytesIO()
        with wave.open(byte_io, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 2 bytes per sample
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        
        byte_io.seek(0)  # Reset pointer to start of the stream

        # Send audio to FastAPI server for transcription
        response = requests.post(FASTAPI_URL, files={"file": byte_io})
        
        if response.status_code == 200:
            print("Transcription:", response.json().get("transcription"))
            print("Detected Intent:", response.json().get("intent"))
        else:
            print(f"Error: {response.status_code}")
        
        time.sleep(1)  # Delay between consecutive audio recordings

def start_audio_recording():
    """Start recording in a separate thread to run in parallel with the FastAPI server."""
    thread = Thread(target=record_audio)
    thread.daemon = True  # Ensures thread will exit when the main program exits
    thread.start()
