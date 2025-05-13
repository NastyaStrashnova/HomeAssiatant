from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

print("Loading models...")
whisper_model = WhisperModel("base", compute_type="int8")  # Or "tiny"
intent_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Models loaded.")
