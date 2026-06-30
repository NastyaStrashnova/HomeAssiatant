import os
from faster_whisper import WhisperModel
from langchain_openai import ChatOpenAI

print("Loading models...")

# Speech-to-text (unchanged)
whisper_model = WhisperModel("base", compute_type="int8")


agent_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
)

print("Models loaded.")
