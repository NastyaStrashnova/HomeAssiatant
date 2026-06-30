import os
from faster_whisper import WhisperModel
from langchain_openai import ChatOpenAI

print("Loading models...")

# Speech-to-text (unchanged)
whisper_model = WhisperModel("base", compute_type="int8")

# LLM used by the LangChain tool-calling agent in intent_utils.py.
# Requires OPENAI_API_KEY to be set in the environment.
# gpt-4o-mini is a good default: cheap, fast, and reliable at tool calling.
agent_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
)

print("Models loaded.")