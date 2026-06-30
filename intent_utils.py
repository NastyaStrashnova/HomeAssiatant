"""
LangChain agentic workflow for the voice assistant.

Each smart-home action is implemented as a standalone LangChain @tool
function. The agent (an LLM with tool-calling) listens to the transcribed
text, decides which tool(s) to call, and each tool's only job is to flip
the relevant entry in `device_states` and report what changed.
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from model_loader import agent_llm

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
device_states = {
    "fan": 0,
    "heater": 0,
    "lights": 0,
    "music": 0,
}

# Track which keys changed during the most recent agent run so the API
# response can report it without the tools needing to return the full state.
_last_change = {}


def _set_state(key: str, value: int) -> str:
    """Helper: update device_states and record the change."""
    changed = device_states[key] != value
    device_states[key] = value
    if changed:
        _last_change[key] = value
    return f"{key} set to {'on' if value else 'off'}"


# ---------------------------------------------------------------------------
# Tools - one per discrete action. Each returns only the state change.
# ---------------------------------------------------------------------------
@tool
def turn_on_fan() -> str:
    """Turn the fan ON. Use this when the user wants more air/breeze or says it's hot."""
    return _set_state("fan", 1)


@tool
def turn_off_fan() -> str:
    """Turn the fan OFF. Use this when the user wants the fan stopped or says it's cold."""
    return _set_state("fan", 0)


@tool
def turn_on_heater() -> str:
    """Turn the heater ON. Use this when the user is cold and the fan is already off, or explicitly asks for heat."""
    return _set_state("heater", 1)


@tool
def turn_off_heater() -> str:
    """Turn the heater OFF. Use this when the user wants to stop the heater or no longer needs heat."""
    return _set_state("heater", 0)


@tool
def turn_on_lights() -> str:
    """Turn the lights ON. Use this when the user says it's too dark or wants more light."""
    return _set_state("lights", 1)


@tool
def turn_off_lights() -> str:
    """Turn the lights OFF. Use this when the user says it's too bright or wants it dark."""
    return _set_state("lights", 0)


@tool
def turn_on_music() -> str:
    """Turn the music ON. Use this when the user wants to hear music or asks to play something."""
    return _set_state("music", 1)


@tool
def turn_off_music() -> str:
    """Turn the music OFF. Use this when the user wants quiet or asks to stop the music."""
    return _set_state("music", 0)


TOOLS = [
    turn_on_fan,
    turn_off_fan,
    turn_on_heater,
    turn_off_heater,
    turn_on_lights,
    turn_off_lights,
    turn_on_music,
    turn_off_music,
]

# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a smart-home voice assistant. The user's speech has
been transcribed for you. Decide which, if any, of the available tools to
call based on what the user said. Only call a tool if the request clearly
maps to a device action. Consider current device states when resolving
ambiguous requests (e.g. "it's cold" could mean turning the heater on OR
the fan off, depending on what's currently running). If nothing applicable
is requested, don't call any tool and just respond briefly.
"""

agent = create_agent(
    model=agent_llm,
    tools=TOOLS,
    system_prompt=_SYSTEM_PROMPT,
)


def run_agent(text: str) -> dict:
    """
    Run the agentic workflow on transcribed text.

    Returns a dict with:
      - "changes": dict of only the device states that changed this turn
      - "states": the full current device_states
    """
    _last_change.clear()
    # Give the agent visibility into current state via the human message itself,
    # since create_agent's prompt is a static system_prompt rather than a template.
    user_message = (
        f"Current device states: {device_states}\n"
        f"User said: {text}"
    )
    agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    return {
        "changes": dict(_last_change),
        "states": dict(device_states),
    }