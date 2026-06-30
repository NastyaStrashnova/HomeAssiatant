"""
LangChain agentic workflow for the voice assistant.

Each smart-home device is controlled by a single LangChain @tool function
that takes the desired state ("on"/"off") as an argument. The agent (an LLM
with tool-calling) listens to the transcribed text, looks at the current
device states (which are included in the prompt), decides which device(s)
need to change and to what state, and calls the matching tool with that
state. Each tool's only job is to flip the relevant entry in
`device_states` and report what changed.
"""

from typing import Literal

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
# Tools - one per device. The agent reads the current state (given to it in
# the prompt) and decides what the new state should be, then passes it in.
# ---------------------------------------------------------------------------
State = Literal["on", "off"]


@tool
def set_fan(state: State) -> str:
    """Set the fan to "on" or "off". Use this when the user wants more air/breeze
    or says it's hot (state="on"), or wants the fan stopped / says it's cold
    (state="off"). Check the current fan state given in the prompt before
    deciding - don't turn it on if it's already on, etc."""
    return _set_state("fan", 1 if state == "on" else 0)


@tool
def set_heater(state: State) -> str:
    """Set the heater to "on" or "off". Use this when the user is cold and the
    fan is already off, or explicitly asks for heat (state="on"), or wants to
    stop the heater / no longer needs heat (state="off"). Check the current
    heater state given in the prompt before deciding."""
    return _set_state("heater", 1 if state == "on" else 0)


@tool
def set_lights(state: State) -> str:
    """Set the lights to "on" or "off". Use this when the user says it's too
    dark or wants more light (state="on"), or says it's too bright / wants it
    dark (state="off"). Check the current lights state given in the prompt
    before deciding."""
    return _set_state("lights", 1 if state == "on" else 0)


@tool
def set_music(state: State) -> str:
    """Set the music to "on" or "off". Use this when the user wants to hear
    music or asks to play something (state="on"), or wants quiet / asks to
    stop the music (state="off"). Check the current music state given in the
    prompt before deciding."""
    return _set_state("music", 1 if state == "on" else 0)


TOOLS = [
    set_fan,
    set_heater,
    set_lights,
    set_music,
]

# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a smart-home voice assistant called Kiki. The user's
speech has been transcribed for you. Each device has exactly one tool
(set_fan, set_heater, set_lights, set_music) that takes a state argument of
"on" or "off". Decide which device(s), if any, need to change based on what
the user said, and call the matching tool with the correct state. Only call
a tool if the request clearly maps to a device action. Always check the
current device states given to you in the prompt before deciding - e.g.
don't call set_fan(state="on") if the fan is already on, and resolve
ambiguous requests (e.g. "it's cold" could mean turning the heater on OR the
fan off) using the current states. If nothing applicable is requested,
don't call any tool and just respond briefly.
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