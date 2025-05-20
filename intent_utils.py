
from model_loader import intent_model
from sentence_transformers import util

known_intents = {
    "lights_on": [
        "turn on the lights", "it's too dark", "light it up", "make it brighter",
        "switch the lights on", "bring some light"
    ],
    "lights_off": [
        "turn off the lights", "it's too bright", "lights out", "make it dark",
        "shut the lights", "let's go dark"
    ],
    "fan_on": [
        "turn on the fan", "it's too hot", "start the fan", "i need air",
        "fan on please", "more breeze"
    ],
    "fan_off": [
        "turn off the fan", "it's cold", "stop the fan", "fan off please",
        "kill the fan", "enough fan"
    ]
    "music_on": [
    "play some music", "start the music", "music on", 
    "i want to hear something", "let’s jam", "turn on the tunes"
    ],
    "music_off": [
        "stop the music", "music off", "turn off the tunes", 
        "kill the music", "quiet time", "enough music"
    ]
    "volume_up": [
        "make it louder", "turn it up", "increase the volume", 
        "i can't hear", "volume up", "raise the sound"
    ],
    "volume_down": [
        "make it quieter", "turn it down", "lower the volume", 
        "it's too loud", "volume down", "reduce the sound"
    ]
    "heat_on": [
        "turn on the heater", "it's cold in here", "warm it up", 
        "start heating", "i need warmth"
    ],
    "heat_off": [
        "turn off the heater", "too warm", "stop the heating", 
        "cool it down", "kill the heat"
    ]

}

def resolve_ambiguous_intent(base_intent):
    # Context-aware resolution
    if base_intent == "heat_on" and device_states["fan"] == 1:
        return "fan_off"
    elif base_intent == "fan_on" and device_states["heater"] == 1:
        return "heat_off"
    elif base_intent == "heat_off" and device_states["fan"] == 0:
        return "fan_on"
    elif base_intent == "fan_off" and device_states["heater"] == 0:
        return "heat_on"
    return base_intent


def detect_intent(text, threshold=0.5):
    input_embedding = intent_model.encode(text, convert_to_tensor=True)
    best_intent, best_score = None, 0

    for intent, examples in known_intents.items():
        example_embeddings = intent_model.encode(examples, convert_to_tensor=True)
        scores = util.cos_sim(input_embedding, example_embeddings)
        max_score = scores.max().item()

        if max_score > best_score:
            best_score = max_score
            best_intent = intent

    if best_score < threshold:
        return "unknown"

    # Add post-processing for context-aware resolution
    resolved_intent = resolve_ambiguous_intent(best_intent)
    return resolved_intent

