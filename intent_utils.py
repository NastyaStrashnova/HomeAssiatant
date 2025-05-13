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
}

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

    return best_intent if best_score >= threshold else "unknown"