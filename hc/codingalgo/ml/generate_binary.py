import json
import random
import datetime
import uuid

# --- Configuration ---
NUM_PROMPTS = 10000
NUM_LABELERS = 100
FILENAME = "large_binary_data.json"

# 1. Define Labels (Binary & Unbalanced)
LABELS_BINARY = ["not_toxic", "toxic"]
LABEL_WEIGHTS = [0.80, 0.20]  # 80% not_toxic, 20% toxic

# 2. Define Metadata Options (with None for missing data)
USER_COUNTRIES = ["US", "CA", "UK", "IN", "AU", "DE", "FR", "JP", None]
USER_AGE_GROUPS = ["18-25", "26-35", "36-45", "46+", None]
LABELER_EXPERIENCE = ["beginner", "intermediate", "expert"]

# 3. Simple prompt templates
PROMPT_TEMPLATES = {
    "not_toxic": [
        "How do I bake a {food}?", "What is the capital of {country}?", "Explain {concept} in simple terms.",
        "Translate '{phrase}' to {language}.", "Who won the {event} in {year}?"
    ],
    "toxic": [
        "You are a {insult} machine.", "That's a {adjective} stupid idea.", "Why can't you {action} right?",
        "This is the {adjective} worst service ever.", "You're completely {insult} useless.",
        "I can't believe how {insult} you are.", "People like you deserve to {action}.",
        "I'm going to {action} you and your {relative}."
    ]
}

# Helper to get a random prompt
def get_random_prompt(label):
    template = random.choice(PROMPT_TEMPLATES.get(label, PROMPT_TEMPLATES["not_toxic"]))
    return (
        template.replace("{food}", random.choice(["bread", "cake", "cookies"]))
        .replace("{country}", random.choice(["France", "Japan", "Brazil"]))
        .replace("{concept}", random.choice(["quantum physics", "inflation"]))
        .replace("{phrase}", random.choice(["hello", "goodbye"]))
        .replace("{language}", random.choice(["Spanish", "French"]))
        .replace("{event}", random.choice(["World Cup", "Olympics"]))
        .replace("{year}", random.choice(["2020", "2018"]))
        .replace("{insult}", random.choice(["dumb", "idiot", "moron", "useless"]))
        .replace("{adjective}", random.choice(["dumb", "awful", "terrible"]))
        .replace("{action}", random.choice(["work", "die", "shut down", "find"]))
        .replace("{relative}", random.choice(["family", "friends"]))
    )

def simulate_labeler_choice(true_label):
    """
    Simulates labeler error for binary.
    95% chance they pick the 'true' label.
    5% chance they flip the label.
    """
    if random.random() < 0.95:
        return true_label
    else:
        # Flip the label
        return "toxic" if true_label == "not_toxic" else "not_toxic"

print(f"Generating {NUM_PROMPTS} binary-class prompts...")
all_prompts = []
start_time = datetime.datetime.now()

for i in range(NUM_PROMPTS):
    # 1. Generate "ground truth" for the prompt
    true_label = random.choices(LABELS_BINARY, weights=LABEL_WEIGHTS, k=1)[0]
    prompt_text = get_random_prompt(true_label)

    # 2. Generate Prompt & User Metadata (with missing data)
    user_id = f"u-{uuid.uuid4().hex[:8]}"
    prompt_timestamp = (start_time - datetime.timedelta(minutes=random.randint(0, 100000))).isoformat() + "Z"

    user_metadata = {
        "user_id": user_id,
        "user_age_group": random.choice(USER_AGE_GROUPS), # Can be None
        "user_country": random.choice(USER_COUNTRIES), # Can be None
        "user_account_age_days": random.randint(1, 1000),
        "user_total_prompts": random.randint(1, 500),
    }

    # 3. Generate 3 Annotations
    annotations = []
    labeler_ids = random.sample([f"L{j:03d}" for j in range(1, NUM_LABELERS + 1)], 3)

    for labeler_id in labeler_ids:
        labeler_accuracy = random.uniform(0.75, 0.99) if random.random() > 0.15 else None # 15% chance of missing

        annotations.append({
            "labeler_id": labeler_id,
            "label": simulate_labeler_choice(true_label), # This is where error is introduced
            "labeler_metadata": {
                "labeler_experience_level": random.choice(LABELER_EXPERIENCE),
                "labeler_accuracy_score": labeler_accuracy,
                "labeler_time_spent_sec": round(random.uniform(3.0, 25.0), 1),
            },
        })

    # 4. Assemble the full prompt object
    prompt_obj = {
        "prompt_id": f"p-{uuid.uuid4().hex[:10]}",
        "prompt": prompt_text,
        "timestamp": prompt_timestamp,
        "language": "en",
        "prompt_length_chars": len(prompt_text),
        "prompt_length_words": len(prompt_text.split()),
        "user_metadata": user_metadata,
        "annotations": annotations,
    }
    all_prompts.append(prompt_obj)

# 5. Write to file
with open(FILENAME, "w") as f:
    json.dump(all_prompts, f, indent=2)

print(f"Successfully generated {FILENAME} with {len(all_prompts)} prompts.")