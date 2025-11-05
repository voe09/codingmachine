import json
import random
import datetime
import uuid

# --- Configuration ---
NUM_PROMPTS = 10000
NUM_LABELERS = 100
FILENAME = "hard_multi_class_data.json"

# 1. Define Labels
LABELS_MULTICLASS = ["not_toxic", "mildly_toxic", "moderately_toxic", "highly_toxic", "severely_toxic"]
LABEL_WEIGHTS = [0.80, 0.10, 0.05, 0.03, 0.02]  # Slightly more toxic prompts

# 2. Define Disagreement Patterns (This is the new "harder" part)
# We will force disagreement patterns
DISAGREEMENT_PATTERNS = ["unanimous", "split", "total_disagreement"]
DISAGREEMENT_WEIGHTS = [0.60, 0.25, 0.15]  # 15% will have 1-1-1 split

# 3. Define Metadata Options (with None for missing data)
USER_COUNTRIES = ["US", "CA", "UK", "IN", "AU", "DE", "FR", "JP", None]
USER_AGE_GROUPS = ["18-25", "26-35", "36-45", "46+", None]
LABELER_EXPERIENCE = ["beginner", "intermediate", "expert"]

# 4. Simple prompt templates (same as before)
PROMPT_TEMPLATES = {
    "not_toxic": [
        "How do I bake a {food}?", "What is the capital of {country}?", "Explain {concept} in simple terms.",
    ],
    "mildly_toxic": [
        "You are a {insult} machine.", "That's a {adjective} stupid idea.", "Why can't you {action} right?",
    ],
    "moderately_toxic": [
        "You're completely {insult} useless.", "I can't believe how {insult} you are.",
        "Everyone thinks you are a {insult}."
    ],
    "highly_toxic": [
        "You are a {slur} and should be {action}.", "I'm going to {action} you and your {relative}.",
    ],
    "severely_toxic": [
        "I'm going to find where you {action} and {violent_action} you.",
        "I will {violent_action} your entire {group}."
    ]
}

# Helper to get a random prompt
def get_random_prompt(label):
    template = random.choice(PROMPT_TEMPLATES.get(label, PROMPT_TEMPLATES["not_toxic"]))
    return (
        template.replace("{food}", random.choice(["bread", "cake", "cookies"]))
        .replace("{country}", random.choice(["France", "Japan", "Brazil"]))
        .replace("{concept}", random.choice(["quantum physics", "inflation"]))
        .replace("{insult}", random.choice(["dumb", "idiot", "moron", "useless"]))
        .replace("{adjective}", random.choice(["dumb", "awful", "terrible"]))
        .replace("{action}", random.choice(["work", "die", "shut down"]))
        .replace("{slur}", random.choice(["[slur_A]", "[slur_B]"]))
        .replace("{relative}", random.choice(["family", "friends"]))
        .replace("{violent_action}", random.choice(["hunt", "find", "hurt"]))
        .replace("{group}", random.choice(["[group_A]", "[group_B]"]))
    )

def get_different_label(avoid_labels: list):
    """Helper to get a label that is NOT in the avoid_labels list."""
    possible_labels = [label for label in LABELS_MULTICLASS if label not in avoid_labels]
    if not possible_labels:  # Fallback just in case
        return random.choice(LABELS_MULTICLASS)
    return random.choice(possible_labels)

print(f"Generating {NUM_PROMPTS} 'hard' multi-class prompts...")
all_prompts = []
start_time = datetime.datetime.now()

for i in range(NUM_PROMPTS):
    # 1. Generate "ground truth" for the prompt
    true_label = random.choices(LABELS_MULTICLASS, weights=LABEL_WEIGHTS, k=1)[0]
    prompt_text = get_random_prompt(true_label)

    # 2. Generate Prompt & User Metadata
    user_id = f"u-{uuid.uuid4().hex[:8]}"
    prompt_timestamp = (start_time - datetime.timedelta(minutes=random.randint(0, 100000))).isoformat() + "Z"
    user_metadata = {
        "user_id": user_id,
        "user_age_group": random.choice(USER_AGE_GROUPS),
        "user_country": random.choice(USER_COUNTRIES),
        "user_account_age_days": random.randint(1, 1000),
        "user_total_prompts": random.randint(1, 500),
    }

    # 3. Generate 3 Annotations based on "hard" disagreement
    annotations = []
    labels_for_this_prompt = []
    labeler_ids = random.sample([f"L{j:03d}" for j in range(1, NUM_LABELERS + 1)], 3)
    
    # Choose a disagreement pattern
    pattern = random.choices(DISAGREEMENT_PATTERNS, weights=DISAGREEMENT_WEIGHTS, k=1)[0]
    
    if pattern == "unanimous":
        # All 3 labelers get the true label (with a tiny chance of error)
        l = true_label if random.random() < 0.98 else get_different_label([true_label])
        labels_for_this_prompt = [l, l, l]
        
    elif pattern == "split":
        # 2 agree, 1 disagrees
        disagreeing_label = get_different_label([true_label])
        labels_for_this_prompt = [true_label, true_label, disagreeing_label]
        
    elif pattern == "total_disagreement":
        # 1-1-1 split. This is the "hard" case.
        label_1 = true_label
        label_2 = get_different_label([label_1])
        label_3 = get_different_label([label_1, label_2])
        labels_for_this_prompt = [label_1, label_2, label_3]

    # Shuffle labels so the "true" one isn't always first
    random.shuffle(labels_for_this_prompt)
    
    for j in range(3):
        labeler_accuracy = random.uniform(0.75, 0.99) if random.random() > 0.15 else None
        annotations.append({
            "labeler_id": labeler_ids[j],
            "label": labels_for_this_prompt[j],
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
print(f"Disagreement pattern: 60% Unanimous, 25% Split (2-1), 15% Total Disagreement (1-1-1)")