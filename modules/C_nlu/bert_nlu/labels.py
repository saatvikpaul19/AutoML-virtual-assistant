"""
labels.py — BERT NLU Label Registry
=====================================
Single source of truth for all intent and slot labels.
intent2id / id2intent and slot2id / id2slot are auto-built from
the lists — adding a new intent/slot here is all that is needed.

IMPORTANT: Changing this file changes the model's output space.
The model MUST be retrained after any additions.
"""

# ── Intent Labels ─────────────────────────────────────────────
#  Order matters — index == class ID used by the model.
#  Always APPEND to the end when adding new intents so that
#  existing model weights remain valid for the already-trained classes.

INTENTS = [
    # ── Stateful: Experiment Control ──────────────────────────
    "load_dataset",
    "select_model",
    "set_learning_rate",
    "set_batch_size",
    "set_epochs",
    "set_layers",
    "load_code",
    "run_code",
    "show_output",
    "start_training",
    "pause_training",
    "stop_training",
    "resume_training",
    "show_status",
    "show_accuracy",
    "show_loss_curve",
    "tell_results",
    "set_activation",
    "clean_dataset",
    "split_dataset",
    "download_weights",

    # ── Stateful: Timer ───────────────────────────────────────
    "set_timer",
    "check_timer",
    "cancel_timer",

    # ── Stateless: Kaggle / Info ──────────────────────────────
    "search_dataset",
    "get_dataset_info",
    "show_competition",
    "show_leaderboard",
    "search_code",
    "suggest_model",
    "suggest_hyperparameters",
    "get_weather",

    # ── Utility ───────────────────────────────────────────────
    "help",
    "repeat",
    "greetings",
    "farewell",
    "out_of_scope",

    # ── New Utility Intents (appended to preserve existing IDs) ─
    "thanks",       # "Thank you", "Thanks a lot", "I appreciate it"
    "sleep_va",     # "Sleep", "Go to sleep", "Stop listening", "Be quiet"
]

# ── Slot Labels ───────────────────────────────────────────────
#  BIO tagging scheme: O | B-<TYPE> | I-<TYPE>
#  slot_labels list is auto-built from SLOT_TYPES below.

SLOT_TYPES = [
    # ML Experiment
    "DATASET_NAME",
    "DATASET_PATH",
    "MODEL_NAME",
    "LEARNING_RATE",
    "BATCH_SIZE",
    "EPOCHS",
    "LAYERS",
    "ACTIVATION",
    "FILE_PATH",
    "CODE_TARGET",
    "OUTPUT_TARGET",

    # Timer
    "TIMER_DURATION",
    "TIMER_TIME",
    "TIMER_NAME",

    # Weather
    "WEATHER_LOCATION",
    "WEATHER_CONDITION",   # NEW: "rain", "sunny", "snow"
    "DATE",                # NEW: "today", "tomorrow", "2024-01-15"

    # Kaggle / Search
    "COMPETITION_NAME",
    "METRIC_NAME",
    "QUERY",

    # Dataset split
    "SPLIT_RATIO",
    "TRAIN_RATIO",
    "VAL_RATIO",
    "TEST_RATIO",

    # Out-of-scope capture
    "ORIGINAL_REQUEST",    # NEW: the original verb phrase for out_of_scope

    # Misc
    "RESULT_TYPE",
]


def get_slot_labels() -> list[str]:
    """Return full BIO label list: [O, B-SLOT1, I-SLOT1, B-SLOT2, I-SLOT2, ...]"""
    labels = ["O"]
    for slot in SLOT_TYPES:
        labels.append(f"B-{slot}")
        labels.append(f"I-{slot}")
    return labels


# ── Auto-built lookup tables ──────────────────────────────────
intent2id = {intent: i for i, intent in enumerate(INTENTS)}
id2intent  = {i: intent for intent, i in intent2id.items()}

slot_labels = get_slot_labels()
slot2id     = {label: i for i, label in enumerate(slot_labels)}
id2slot     = {i: label for label, i in slot2id.items()}


if __name__ == "__main__":
    print(f"Total intents : {len(INTENTS)}")
    print(f"Total slot types: {len(SLOT_TYPES)}")
    print(f"Total slot labels (BIO): {len(slot_labels)}")
    print("\nIntent → ID:")
    for intent, idx in intent2id.items():
        print(f"  [{idx:2d}] {intent}")
    print("\nSlot type count (each has B- and I- prefix):")
    for st in SLOT_TYPES:
        print(f"  {st}")
