"""
data_generation.py — BERT NLU Training Data Generator
=======================================================
Generates diverse, properly token-labeled training data for the
JointIntentSlotModel (DistilBERT joint intent + slot filling).

Run this script from the project root:
    python -m modules.C_nlu.bert_nlu.data.data_generation

Output files (saved to bert_nlu/data/):
    train.json   (~80% of data)
    val.json     (~10%)
    test.json    (~10%)

Each sample:
    {
        "intent":      str,
        "text":        str,
        "tokens":      [str, ...],
        "slot_labels": [str, ...]   # BIO scheme, same length as tokens
    }
"""
import json
import os
import random

random.seed(42)

# ─────────────────────────────────────────────────────────────
# Slot value pools
# ─────────────────────────────────────────────────────────────
DATASETS   = ["mnist", "cifar10", "iris", "titanic", "boston", "diabetes", "wine", "breast cancer"]
MODELS     = ["xgboost", "random forest", "cnn", "mlp", "resnet", "logistic regression"]
LRS        = ["0.01", "0.001", "0.0001", "1e-4", "5e-5", "0.05", "0.1"]
BATCHES    = ["16", "32", "64", "128", "8", "256"]
EPOCHS_V   = ["5", "10", "15", "20", "30", "50", "100"]
LOCATIONS  = ["Ottawa", "Toronto", "New York", "London", "Paris", "Tokyo", "Berlin", "Sydney", "Montreal", "Vancouver"]
CONDITIONS = ["rain", "snow", "sunny", "cloudy", "fog", "storm"]
DATES      = ["today", "tomorrow"]
DURATIONS  = [("10", "minutes"), ("5", "minutes"), ("2", "minutes"), ("30", "seconds"),
              ("1", "hour"), ("45", "minutes"), ("15", "minutes"), ("3", "minutes")]
TIMER_NAMES = ["cooking", "soup", "laundry", "workout", "meditation", "pasta", "meeting", "break", "coffee"]
OOS_ACTIONS = [
    ("buy", "groceries"), ("do", "shopping"), ("call", "mom"), ("order", "pizza"),
    ("book", "a flight"), ("send", "an email"), ("play", "music"), ("turn on", "the lights"),
    ("clean", "the house"), ("wash", "the dishes"), ("walk", "the dog"), ("make", "coffee"),
    ("buy", "tickets"), ("book", "a table"), ("find", "a plumber"),
]
SEARCH_QUERIES = ["fraud detection", "image classification", "sentiment analysis",
                  "house prices", "customer churn", "stock prediction", "nlp text classification"]

# ─────────────────────────────────────────────────────────────
# Helper: build a labeled sample from a pre-tagged token list
# ─────────────────────────────────────────────────────────────

def _sample(intent: str, tokens: list[str], labels: list[str]) -> dict:
    assert len(tokens) == len(labels), f"Token/label mismatch for intent={intent}: {tokens} vs {labels}"
    return {
        "intent":      intent,
        "text":        " ".join(tokens),
        "tokens":      tokens,
        "slot_labels": labels,
    }


def _O(n: int) -> list[str]:
    return ["O"] * n

# ─────────────────────────────────────────────────────────────
# Generators per intent
# ─────────────────────────────────────────────────────────────

def gen_load_dataset(n: int) -> list[dict]:
    prefixes = [
        (["load"],),
        (["load", "the"],),
        (["use", "the"],),
        (["fetch"],),
        (["retrieve"],),
        (["import"],),
        (["open"],),
        (["load", "me", "the"],),
        (["use"],),
        (["get"],),
    ]
    suffixes = [
        (["dataset"],),
        (["data"],),
        (["dataset", "from", "kaggle"],),
        ([],),
    ]
    out = []
    for _ in range(n):
        ds     = random.choice(DATASETS)
        ds_tok = ds.split()
        pre    = list(random.choice(prefixes)[0])
        suf    = list(random.choice(suffixes)[0])
        tokens = pre + ds_tok + suf
        labels = _O(len(pre)) + ["B-DATASET_NAME"] + ["I-DATASET_NAME"] * (len(ds_tok) - 1) + _O(len(suf))
        out.append(_sample("load_dataset", tokens, labels))
    return out


def gen_select_model(n: int) -> list[dict]:
    prefixes = [
        ["select"],
        ["use"],
        ["choose"],
        ["switch", "to"],
        ["i", "want", "to", "use"],
        ["set", "model", "to"],
        ["train", "with"],
        ["apply"],
        ["run"],
        ["try"],
    ]
    suffixes = [
        [],
        ["model"],
        ["as", "my", "model"],
        ["for", "training"],
    ]
    out = []
    for _ in range(n):
        m      = random.choice(MODELS)
        m_tok  = m.split()
        pre    = list(random.choice(prefixes))
        suf    = list(random.choice(suffixes))
        tokens = pre + m_tok + suf
        labels = _O(len(pre)) + ["B-MODEL_NAME"] + ["I-MODEL_NAME"] * (len(m_tok) - 1) + _O(len(suf))
        out.append(_sample("select_model", tokens, labels))
    return out


def gen_set_learning_rate(n: int) -> list[dict]:
    templates = [
        lambda lr: (["set", "learning", "rate", "to", lr], _O(4) + ["B-LEARNING_RATE"]),
        lambda lr: (["change", "lr", "to", lr], _O(3) + ["B-LEARNING_RATE"]),
        lambda lr: (["use", lr, "for", "lr"], ["O", "B-LEARNING_RATE", "O", "O"]),
        lambda lr: (["update", "learning", "rate", "to", lr], _O(4) + ["B-LEARNING_RATE"]),
        lambda lr: (["set", "lr", "=", lr], _O(3) + ["B-LEARNING_RATE"]),
        lambda lr: (["learning", "rate", lr], _O(2) + ["B-LEARNING_RATE"]),
        lambda lr: (["i", "want", "lr", lr], _O(3) + ["B-LEARNING_RATE"]),
        lambda lr: (["set", "the", "learning", "rate", "to", lr], _O(5) + ["B-LEARNING_RATE"]),
    ]
    out = []
    for _ in range(n):
        lr     = random.choice(LRS)
        t, l   = random.choice(templates)(lr)
        out.append(_sample("set_learning_rate", t, l))
    return out


def gen_set_batch_size(n: int) -> list[dict]:
    templates = [
        lambda b: (["set", "batch", "size", "to", b], _O(4) + ["B-BATCH_SIZE"]),
        lambda b: (["use", "batch", "size", b], _O(3) + ["B-BATCH_SIZE"]),
        lambda b: (["batch", "size", b], _O(2) + ["B-BATCH_SIZE"]),
        lambda b: (["change", "batch", "size", "to", b], _O(4) + ["B-BATCH_SIZE"]),
        lambda b: (["set", "the", "batch", "to", b], _O(4) + ["B-BATCH_SIZE"]),
        lambda b: (["batch", b], ["O", "B-BATCH_SIZE"]),
        lambda b: (["update", "batch", "size", "to", b], _O(4) + ["B-BATCH_SIZE"]),
    ]
    out = []
    for _ in range(n):
        b    = random.choice(BATCHES)
        t, l = random.choice(templates)(b)
        out.append(_sample("set_batch_size", t, l))
    return out


def gen_set_epochs(n: int) -> list[dict]:
    templates = [
        lambda e: (["train", "for", e, "epochs"], _O(2) + ["B-EPOCHS", "O"]),
        lambda e: (["set", "epochs", "to", e], _O(3) + ["B-EPOCHS"]),
        lambda e: (["change", "epochs", "to", e], _O(3) + ["B-EPOCHS"]),
        lambda e: (["run", e, "epochs"], ["O", "B-EPOCHS", "O"]),
        lambda e: (["epochs", e], ["O", "B-EPOCHS"]),
        lambda e: (["i", "want", e, "epochs"], _O(2) + ["B-EPOCHS", "O"]),
        lambda e: (["set", "the", "number", "of", "epochs", "to", e], _O(6) + ["B-EPOCHS"]),
        lambda e: (["train", e, "epochs"], ["O", "B-EPOCHS", "O"]),
    ]
    out = []
    for _ in range(n):
        e    = random.choice(EPOCHS_V)
        t, l = random.choice(templates)(e)
        out.append(_sample("set_epochs", t, l))
    return out


def gen_start_training(n: int) -> list[dict]:
    variants = [
        (["start", "training"], _O(2)),
        (["begin", "training"], _O(2)),
        (["train", "the", "model"], _O(3)),
        (["launch", "training"], _O(2)),
        (["begin", "the", "training"], _O(3)),
        (["start", "the", "training"], _O(3)),
        (["run", "training"], _O(2)),
        (["go", "ahead", "and", "train"], _O(4)),
        (["start", "training", "now"], _O(3)),
        (["kick", "off", "training"], _O(3)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("start_training", t, l))
    return out


def gen_pause_training(n: int) -> list[dict]:
    variants = [
        (["pause", "training"], _O(2)),
        (["hold", "training"], _O(2)),
        (["pause", "the", "training"], _O(3)),
        (["stop", "training", "temporarily"], _O(3)),
        (["put", "training", "on", "hold"], _O(4)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("pause_training", t, l))
    return out


def gen_resume_training(n: int) -> list[dict]:
    variants = [
        (["resume", "training"], _O(2)),
        (["continue", "training"], _O(2)),
        (["unpause", "training"], _O(2)),
        (["resume", "the", "training"], _O(3)),
        (["continue", "the", "training"], _O(3)),
        (["keep", "going"], _O(2)),
        (["carry", "on", "with", "training"], _O(4)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("resume_training", t, l))
    return out


def gen_stop_training(n: int) -> list[dict]:
    variants = [
        (["stop", "training"], _O(2)),
        (["cancel", "training"], _O(2)),
        (["abort", "training"], _O(2)),
        (["halt", "training"], _O(2)),
        (["stop", "the", "training"], _O(3)),
        (["end", "training"], _O(2)),
        (["kill", "training"], _O(2)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("stop_training", t, l))
    return out


def gen_show_status(n: int) -> list[dict]:
    variants = [
        (["show", "status"], _O(2)),
        (["what", "is", "the", "status"], _O(4)),
        (["check", "status"], _O(2)),
        (["how", "is", "training", "going"], _O(4)),
        (["training", "progress"], _O(2)),
        (["is", "training", "done"], _O(3)),
        (["current", "status"], _O(2)),
        (["show", "training", "status"], _O(3)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("show_status", t, l))
    return out


def gen_show_accuracy(n: int) -> list[dict]:
    variants = [
        (["show", "accuracy"], _O(2)),
        (["what", "is", "the", "accuracy"], _O(4)),
        (["current", "accuracy"], _O(2)),
        (["how", "accurate", "is", "the", "model"], _O(5)),
        (["show", "me", "the", "accuracy"], _O(4)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("show_accuracy", t, l))
    return out


def gen_tell_results(n: int) -> list[dict]:
    variants = [
        (["show", "results"], _O(2)),
        (["tell", "me", "the", "results"], _O(4)),
        (["display", "results"], _O(2)),
        (["what", "are", "the", "results"], _O(4)),
        (["show", "me", "results"], _O(3)),
        (["give", "me", "the", "results"], _O(4)),
        (["tell", "results"], _O(2)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("tell_results", t, l))
    return out


def gen_load_code(n: int) -> list[dict]:
    variants = [
        (["load", "corresponding", "code"], _O(3)),
        (["load", "the", "code"], _O(3)),
        (["get", "the", "code"], _O(3)),
        (["fetch", "code"], _O(2)),
        (["generate", "code"], _O(2)),
        (["load", "notebook"], _O(2)),
        (["load", "the", "notebook"], _O(3)),
        (["show", "code"], _O(2)),
        (["get", "runnable", "code"], _O(3)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("load_code", t, l))
    return out


def gen_run_code(n: int) -> list[dict]:
    variants = [
        (["run", "code"], _O(2)),
        (["execute", "code"], _O(2)),
        (["run", "the", "code"], _O(3)),
        (["execute", "the", "experiment"], _O(3)),
        (["run", "experiment"], _O(2)),
        (["run", "it"], _O(2)),
        (["go"], _O(1)),
        (["execute"], _O(1)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("run_code", t, l))
    return out


def gen_clean_dataset(n: int) -> list[dict]:
    variants = [
        (["clean", "dataset"], _O(2)),
        (["clean", "the", "data"], _O(3)),
        (["handle", "missing", "values"], _O(3)),
        (["preprocess", "data"], _O(2)),
        (["analyze", "dataset"], _O(2)),
        (["clean", "the", "dataset"], _O(3)),
        (["fix", "missing", "data"], _O(3)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("clean_dataset", t, l))
    return out


def gen_search_dataset(n: int) -> list[dict]:
    templates = [
        lambda q: (["search", "dataset"] + q.split(), _O(2) + ["B-QUERY"] + ["I-QUERY"] * (len(q.split()) - 1)),
        lambda q: (["find", "dataset"] + q.split(), _O(2) + ["B-QUERY"] + ["I-QUERY"] * (len(q.split()) - 1)),
        lambda q: (["search", "for"] + q.split() + ["dataset"], _O(2) + ["B-QUERY"] + ["I-QUERY"] * (len(q.split()) - 1) + ["O"]),
        lambda q: (["look", "for"] + q.split(), _O(2) + ["B-QUERY"] + ["I-QUERY"] * (len(q.split()) - 1)),
    ]
    out = []
    for _ in range(n):
        q    = random.choice(SEARCH_QUERIES)
        t, l = random.choice(templates)(q)
        out.append(_sample("search_dataset", t, l))
    return out


def gen_suggest_model(n: int) -> list[dict]:
    variants = [
        (["suggest", "a", "model"], _O(3)),
        (["what", "model", "should", "i", "use"], _O(5)),
        (["recommend", "a", "model"], _O(3)),
        (["suggest", "me", "a", "model"], _O(4)),
        (["which", "model", "is", "best"], _O(4)),
        (["what", "model", "do", "you", "recommend"], _O(5)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("suggest_model", t, l))
    return out


def gen_suggest_hyperparameters(n: int) -> list[dict]:
    variants = [
        (["suggest", "hyperparameters"], _O(2)),
        (["what", "hyperparameters", "should", "i", "use"], _O(5)),
        (["recommend", "hyperparameters"], _O(2)),
        (["suggest", "parameters"], _O(2)),
        (["what", "are", "good", "hyperparameters"], _O(4)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("suggest_hyperparameters", t, l))
    return out


def gen_show_competition(n: int) -> list[dict]:
    variants = [
        (["show", "competitions"], _O(2)),
        (["list", "competitions"], _O(2)),
        (["what", "competitions", "are", "there"], _O(4)),
        (["show", "me", "competitions"], _O(3)),
        (["kaggle", "competitions"], _O(2)),
        (["find", "competitions"], _O(2)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("show_competition", t, l))
    return out


def gen_get_weather(n: int) -> list[dict]:
    """Weather with WEATHER_LOCATION (required), DATE (optional), WEATHER_CONDITION (optional)."""
    out = []
    for _ in range(n):
        loc     = random.choice(LOCATIONS)
        loc_tok = loc.split()
        L       = len(loc_tok)
        loc_labels = ["B-WEATHER_LOCATION"] + ["I-WEATHER_LOCATION"] * (L - 1)

        date = random.choice(DATES + [None, None])  # None = no date
        cond = random.choice(CONDITIONS + [None, None, None])  # None = no condition

        # Pattern: "what is the weather in <city> <date>"
        if cond is None and date is not None:
            r = random.randint(0, 2)
            if r == 0:
                t = ["what", "is", "the", "weather", "in"] + loc_tok + [date]
                l = _O(5) + loc_labels + ["B-DATE"]
            elif r == 1:
                t = ["weather", "in"] + loc_tok + [date]
                l = _O(2) + loc_labels + ["B-DATE"]
            else:
                t = ["forecast", "for"] + loc_tok + [date]
                l = _O(2) + loc_labels + ["B-DATE"]

        # Pattern with condition: "does it rain in <city> today"
        elif cond is not None and date is not None:
            r = random.randint(0, 1)
            if r == 0:
                t = ["does", "it", cond, "in"] + loc_tok + [date]
                l = _O(2) + ["B-WEATHER_CONDITION", "O"] + loc_labels + ["B-DATE"]
            else:
                t = ["is", "it", cond + "y", "in"] + loc_tok + [date]
                l = _O(2) + ["B-WEATHER_CONDITION", "O"] + loc_labels + ["B-DATE"]

        # Pattern with condition no date
        elif cond is not None:
            t = ["does", "it", cond, "in"] + loc_tok
            l = _O(2) + ["B-WEATHER_CONDITION", "O"] + loc_labels

        # Just location
        else:
            t = ["what", "is", "the", "weather", "in"] + loc_tok
            l = _O(5) + loc_labels

        out.append(_sample("get_weather", t, l))
    return out


def gen_set_timer(n: int) -> list[dict]:
    out = []
    for _ in range(n):
        num, unit = random.choice(DURATIONS)
        use_name  = random.random() < 0.4
        name      = random.choice(TIMER_NAMES) if use_name else None

        if name and random.random() < 0.5:
            # "set a <name> timer for <num> <unit>"
            t = ["set", "a", name, "timer", "for", num, unit]
            l = ["O", "O", "B-TIMER_NAME", "O", "O", "B-TIMER_DURATION", "I-TIMER_DURATION"]
        elif use_name:
            # "timer for <name> for <num> <unit>"
            t = ["timer", "for", name, "for", num, unit]
            l = ["O", "O", "B-TIMER_NAME", "O", "B-TIMER_DURATION", "I-TIMER_DURATION"]
        else:
            r = random.randint(0, 4)
            if r == 0:
                t = ["set", "a", "timer", "for", num, unit]
                l = _O(4) + ["B-TIMER_DURATION", "I-TIMER_DURATION"]
            elif r == 1:
                t = ["countdown", "for", num, unit]
                l = _O(2) + ["B-TIMER_DURATION", "I-TIMER_DURATION"]
            elif r == 2:
                t = ["start", "a", num, unit, "timer"]
                l = _O(2) + ["B-TIMER_DURATION", "I-TIMER_DURATION", "O"]
            elif r == 3:
                t = ["alarm", "for", num, unit]
                l = _O(2) + ["B-TIMER_DURATION", "I-TIMER_DURATION"]
            else:
                t = ["timer", num, unit]
                l = ["O", "B-TIMER_DURATION", "I-TIMER_DURATION"]

        out.append(_sample("set_timer", t, l))
    return out


def gen_check_timer(n: int) -> list[dict]:
    variants = [
        (["check", "timer"], _O(2)),
        (["how", "much", "time", "is", "left"], _O(5)),
        (["timer", "status"], _O(2)),
        (["how", "long", "remaining"], _O(3)),
        (["what", "is", "the", "timer", "at"], _O(5)),
        (["timer", "remaining"], _O(2)),
        (["show", "timer"], _O(2)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("check_timer", t, l))
    return out


def gen_cancel_timer(n: int) -> list[dict]:
    variants = [
        (["cancel", "timer"], _O(2)),
        (["stop", "timer"], _O(2)),
        (["clear", "timer"], _O(2)),
        (["delete", "timer"], _O(2)),
        (["remove", "the", "timer"], _O(3)),
        (["cancel", "the", "timer"], _O(3)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("cancel_timer", t, l))
    return out


def gen_greetings(n: int) -> list[dict]:
    variants = [
        (["hello"], _O(1)),
        (["hi"], _O(1)),
        (["hey"], _O(1)),
        (["hello", "mycroft"], _O(2)),
        (["hi", "mycroft"], _O(2)),
        (["hey", "mycroft"], _O(2)),
        (["good", "morning"], _O(2)),
        (["good", "afternoon"], _O(2)),
        (["good", "evening"], _O(2)),
        (["greetings"], _O(1)),
        (["morning"], _O(1)),
        (["sup"], _O(1)),
        (["howdy"], _O(1)),
        (["what", "is", "up"], _O(3)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("greetings", t, l))
    return out


def gen_farewell(n: int) -> list[dict]:
    variants = [
        (["bye"], _O(1)),
        (["goodbye"], _O(1)),
        (["see", "you", "later"], _O(3)),
        (["see", "ya"], _O(2)),
        (["farewell"], _O(1)),
        (["good", "night"], _O(2)),
        (["good", "bye"], _O(2)),
        (["bye", "bye"], _O(2)),
        (["take", "care"], _O(2)),
        (["talk", "later"], _O(2)),
        (["catch", "you", "later"], _O(3)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("farewell", t, l))
    return out


def gen_thanks(n: int) -> list[dict]:
    variants = [
        (["thank", "you"], _O(2)),
        (["thanks"], _O(1)),
        (["thanks", "a", "lot"], _O(3)),
        (["thank", "you", "very", "much"], _O(4)),
        (["i", "appreciate", "it"], _O(3)),
        (["appreciate", "it"], _O(2)),
        (["many", "thanks"], _O(2)),
        (["cheers"], _O(1)),
        (["that", "is", "helpful", "thanks"], _O(4)),
        (["great", "thank", "you"], _O(3)),
        (["perfect", "thanks"], _O(2)),
        (["awesome", "thank", "you"], _O(3)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("thanks", t, l))
    return out


def gen_sleep_va(n: int) -> list[dict]:
    variants = [
        (["sleep"], _O(1)),
        (["go", "to", "sleep"], _O(3)),
        (["stop", "listening"], _O(2)),
        (["be", "quiet"], _O(2)),
        (["quiet"], _O(1)),
        (["mute"], _O(1)),
        (["standby"], _O(1)),
        (["go", "standby"], _O(2)),
        (["rest"], _O(1)),
        (["pause", "listening"], _O(2)),
        (["stop", "the", "microphone"], _O(3)),
        (["turn", "off", "mic"], _O(3)),
        (["back", "to", "sleep"], _O(3)),
        (["sleep", "mode"], _O(2)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("sleep_va", t, l))
    return out


def gen_help(n: int) -> list[dict]:
    variants = [
        (["help"], _O(1)),
        (["what", "can", "you", "do"], _O(4)),
        (["show", "help"], _O(2)),
        (["commands"], _O(1)),
        (["show", "commands"], _O(2)),
        (["what", "do", "you", "support"], _O(4)),
        (["list", "commands"], _O(2)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("help", t, l))
    return out


def gen_repeat(n: int) -> list[dict]:
    variants = [
        (["repeat"], _O(1)),
        (["say", "that", "again"], _O(3)),
        (["repeat", "that"], _O(2)),
        (["pardon"], _O(1)),
        (["come", "again"], _O(2)),
        (["what", "did", "you", "say"], _O(4)),
        (["can", "you", "repeat", "that"], _O(4)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("repeat", t, l))
    return out


def gen_out_of_scope(n: int) -> list[dict]:
    """Out-of-scope with ORIGINAL_REQUEST slot covering the main verb phrase."""
    templates = [
        lambda v, o: (["can", "you"] + v.split() + o.split(),
                      _O(2) + ["B-ORIGINAL_REQUEST"] + ["I-ORIGINAL_REQUEST"] * (len(v.split() + o.split()) - 1)),
        lambda v, o: (v.split() + ["my"] + o.split(),
                      ["B-ORIGINAL_REQUEST"] + ["I-ORIGINAL_REQUEST"] * (len(v.split()) + len(o.split()))),
        lambda v, o: (["please"] + v.split() + o.split(),
                      ["O"] + ["B-ORIGINAL_REQUEST"] + ["I-ORIGINAL_REQUEST"] * (len(v.split() + o.split()) - 1)),
        lambda v, o: (["could", "you"] + v.split() + o.split(),
                      _O(2) + ["B-ORIGINAL_REQUEST"] + ["I-ORIGINAL_REQUEST"] * (len(v.split() + o.split()) - 1)),
        lambda v, o: (v.split() + o.split(),
                      ["B-ORIGINAL_REQUEST"] + ["I-ORIGINAL_REQUEST"] * (len(v.split() + o.split()) - 1)),
    ]
    out = []
    for _ in range(n):
        v, o = random.choice(OOS_ACTIONS)
        t, l = random.choice(templates)(v, o)
        out.append(_sample("out_of_scope", t, l))
    return out


def gen_download_weights(n: int) -> list[dict]:
    variants = [
        (["download", "weights"], _O(2)),
        (["download", "the", "model", "weights"], _O(4)),
        (["save", "the", "model"], _O(3)),
        (["get", "model", "weights"], _O(3)),
        (["export", "model"], _O(2)),
    ]
    out = []
    for _ in range(n):
        t, l = random.choice(variants)
        out.append(_sample("download_weights", t, l))
    return out


# ─────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────

def generate_dataset() -> list[dict]:
    """Generate the full dataset and return it."""
    all_samples: list[dict] = []

    # Each intent gets enough samples for good training coverage
    generators = [
        (gen_load_dataset,          50),
        (gen_select_model,          40),
        (gen_set_learning_rate,     40),
        (gen_set_batch_size,        35),
        (gen_set_epochs,            35),
        (gen_start_training,        35),
        (gen_pause_training,        25),
        (gen_resume_training,       25),
        (gen_stop_training,         25),
        (gen_show_status,           25),
        (gen_show_accuracy,         20),
        (gen_tell_results,          25),
        (gen_load_code,             30),
        (gen_run_code,              25),
        (gen_clean_dataset,         20),
        (gen_search_dataset,        30),
        (gen_suggest_model,         20),
        (gen_suggest_hyperparameters, 20),
        (gen_show_competition,      20),
        (gen_get_weather,           50),   # more weather samples → better slot extraction
        (gen_set_timer,             50),   # more timer samples → better DURATION/NAME slots
        (gen_check_timer,           25),
        (gen_cancel_timer,          20),
        (gen_greetings,             40),
        (gen_farewell,              30),
        (gen_thanks,                35),   # NEW
        (gen_sleep_va,              30),   # NEW
        (gen_help,                  20),
        (gen_repeat,                20),
        (gen_out_of_scope,          40),   # NEW: OOS with ORIGINAL_REQUEST
        (gen_download_weights,      15),
    ]

    for gen_fn, count in generators:
        samples = gen_fn(count)
        all_samples.extend(samples)
        print(f"  Generated {len(samples):3d} samples for  {samples[0]['intent']}")

    print(f"\n  Total samples: {len(all_samples)}")
    return all_samples


def save_splits(samples: list[dict], output_dir: str) -> None:
    """Shuffle and split 80/10/10, then save JSON files."""
    random.shuffle(samples)
    n          = len(samples)
    train_end  = int(0.80 * n)
    val_end    = int(0.90 * n)

    splits = {
        "train": samples[:train_end],
        "val":   samples[train_end:val_end],
        "test":  samples[val_end:],
    }

    os.makedirs(output_dir, exist_ok=True)
    for name, data in splits.items():
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(data):4d} samples -> {path}")


if __name__ == "__main__":
    print("Generating BERT NLU training dataset …\n")
    samples    = generate_dataset()
    output_dir = os.path.dirname(os.path.abspath(__file__))   # bert_nlu/data/
    print("\nSaving splits …")
    save_splits(samples, output_dir)
    print("\nDone. Run training with:")
    print("  python -m modules.C_nlu.bert_nlu.train")
