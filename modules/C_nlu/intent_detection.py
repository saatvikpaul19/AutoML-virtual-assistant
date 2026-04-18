"""
intent_detection.py — Regex-Based Intent Classifier (Fallback)
================================================================
Used ONLY when BERT NLU confidence is below threshold or critical
slots are missing. This is the safety net, not the primary path.

All intent sets are exported so CommandRouter can do type checks.
"""
from __future__ import annotations

import re

# ── Intent category sets ───────────────────────────────────────
STATEFUL_INTENTS = {
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
    "resume_training",
    "stop_training",
    "show_status",
    "show_accuracy",
    "show_loss_curve",
    "tell_results",
    "set_activation",
    "split_dataset",
    "clean_dataset",
    "download_weights",

    # Timer
    "set_timer",
    "check_timer",
    "pause_timer",
    "resume_timer",
    "stop_timer",
    "restart_timer",
    "reset_timer",
    "add_time_to_timer",
    "cancel_timer",
}

STATELESS_INTENTS = {
    "search_dataset",
    "get_dataset_info",
    "show_competition",
    "show_leaderboard",
    "search_code",
    "suggest_model",
    "suggest_hyperparameters",
    "get_weather",
}

UTILITY_INTENTS = {
    "help",
    "repeat",
    "greetings",
    "farewell",
    "thanks",      # NEW
    "sleep_va",    # NEW
    "out_of_scope",
}

MODEL_KEYWORDS = {
    "xgboost":              "xgboost",
    "xgb":                  "xgboost",
    "random forest":        "random_forest",
    "random_forest":        "random_forest",
    "rf":                   "random_forest",
    "logistic regression":  "logistic_regression",
    "logistic_regression":  "logistic_regression",
    "cnn":                  "cnn",
    "mlp":                  "mlp",
    "resnet":               "resnet",
}

# Common ASR transcription errors → corrections
_ASR_CORRECTIONS = [
    (r"\bx\s*g\s*boost\b",      "xgboost"),
    (r"\blearning\s+great\b",   "learning rate"),
    (r"\bbatch\s+eyes\b",       "batch size"),
    (r"\bepics?\b",             "epochs"),
    (r"\bep[io]cks?\b",         "epochs"),
    (r"\bmicro\s*soft\b",       "mycroft"),
]


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    t = text.strip()
    if not t:
        return ""
    t = t.lower()
    for pattern, replacement in _ASR_CORRECTIONS:
        t = re.sub(pattern, replacement, t)
    t = re.sub(r"(?<!\d)\.(?!\d)", " ", t)
    t = re.sub(r"[^\w\s.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_stateful_intent(intent: str) -> bool:
    return intent in STATEFUL_INTENTS


def is_stateless_intent(intent: str) -> bool:
    return intent in STATELESS_INTENTS


def is_utility_intent(intent: str) -> bool:
    return intent in UTILITY_INTENTS


def get_intent_category(intent: str) -> str:
    if intent in STATEFUL_INTENTS:  return "stateful"
    if intent in STATELESS_INTENTS: return "stateless"
    if intent in UTILITY_INTENTS:   return "utility"
    return "unknown"


def get_supported_intents() -> list[str]:
    return sorted(STATEFUL_INTENTS | STATELESS_INTENTS | UTILITY_INTENTS)


def detect_intent(text: str) -> str:
    """Regex-based intent detection — safety net when BERT is not confident."""
    t = normalize_text(text)

    if not t:
        return "out_of_scope"

    # ── Utility ───────────────────────────────────────────────
    if t in {"help", "what can you do", "show help", "list commands", "show commands"}:
        return "help"

    if t in {"repeat", "say that again", "say again", "pardon", "come again", "what did you say"}:
        return "repeat"

    # Greetings
    if t in {
        "hello", "hi", "hey", "good morning", "good evening", "good afternoon",
        "greetings", "hi mycroft", "hello mycroft", "hey mycroft", "morning",
        "sup", "howdy", "what is up",
    }:
        return "greetings"

    # Thanks — NEW
    if any(p in t for p in [
        "thank you", "thanks a lot", "thanks", "i appreciate it",
        "appreciate it", "many thanks", "cheers", "that is helpful",
    ]):
        return "thanks"

    # Sleep VA — NEW
    if any(p in t for p in [
        "go to sleep", "stop listening", "be quiet", "sleep mode",
        "pause listening", "back to sleep", "turn off mic",
        "stop the microphone",
    ]) or t in {"sleep", "quiet", "mute", "standby", "rest"}:
        return "sleep_va"

    # Farewell
    if t in {
        "bye", "good bye", "goodbye", "see ya", "see you later",
        "good night", "farewell", "bye mycroft", "goodbye mycroft",
        "good night mycroft", "take care", "talk later", "catch you later",
    }:
        return "farewell"

    # ── Stateless ─────────────────────────────────────────────
    if "leaderboard" in t:
        return "show_leaderboard"

    if re.search(r"\bcompetition", t):
        return "show_competition"

    if any(p in t for p in ["search code", "find code", "search notebook",
                             "find notebook", "search kernel"]):
        return "search_code"

    if any(p in t for p in [
        "dataset info", "about dataset", "files for dataset",
        "describe dataset", "describe the dataset",
        "info on dataset", "info about", "tell me about",
        "how many rows", "how many columns", "rows and columns",
    ]):
        return "get_dataset_info"

    if any(p in t for p in [
        "search dataset", "find dataset", "search data", "find data",
        "search for dataset", "search for data", "browse dataset",
        "look for dataset", "look for data",
    ]):
        return "search_dataset"

    # Weather — also matches conditions like "does it rain in ..."
    if any(k in t for k in ["weather", "forecast", "temperature",
                             "rain", "raining", "snow", "sunny", "cloudy",
                             "does it rain", "will it rain"]):
        if any(clue in t for clue in [" in ", " today", " tomorrow",
                                       "weather", "forecast", "does it", "is it", "will it"]):
            return "get_weather"

    if any(k in t for k in ["suggest model", "suggest a model", "model suggestion",
                             "suggest me a model", "what model", "recommend a model",
                             "suggestions for models", "suggest some models", "recommend models"]):
        return "suggest_model"

    if any(k in t for k in ["suggest hyperparameters", "suggest parameters", "parameter suggestion",
                             "what hyperparameters", "recommend hyperparameters"]):
        return "suggest_hyperparameters"
        
    if any(k in t for k in ["code suggestion", "suggest code"]):
        return "search_code"

    # ── Stateful: timer (order matters) ──────────────────────
    if any(k in t for k in ["add", "extend", "increase", "plus"]) and "timer" in t and any(
        u in t for u in ["second", "seconds", "minute", "minutes", "hour", "hours", "sec", "min", "hr"]
    ):
        return "add_time_to_timer"

    if "timer" in t and any(v in t for v in ["pause", "hold"]):
        return "pause_timer"

    if "timer" in t and any(v in t for v in ["resume", "continue", "unpause"]):
        return "resume_timer"

    if "timer" in t and "restart" in t:
        return "restart_timer"

    if "timer" in t and "reset" in t:
        return "reset_timer"

    if "timer" in t and any(v in t for v in ["cancel", "clear", "delete", "remove"]):
        return "cancel_timer"

    if "timer" in t and "stop" in t:
        return "stop_timer"

    if "timer" in t and any(v in t for v in ["status", "remaining", "left", "check", "show"]):
        return "check_timer"

    if (("timer" in t or "countdown" in t or "alarm" in t) and any(
        u in t for u in ["second", "seconds", "minute", "minutes", "hour", "hours", "sec", "min", "hr"]
    )):
        return "set_timer"

    # ── Stateful: experiment ──────────────────────────────────
    if any(v in t for v in ["load", "retrieve", "fetch", "download", "get", "pull"]) and any(
        k in t for k in ["code", "notebook", "script"]
    ):
        return "load_code"

    if any(v in t for v in ["run", "execute"]) and any(k in t for k in ["code", "experiment"]):
        return "run_code"

    if "show output" in t or "display output" in t:
        return "show_output"

    if "learning rate" in t or re.search(r"\blr\b", t):
        return "set_learning_rate"

    if "batch size" in t:
        return "set_batch_size"

    if re.search(r"\bepochs?\b", t) and any(w in t for w in ["set", "change", "to", "for", "train"]):
        return "set_epochs"

    if "set layers" in t or "change layers" in t or "layers to" in t:
        return "set_layers"

    if any(k in t for k in ["set activation", "update activation", "activation function"]):
        return "set_activation"

    if any(k in t for k in [
        "clean dataset", "clean data", "handle missing", "preprocess data",
        "analyze dataset",
    ]):
        return "clean_dataset"

    if "split" in t and ("dataset" in t or "data" in t):
        return "split_dataset"

    if any(k in t for k in ["download weights", "download model"]):
        return "download_weights"

    if any(k in t for k in ["show results", "tell results", "show the results",
                             "tell the results", "display results"]):
        return "tell_results"

    if any(p in t for p in [
        "start training", "begin training", "begin the training",
        "run training", "launch training", "train the model", "start the training",
    ]):
        return "start_training"

    if any(p in t for p in [
        "resume training", "resume the training",
        "continue training", "continue the training",
        "unpause training",
    ]):
        return "resume_training"

    if any(p in t for p in ["pause training", "pause the training",
                             "hold training", "hold the training"]):
        return "pause_training"

    if any(p in t for p in [
        "stop training", "stop the training", "cancel training",
        "abort training",
    ]):
        return "stop_training"

    if any(p in t for p in [
        "show status", "check status", "current status", "training status",
        "what is the status", "is training done", "is it done", "training progress",
    ]):
        return "show_status"

    if any(p in t for p in ["show accuracy", "current accuracy",
                             "what is the accuracy", "how accurate"]):
        return "show_accuracy"

    if any(p in t for p in ["show loss curve", "plot loss", "loss curve",
                             "loss history", "show loss history"]):
        return "show_loss_curve"

    if any(m in t for m in MODEL_KEYWORDS):
        return "select_model"

    if re.search(r"\b(load|import|open|retrieve|fetch|use|download|get|grab|pull)\b", t) and not any(
        k in t for k in ["code", "notebook", "script", "model"]
    ):
        return "load_dataset"

    return "out_of_scope"