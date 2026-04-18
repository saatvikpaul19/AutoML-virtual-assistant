"""
slot_filling.py — Regex-Based Slot Extractor (Fallback)
=========================================================
Used ONLY when BERT NLU confidence is below threshold or BERT
missed a critical slot. This is the safety net.
"""
from __future__ import annotations

import re

from modules.C_nlu.intent_detection import MODEL_KEYWORDS, normalize_text

_KNOWN_DATASETS = {
    "titanic":       "titanic",
    "iris":          "iris",
    "mnist":         "mnist",
    "digits":        "mnist",
    "cifar10":       "cifar10",
    "cifar 10":      "cifar10",
    "cifar-10":      "cifar10",
    "boston":        "boston",
    "wine":          "wine",
    "diabetes":      "diabetes",
    "breast cancer": "breast_cancer",
    "breast_cancer": "breast_cancer",
}

# Weather condition keywords
_WEATHER_CONDITIONS = [
    "raining", "rain", "snow", "snowing", "sunny", "cloudy",
    "foggy", "fog", "stormy", "storm", "hailing", "hail",
    "windy", "wind", "clear",
]

# Out-of-scope modal verb prefixes to strip before extracting original_request
_OOS_MODAL_PREFIXES = re.compile(
    r"^(can you|could you|please|would you|help me|i need you to|i want you to)\s+",
    re.IGNORECASE,
)


# ── Internal helpers ──────────────────────────────────────────

def _extract_first_number(text: str):
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    return m.group(1) if m else None


def _try_known_dataset(text: str) -> str | None:
    t = text.lower().strip()
    for name in sorted(_KNOWN_DATASETS.keys(), key=len, reverse=True):
        if name in t:
            return _KNOWN_DATASETS[name]
    return None


def _extract_dataset_phrase(text: str, for_code: bool = False) -> str | None:
    t = text.lower().strip()

    known = _try_known_dataset(t)
    if known:
        return known

    patterns = [
        r"(?:load|retrieve|fetch|use|import|open)\s+(?:the\s+)?(.+?)(?:\s+dataset|\s+data)?\s*$",
        r"(?:get dataset info|tell me about|about dataset|describe dataset|describe the dataset)\s+(.+)$",
        r"(?:search dataset|find dataset|search data|find data)\s+(.+)$",
    ]

    if for_code:
        patterns = [
            r"(?:load|retrieve|fetch)\s+(.+?)\s+(?:code|notebook|script)$",
            r"(?:load corresponding code(?: for)?\s*)(.*)$",
        ] + patterns

    for pattern in patterns:
        m = re.search(pattern, t)
        if m:
            value = m.group(1).strip(" ,:-")
            value = re.sub(r"\s+from\s+kaggl?e[l]?\s*$", "", value)
            value = re.sub(r"\s+dataset\s*$", "", value)
            value = re.sub(r"\s+data\s*$", "", value)
            value = value.strip(" ,:-")
            if value and value != "corresponding":
                known2 = _try_known_dataset(value)
                return known2 or value

    return None


def _extract_timer_duration_seconds(text: str) -> int | None:
    total = 0
    found = False

    for value, unit in re.findall(
        r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|minutes?|mins?|min|seconds?|secs?|sec)",
        text,
    ):
        found = True
        amount = float(value)
        unit = unit.lower()

        if unit.startswith(("hour", "hr")):
            total += int(amount * 3600)
        elif unit.startswith(("minute", "min")):
            total += int(amount * 60)
        else:
            total += int(amount)

    return total if found and total > 0 else None


def _extract_timer_label(text: str) -> str | None:
    # "set a soup timer for 2 minutes" → "soup"
    m = re.search(r"(?:set\s+a?\s+)?(\w+)\s+timer\s+for", text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().lower()
        if candidate not in {"a", "the", "timer", "countdown", "an"}:
            return candidate

    # "timer for cooking for 5 minutes"
    m = re.search(r"timer\s+for\s+([a-z0-9 ]+?)\s+(?:for\s+)?\d", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # "for laundry timer"
    m = re.search(r"for\s+([a-z0-9 ]+?)\s+timer", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return None


def _extract_weather_city(text: str) -> str | None:
    patterns = [
        r"(?:weather|forecast|temperature)\s+(?:in|for)\s+([a-zA-Z .'-]+?)(?:\s+(?:today|tomorrow|on\s+\d{4}-\d{2}-\d{2}))?$",
        r"(?:does it rain|is it raining|will it rain|is it sunny|is it snowing|does it snow)\s+in\s+([a-zA-Z .'-]+?)(?:\s+(?:today|tomorrow|on\s+\d{4}-\d{2}-\d{2}))?$",
        r"(?:what is the weather in|what's the weather in)\s+([a-zA-Z .'-]+?)(?:\s+(?:today|tomorrow|on\s+\d{4}-\d{2}-\d{2}))?$",
        r"\bin\s+([A-Z][a-zA-Z .'-]+?)(?:\s+(?:today|tomorrow))?(?:\s*[?!.])?$",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" ,?.!")
    return None


def _extract_weather_day(text: str) -> str | None:
    if "tomorrow" in text:
        return "tomorrow"
    if "today" in text:
        return "today"
    m = re.search(r"\bon\s+(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        return m.group(1)
    return None


def _extract_weather_condition(text: str) -> str | None:
    t = text.lower()
    for cond in _WEATHER_CONDITIONS:
        if re.search(r"\b" + cond + r"\b", t):
            return cond
    return None


def _extract_original_request(text: str) -> str | None:
    """
    Strip modal verb prefixes from out-of-scope text to get the core request.
    "can you buy groceries for me" → "buy groceries for me"
    "please call my mom" → "call my mom"
    """
    cleaned = _OOS_MODAL_PREFIXES.sub("", text.strip())
    return cleaned.strip(" .?!") if cleaned else None


# ── Public API ────────────────────────────────────────────────

def fill_slots(text: str, intent: str) -> dict:
    """Extract slots for a given intent using regex rules."""
    t = normalize_text(text)
    slots: dict = {}

    if intent in {"load_dataset", "search_dataset", "get_dataset_info"}:
        dataset = _extract_dataset_phrase(t)
        if dataset:
            if intent == "search_dataset":
                slots["query"] = dataset
            else:
                slots["dataset"] = dataset

    if intent in {"load_code", "search_code"}:
        dataset = _extract_dataset_phrase(t, for_code=True)
        if dataset:
            if intent == "search_code":
                slots["query"] = dataset
            else:
                slots["dataset"] = dataset

    if intent == "show_leaderboard":
        m = re.search(r"leaderboard\s+(.+)$", t)
        if m:
            slots["query"] = m.group(1).strip(" ,:-")

    if intent == "select_model":
        for k, v in MODEL_KEYWORDS.items():
            if k in t:
                slots["model"] = v
                break

    if intent == "set_learning_rate":
        num = _extract_first_number(t)
        if num is not None:
            slots["learning_rate"] = float(num)

    if intent == "set_batch_size":
        num = _extract_first_number(t)
        if num is not None:
            slots["batch_size"] = int(float(num))

    if intent == "set_epochs":
        num = _extract_first_number(t)
        if num is not None:
            slots["epochs"] = int(float(num))

    if intent == "set_layers":
        num = _extract_first_number(t)
        if num is not None:
            slots["layers"] = int(float(num))

    if intent == "set_activation":
        for act in ["relu", "sigmoid", "tanh", "softmax", "linear"]:
            if act in t:
                slots["activation"] = act
                break

    if intent == "split_dataset":
        num = _extract_first_number(t)
        if num is not None:
            val = float(num)
            if val > 1.0:
                val = val / 100.0
            slots["ratio"] = val

    if intent in {"set_timer", "add_time_to_timer"}:
        duration = _extract_timer_duration_seconds(t)
        if duration is not None:
            slots["duration_seconds"] = duration

    if intent == "set_timer":
        label = _extract_timer_label(t)
        if label:
            slots["label"] = label

    if intent == "get_weather":
        city = _extract_weather_city(text)       # use original text (preserves capitalisation)
        if city:
            slots["city"] = city
        day = _extract_weather_day(t)
        if day:
            slots["day"] = day
        condition = _extract_weather_condition(text)  # NEW
        if condition:
            slots["weather_condition"] = condition

    # NEW: out_of_scope — extract original_request from text
    if intent == "out_of_scope":
        orig = _extract_original_request(text)
        if orig:
            slots["original_request"] = orig

    return slots